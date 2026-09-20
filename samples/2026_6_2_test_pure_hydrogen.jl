import Pkg
Pkg.activate(".")

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW
using SparseArrays
using SpecialFunctions
using Printf

using Base.Threads: @threads, threadid, nthreads

using SpecialFunctions
using HypergeometricFunctions

# store the dipole results
_write_complex(h, name::String, x) = begin
    h[name * "/real"] = real.(x)
    h[name * "/imag"] = imag.(x)
end

_read_complex(h, name::String) = read(h[name * "/real"]) .+ im .* read(h[name * "/imag"])


# ------------------------------------------------------------
# α = (n,l,m) -> linear index
# This is exactly the mapping specified by the user.
# ------------------------------------------------------------

function get_eigen_label(n::Int, l::Int, m::Int)
    id = get_index_from_lm(l, m, n)
    id < 0 && error("Invalid quantum numbers: n=$n, l=$l, m=$m")
    return id + (n - 1) * n * (2n - 1) ÷ 6
end

# ------------------------------------------------------------
# Bound hydrogen radial function u_nl(r)=r R_nl(r)
# normalized by ∫ |u_nl(r)|^2 dr = 1.
# ------------------------------------------------------------

function laguerreL(p::Int, a::Int, x::Float64)
    p == 0 && return 1.0

    L0 = 1.0
    L1 = 1.0 + a - x
    p == 1 && return L1

    for q in 1:(p - 1)
        L2 = ((2q + 1 + a - x) * L1 - (q + a) * L0) / (q + 1)
        L0, L1 = L1, L2
    end

    return L1
end

function hydrogen_u(n::Int, l::Int, Z::Float64, r::Float64)
    ρ = 2Z * r / n
    p = n - l - 1

    c = 2 * Z^(3/2) / n^2 *
        exp(0.5 * (loggamma(n - l) - loggamma(n + l + 1)))

    R = c * exp(-ρ / 2) * ρ^l * laguerreL(p, 2l + 1, ρ)

    return r * R
end

# ------------------------------------------------------------
# Spherical harmonics Y_lm(θ,φ).
# ------------------------------------------------------------

function assoc_legendre(l::Int, m::Int, x::Float64)
    m < 0 && error("Use m >= 0 here.")
    abs(m) > l && return 0.0

    pmm = 1.0

    if m > 0
        s = sqrt(max(0.0, 1.0 - x^2))
        fact = 1.0

        for _ in 1:m
            pmm *= -fact * s
            fact += 2.0
        end
    end

    l == m && return pmm

    pmmp1 = x * (2m + 1) * pmm
    l == m + 1 && return pmmp1

    p0, p1 = pmm, pmmp1

    for ll in (m + 2):l
        p2 = ((2ll - 1) * x * p1 - (ll + m - 1) * p0) / (ll - m)
        p0, p1 = p1, p2
    end

    return p1
end

function Ylm(l::Int, m::Int, θ::Float64, φ::Float64 = 0.0)
    abs(m) > l && return 0.0 + 0.0im

    if m < 0
        s = isodd(-m) ? -1.0 : 1.0
        return s * conj(Ylm(l, -m, θ, φ))
    end

    x = cos(θ)
    P = assoc_legendre(l, m, x)

    c = sqrt(
        (2l + 1) / (4π) *
        exp(loggamma(l - m + 1) - loggamma(l + m + 1))
    )

    return c * P * cis(m * φ)
end

# ------------------------------------------------------------
# Angular matrix element:
# <Y_l1,m1 | cosθ | Y_l2,m2>
# ------------------------------------------------------------

function z_ang(l1::Int, m1::Int, l2::Int, m2::Int)
    m1 == m2 || return 0.0

    m = m2

    if l1 == l2 + 1
        return sqrt(((l2 + 1)^2 - m^2) / ((2l2 + 1) * (2l2 + 3)))
    elseif l1 == l2 - 1 && l2 > 0
        return sqrt((l2^2 - m^2) / ((2l2 - 1) * (2l2 + 1)))
    else
        return 0.0
    end
end

# ------------------------------------------------------------
# Coulomb continuum radial wave.
#
# u_kl(r) = sqrt(2/pi) F_l(η,kr), η = -Z/k
# where F_l is the regular Coulomb radial function.
# ------------------------------------------------------------

function coulomb_sigma(l::Int, η::Float64)
    return imag(loggamma(complex(l + 1, η)))
end

function coulombF(l::Int, η::Float64, ρ::Float64)
    ρ == 0.0 && return 0.0

    C = 2.0^l *
        exp(
            -π * η / 2 +
            real(loggamma(complex(l + 1, η))) -
            loggamma(2l + 2)
        )

    # Denominator parameter must remain real/integer-like.
    M = pFq((complex(l + 1, -η),), (2l + 2,), 2im * ρ)

    return real(C * ρ^(l + 1) * exp(-1im * ρ) * M)
end

function continuum_u(l::Int, k::Float64, Z::Float64, r::Float64)
    η = -Z / k
    return sqrt(2 / π) * coulombF(l, η, k * r)
end

# ------------------------------------------------------------
# Numerical integration helpers.
# ------------------------------------------------------------

function trap_weights(rgrid::Vector{Float64})
    length(rgrid) >= 2 || error("rgrid must contain at least two points.")

    dr = rgrid[2] - rgrid[1]
    w = fill(dr, length(rgrid))

    w[1] *= 0.5
    w[end] *= 0.5

    return w
end

function radial_int(w, rgrid, u1, u2)
    s = 0.0

    @inbounds for i in eachindex(rgrid)
        s += w[i] * u1[i] * rgrid[i] * u2[i]
    end

    return s
end

function is_uniform_grid(x::AbstractVector{<:Real})
    length(x) < 3 && return true

    dx = x[2] - x[1]

    return all(
        abs((x[i + 1] - x[i]) - dx) <= 1e-10 * max(1.0, abs(dx))
        for i in 1:(length(x) - 1)
    )
end

# ------------------------------------------------------------
# Fast column dot:
#   sum_i A[i, ca] * B[i, cb]
#
# Used for:
#   <uB_α | r | uB_β>  with B = weighted uB
#   <uC_L | r | uB_α>  with B = weighted uB
# ------------------------------------------------------------

@inline function col_dot(
    A::AbstractMatrix{Float64},
    B::AbstractMatrix{Float64},
    ca::Int,
    cb::Int,
    Nr::Int,
)
    s = 0.0

    @inbounds @simd for i in 1:Nr
        s += A[i, ca] * B[i, cb]
    end

    return s
end


# ------------------------------------------------------------
# Fill one bound radial column.
#
# This avoids recalculating the n,l-dependent normalization
# constant for every r point.
# ------------------------------------------------------------

function fill_hydrogen_u_col!(
    dest::AbstractVector{Float64},
    n::Int,
    l::Int,
    Z::Float64,
    rgrid::AbstractVector{Float64},
)
    p = n - l - 1

    c = 2 * Z^(3/2) / n^2 *
        exp(0.5 * (loggamma(n - l) - loggamma(n + l + 1)))

    @inbounds @simd for i in eachindex(rgrid)
        r = rgrid[i]
        ρ = 2Z * r / n
        R = c * exp(-ρ / 2) * ρ^l * laguerreL(p, 2l + 1, ρ)
        dest[i] = r * R
    end

    return nothing
end


# ------------------------------------------------------------
# Fill all continuum radial columns for a fixed k:
#
#   uC[i, L+1] = sqrt(2/pi) F_L(η, k r_i)
#
# The expensive L,k-dependent constants are computed once per L,
# not once per r point.
# ------------------------------------------------------------

function fill_continuum_u_cols!(
    uC::Matrix{Float64},
    phase::Vector{ComplexF64},
    rgrid::AbstractVector{Float64},
    k::Float64,
    Z::Float64,
    Lmax::Int,
    logγ_den::Vector{Float64},
    pow2L::Vector{Float64},
)
    η = -Z / k
    c_norm = sqrt(2 / π)

    @inbounds for L in 0:Lmax
        col = L + 1

        lg = loggamma(complex(L + 1, η))

        C = pow2L[col] *
            exp(-π * η / 2 + real(lg) - logγ_den[col])

        scale = c_norm * C

        # Same phase convention as the original code:
        # phase[L+1] = (-i)^L exp(i σ_L) / k
        phase[col] = (-1im)^L * exp(1im * imag(lg)) / k

        a = (complex(L + 1, -η),)
        b = (2L + 2,)

        for i in eachindex(rgrid)
            ρ = k * rgrid[i]

            if ρ == 0.0
                uC[i, col] = 0.0
            else
                M = pFq(a, b, 2im * ρ)
                uC[i, col] = real(scale * ρ^(L + 1) * cis(-ρ) * M)
            end
        end
    end

    return nothing
end


# ------------------------------------------------------------
# Precompute all needed spherical harmonics:
#
#   Ycache[L+1][ith, m+L+1] = Y_Lm(θ_ith, kφ)
#
# This removes repeated Ylm calls inside the k loop.
# ------------------------------------------------------------

function precompute_Ylm_cache(
    Lmax::Int,
    thetagrid::AbstractVector{Float64},
    kφ::Float64,
)
    Nθ = length(thetagrid)
    Ycache = Vector{Matrix{ComplexF64}}(undef, Lmax + 1)

    @inbounds for L in 0:Lmax
        Y = Matrix{ComplexF64}(undef, Nθ, 2L + 1)

        for m in -L:L
            mi = m + L + 1

            for ith in 1:Nθ
                Y[ith, mi] = Ylm(L, m, thetagrid[ith], kφ)
            end
        end

        Ycache[L + 1] = Y
    end

    return Ycache
end


# ------------------------------------------------------------
# Optimized main function.
#
# External interface and return format are unchanged:
#
#   hydrogen_dipole_matrices(
#       nmax,
#       kgrid,
#       thetagrid;
#       dr,
#       rmax,
#       Z = 1.0,
#       kphi = 0.0,
#   )
#
# returns:
#
#   (Dbb = Dbb, Dbc = Dbc, states = states, rgrid = rgrid)
# ------------------------------------------------------------

function hydrogen_dipole_matrices(
    nmax::Int,
    kgrid::AbstractVector{<:Real},
    thetagrid::AbstractVector{<:Real};
    dr::Real,
    rmax::Real,
    Z::Real = 1.0,
    kphi::Real = 0.0,
    only_m0_bound::Bool = false,
)
    nmax >= 1 || error("nmax must be >= 1.")
    is_uniform_grid(kgrid) || error("kgrid must be equally spaced.")
    is_uniform_grid(thetagrid) || error("thetagrid must be equally spaced.")

    kvals = Float64.(kgrid)
    θvals = Float64.(thetagrid)

    all(k -> k > 0.0, kvals) || error("All k values must be positive.")

    Zf = Float64(Z)
    kφ = Float64(kphi)

    rgrid = collect(0.0:Float64(dr):Float64(rmax))
    w = trap_weights(rgrid)

    Na = nmax * (nmax + 1) * (2nmax + 1) ÷ 6
    Nr = length(rgrid)
    Nk = length(kvals)
    Nθ = length(θvals)

    # --------------------------------------------------------
    # State table.
    # states[α] keeps the original return format.
    # n_of[α], l_of[α], m_of[α] are faster hot-loop accessors.
    # --------------------------------------------------------

    states = Vector{Tuple{Int,Int,Int}}(undef, Na)

    n_of = Vector{Int}(undef, Na)
    l_of = Vector{Int}(undef, Na)
    m_of = Vector{Int}(undef, Na)

    @inbounds for n in 1:nmax, l in 0:(n - 1), m in -l:l
        α = get_eigen_label(n, l, m)

        states[α] = (n, l, m)
        n_of[α] = n
        l_of[α] = l
        m_of[α] = m
    end

    # --------------------------------------------------------
    # Bound radial functions.
    #
    # uB[i, α] = u_{nα,lα}(r_i)
    # WUB[i, α] = w_i * r_i * uB[i, α]
    #
    # Then radial integrals are simple column dot products.
    # --------------------------------------------------------

    uB = Matrix{Float64}(undef, Nr, Na)

    @inbounds for α in 1:Na
        fill_hydrogen_u_col!(
            view(uB, :, α),
            n_of[α],
            l_of[α],
            Zf,
            rgrid,
        )
    end

    wr = Vector{Float64}(undef, Nr)

    @inbounds @simd for i in 1:Nr
        wr[i] = w[i] * rgrid[i]
    end

    WUB = Matrix{Float64}(undef, Nr, Na)

    @inbounds for α in 1:Na
        for i in 1:Nr
            WUB[i, α] = wr[i] * uB[i, α]
        end
    end

    # --------------------------------------------------------
    # Precompute nonzero bound-bound angular couplings.
    # --------------------------------------------------------

    bb_pairs = Vector{Tuple{Int,Int,Float64}}()

    @inbounds for α in 1:Na, β in 1:Na
        A = z_ang(l_of[α], m_of[α], l_of[β], m_of[β])
        A == 0.0 && continue
        push!(bb_pairs, (α, β, A))
    end

    Dbb = zeros(ComplexF64, Na, Na)

    @threads :static for p in eachindex(bb_pairs)
        α, β, A = bb_pairs[p]
        I = col_dot(uB, WUB, α, β, Nr)
        Dbb[α, β] = A * I
    end

    # --------------------------------------------------------
    # Bound-continuum part.
    # --------------------------------------------------------

    Dbc = [zeros(ComplexF64, Nk, Nθ) for _ in 1:Na]

    # Since bound l <= nmax-1 and z couples l -> L=l±1,
    # the largest needed continuum partial wave is Lmax=nmax.
    Lmax = nmax

    # Precompute angular factors for every allowed L,m,θ.
    Ycache = precompute_Ylm_cache(Lmax, θvals, kφ)

    # Precompute nonzero bound-continuum coupling channels:
    #
    #   α, L, m, A = <Y_Lm | cosθ | Y_lm>
    #
    # Each bound state has at most two L channels: l-1 and l+1.
    bc_pairs = Vector{Tuple{Int,Int,Int,Float64}}()

    @inbounds for α in 1:Na
        l = l_of[α]
        m = m_of[α]
    
        # Optional restriction:
        # only compute bound-continuum matrix elements whose bound state has m = 0.
        # Dbc[α] for m != 0 remains the zero matrix initialized above.
        if only_m0_bound && m != 0
            continue
        end
    
        for L in (l - 1, l + 1)
            (L < 0 || L > Lmax || abs(m) > L) && continue
    
            A = z_ang(L, m, l, m)
            A == 0.0 && continue
    
            push!(bc_pairs, (α, L, m, A))
        end
    end

    # Constants used in continuum normalization.
    logγ_den = [loggamma(2L + 2) for L in 0:Lmax]
    pow2L = [2.0^L for L in 0:Lmax]

    # Thread-local scratch buffers.
    # Each thread owns its own uC and phase arrays, so no race occurs.
    caches = [
        (
            uC = Matrix{Float64}(undef, Nr, Lmax + 1),
            phase = Vector{ComplexF64}(undef, Lmax + 1),
        )
        for _ in 1:nthreads()
    ]

    @threads :static for ik in 1:Nk
        tid = threadid()
        cache = caches[tid]

        uC = cache.uC
        phase = cache.phase

        k = kvals[ik]

        fill_continuum_u_cols!(
            uC,
            phase,
            rgrid,
            k,
            Zf,
            Lmax,
            logγ_den,
            pow2L,
        )

        @inbounds for q in eachindex(bc_pairs)
            α, L, m, A = bc_pairs[q]

            Lp1 = L + 1
            I = col_dot(uC, WUB, Lp1, α, Nr)

            pref = phase[Lp1] * A * I

            Dα = Dbc[α]
            Y = Ycache[Lp1]
            mi = m + L + 1

            @simd for ith in 1:Nθ
                Dα[ik, ith] += pref * Y[ith, mi]
            end
        end
    end

    return (Dbb = Dbb, Dbc = Dbc, states = states, rgrid = rgrid)
end


nmax = 5

p_max = 5.0
p_min = 0.02
Np = 1000 * 5 ÷ 2
Δp = (p_max - p_min) / Np
N_theta = 180
Δtheta = π / N_theta
p_grid = [p_min + (i - 1 + 0.5) * Δp for i = 1: Np]                 # use mid point grid
theta_grid = [(i - 1 + 0.5) * Δtheta for i = 1: N_theta]

# res = hydrogen_dipole_matrices(
#     nmax,
#     p_grid,
#     theta_grid;
#     dr = 0.02,
#     rmax = 200.0,
#     Z = 1.0,
#     only_m0_bound = true,
# )

# dipole_z_bb = res.Dbb
# dipole_z_cb = res.Dbc
# states = res.states

# example_name = "2026_6_2_precalc"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "dipole_z_cb", hcat(dipole_z_cb...))
#     write(file, "dipole_z_bb", dipole_z_bb)
# end

example_name = "2026_6_2_precalc"
dipole_z_cb_hcat = retrieve_mat(example_name, "dipole_z_cb")
dipole_z_bb = retrieve_mat(example_name, "dipole_z_bb")
dipole_z_cb = collect(eachslice(reshape(dipole_z_cb_hcat, size(dipole_z_cb_hcat,1), N_theta, :), dims=3))

α = get_eigen_label(1, 0, 0)
des = norm.(dipole_z_cb[α])
heatmap([theta_grid; theta_grid .+ pi], p_grid[1: end ÷ 1], [des des[:, end:-1:1]][1: end ÷ 1, :], projection=:polar, color=:cork)


# α = get_eigen_label(1, 0, 0)
# pseudo_dipole_res = h5open("./data/2026_5_27_dipole_z_cb.h5", "r") do h
#     _read_complex(h, "dipole_z_cb_$α")
# end
# des1 = norm.(dipole_z_cb[α])
# des2 = norm.(pseudo_dipole_res)
# plot(p_grid[1: end ÷ 4], [des1[1: end ÷ 4, end ÷ 1], des2[1: end ÷ 4, end ÷ 1]])




##########################################

n_upper_limit = nmax

# calculate eigen_states for m = 0 special case
struct eigen_state_t
    n::Int64
    l::Int64
    Ip::Float64
    data::Vector{ComplexF64}
end

eigen_states = Vector{eigen_state_t}(undef, get_eigen_label(n_upper_limit, n_upper_limit - 1, 0))
alpha_list = Int64[]

for n in 1: n_upper_limit
    for l in 0: 0 + n - 1
        m = 0
        α = get_eigen_label(n, l, m)
        Ip = 0.5 / n ^ 2
        eigen_states[α] = eigen_state_t(n, l, Ip, ComplexF64[])
        push!(alpha_list, α)
    end
end

N_alpha = length(eigen_states)




##############################################################


# create kappa grid (Spherical)
kappa_p_max = 2.0
kappa_delta_p = Δp
kappa_N_p = floor(Int64, kappa_p_max / kappa_delta_p)
kappa_N_theta = N_theta
kappa_delta_theta = π / kappa_N_theta
kappa_p_subgrid = [(i - 0.5) * kappa_delta_p for i in 1: kappa_N_p]
kappa_theta_subgrid = [(q - 0.5) * kappa_delta_theta for q in 1: kappa_N_theta]
N_kappa = length(kappa_p_subgrid) * length(kappa_theta_subgrid)
kappa_grid_p = zeros(Float64, N_kappa)
kappa_grid_theta = zeros(Float64, N_kappa)
kappa_grid_x = zeros(Float64, N_kappa)
kappa_grid_y = zeros(Float64, N_kappa)
kappa_grid_z = zeros(Float64, N_kappa)

k = 1
for κ_p in kappa_p_subgrid, κ_theta in kappa_theta_subgrid
    kappa_grid_p[k] = κ_p
    kappa_grid_theta[k] = κ_theta
    kappa_grid_x[k], kappa_grid_y[k], kappa_grid_z[k] = sphere_to_xyz(κ_p, κ_theta, 0.0)
    k += 1
end

# define laser field
@expo E_fs =          0.02              # peak electric field of the fs pulse
@expo ω_fs =          0.057 * 1.0        # angular frequency of the fs pulse
@expo nc =            6                 # number of optical cycles in the fs pulse
Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, 0, ellipticity=0.0, phase1=0.5pi)        # create the light pulse from the given parameters (+ ellipticity)
E_field(t) = Ex_fs(t)
Tp = 2 * nc * pi / ω_fs


# create time grid and auxiliaries for characteristic curves
Δt = 0.2
Nt = Int64(floor(Tp / Δt))
ts = [i * Δt for i in range(0, Nt - 1)]
Et_data = E_field.(ts)
eta_data = get_integral(Et_data, Δt)
eta_1_data = get_integral(eta_data, Δt)
eta_2_data = get_integral(eta_data .^ 2, Δt)

u_bound = zeros(ComplexF64, N_alpha)
u_free = zeros(ComplexF64, N_kappa)
p_curves_x = zeros(Float64, N_kappa)
p_curves_y = zeros(Float64, N_kappa)
p_curves_z = zeros(Float64, N_kappa)
p_curves_p = zeros(Float64, N_kappa)
p_curves_theta = zeros(Float64, N_kappa)
chi_curves = zeros(Float64, N_kappa)



# pre-judgement
function pre_judgement(
    p_curves_p, p_curves_theta, chi_curves,
    kappa_grid_p, kappa_grid_theta, N_kappa,
    Δp, Δtheta, coarse_dipole_z_cb,
    Et_data, eta_data, eta_1_data, eta_2_data, 
    eigen_states, N_theta, ts,
    kappa_included_threshold = 1e-3
)
    coupling_amplitude = zeros(ComplexF64, N_kappa) 
    dt = ts[2] - ts[1]

    @time for (it, t) in enumerate(ts)
        t0 = 0.0
        # @. p_curves_x = kappa_grid_x
        # @. p_curves_y = kappa_grid_y
        # @. p_curves_z = kappa_grid_z + eta_data[it]
        # @. chi_curves = (kappa_grid_x ^ 2 + kappa_grid_y ^ 2 + kappa_grid_z ^ 2) * (t - t0) / 2 + kappa_grid_z * eta_1_data[it] + eta_2_data[it] / 2
        
        @. p_curves_p = sqrt(kappa_grid_p ^ 2 + 2 * eta_data[it] * kappa_grid_p * cos(kappa_grid_theta) + eta_data[it] ^ 2)
        @. p_curves_theta = acos((kappa_grid_p * cos(kappa_grid_theta) + eta_data[it]) / (p_curves_p + 1e-8))
        @. chi_curves = (kappa_grid_p ^ 2) * (t - t0) / 2 + kappa_grid_p * cos(kappa_grid_theta) * eta_1_data[it] + eta_2_data[it] / 2  

        Threads.@threads for i in 1: N_kappa
            p_id::Int64 = 0
            theta_id::Int64 = 0

            # interp to a certain position on (p_abs, θ, ϕ) grid (namely pgrid aforehead)
            p_id = floor(Int64, (p_curves_p[i] - 0.0) / Δp) + 1
            theta_id = floor(Int64, (p_curves_theta[i] - 0.0) / Δtheta) + 1

            if theta_id == N_theta + 1
                theta_id = N_theta
            end

            α = 1       # let alpha = 1 (because it has the largest dipole-coupling area in p space)
            amp = abs(Et_data[it] * coarse_dipole_z_cb[α][p_id, theta_id])
            coupling_amplitude[i] += dt * im * Et_data[it] * coarse_dipole_z_cb[α][p_id, theta_id] * exp(im * eigen_states[α].Ip * t) * exp(im * chi_curves[i])
        end

        if it % 100 == 0
            println("step $it")
        end
    end
    return coupling_amplitude
end

# coupling_amplitude = pre_judgement(p_curves_p, p_curves_theta, chi_curves,
#     kappa_grid_p, kappa_grid_theta, N_kappa, Δp, Δtheta, dipole_z_cb, Et_data, eta_data, eta_1_data, eta_2_data, eigen_states, N_theta, ts[1: (length(ts))])

# # get a 2D slice of coupling_amplitude in x-z plane (ϕ = 0) (Spherical)
# coupling_amplitude_xz = zeros(Float64, length(kappa_p_subgrid), length(kappa_theta_subgrid))
# for i in 1: length(kappa_p_subgrid), j in 1: length(kappa_theta_subgrid)
#     kappa_index = (i - 1) * length(kappa_theta_subgrid) + j
#     coupling_amplitude_xz[i, j] = real(norm(coupling_amplitude[kappa_index]) ^ 2)
# end

# colormap = cgrad([:white, palette(:jet1, 6)...], rev = true)
# display_data = clamp.(abs.(log10.(norm.(coupling_amplitude_xz) ./ maximum(norm.(coupling_amplitude_xz)))), 0, 3.0)
# p1_heat = heatmap([kappa_theta_subgrid; π .+ kappa_theta_subgrid], kappa_p_subgrid, [display_data display_data[:, end:-1:1]], projection=:polar, color=colormap)

# colormap_2 = cgrad(["#710000", "#C60001", "#FB0001", "#FE3A01", "#FF7A00", "#FEAB01", "#FEAB01", "#FEAB01", "#D5FF17", "#91FF60", "#4AFFAE", "#02FFFC", "#00A3FE", "#003DFF", "#2120FF", "#8C8CFE", :white], rev=true)
# p1 = heatmap([kappa_theta_subgrid .+ π/2; kappa_theta_subgrid .+ 3π/2], kappa_p_subgrid[1: end ÷ 2], [coupling_amplitude_xz coupling_amplitude_xz[:, end:-1:1]][1: end ÷ 2, :], projection=:polar, color=colormap_2)




#############################################################
# SFA calculation

struct sfa_buffer_t
    N_alpha::Int64
    alpha_list::Vector{Int64}
    eigen_states::Vector{eigen_state_t}
    kappa_grid_p::Vector{Float64}
    kappa_grid_theta::Vector{Float64}

    N_kappa::Int64
    kappa_delta_p::Float64
    kappa_delta_theta::Float64    # kappa grid

    p_grid_delta::Float64
    theta_delta::Float64
    Et_data::Vector{Float64}
    eta_data::Vector{Float64}
    eta_1_data::Vector{Float64}
    eta_2_data::Vector{Float64}           # Et_data & its derivative
    coarse_dipole_z_cb::Vector{Matrix{ComplexF64}}
    dipole_z_bb::Matrix{ComplexF64}                                    # precompute results
    p_curves_p::Vector{Float64}
    p_curves_theta::Vector{Float64}
    chi_curves::Vector{Float64}                          # runtime buffer
    chunk_buffer::Vector{ComplexF64}
    chunk_sum_buffer::Vector{ComplexF64}
end

function sfa_rhs(
    u::Vector{ComplexF64},
    res_u::Vector{ComplexF64},    # res: the destination arrays
    t::Float64, it::Int64,        # time
    bf::sfa_buffer_t              # sfa buffer
    )

    t0::Float64 = 0.0
    fill!(res_u, 0.0 + 0.0im)

    # calculate the curves
    @. bf.p_curves_p = sqrt(bf.kappa_grid_p ^ 2 + 2 * bf.eta_data[it] * bf.kappa_grid_p * cos(bf.kappa_grid_theta) + bf.eta_data[it] ^ 2)
    @. bf.p_curves_theta = acos((bf.kappa_grid_p * cos(bf.kappa_grid_theta) + bf.eta_data[it]) / (bf.p_curves_p + 1e-8))
    @. bf.chi_curves = (bf.kappa_grid_p ^ 2) * (t - t0) / 2 + bf.kappa_grid_p * cos(bf.kappa_grid_theta) * bf.eta_1_data[it] + bf.eta_2_data[it] / 2  

    # update part 1
    # @time begin
    # for i in 1: N_kappa

    Threads.@threads for i in 1: bf.N_kappa
        p_id = floor(Int64, (bf.p_curves_p[i] - 0.0) / bf.p_grid_delta) + 1
        theta_id = floor(Int64, (bf.p_curves_theta[i] - 0.0) / bf.theta_delta) + 1
        res_u[bf.N_alpha + i] = 0.0
        w = 2 * pi * bf.kappa_grid_p[i] ^ 2 * sin(bf.kappa_grid_theta[i]) * bf.kappa_delta_p * bf.kappa_delta_theta
        for α in bf.alpha_list
            res_u[bf.N_alpha + i] += bf.coarse_dipole_z_cb[α][p_id, theta_id] * u[α] * exp(im * bf.eigen_states[α].Ip * t)
        end
        res_u[bf.N_alpha + i] *= im * bf.Et_data[it] * exp(im * bf.chi_curves[i]) * sqrt(w)
    end
    # end

    # @time begin
    # update part 3
    # for α in alpha_list

    nt = Threads.nthreads()
    chunk_size = cld(bf.N_kappa, nt)
    for α in bf.alpha_list
        res_u[α] = 0.0
        Threads.@threads for j in 1: nt
            start_idx = (j - 1) * chunk_size + 1
            end_idx = min(j * chunk_size, bf.N_kappa)
            for i in start_idx: end_idx
                p_id = floor(Int64, (bf.p_curves_p[i] - 0.0) / bf.p_grid_delta) + 1
                theta_id = floor(Int64, (bf.p_curves_theta[i] - 0.0) / bf.theta_delta) + 1
                w = 2 * pi * bf.kappa_grid_p[i] ^ 2 * sin(bf.kappa_grid_theta[i]) * bf.kappa_delta_p * bf.kappa_delta_theta
                bf.chunk_buffer[i] = im * bf.Et_data[it] * sqrt(w) * conj(bf.coarse_dipole_z_cb[α][p_id, theta_id]) * exp(-im * bf.eigen_states[α].Ip * t) * u[bf.N_alpha + i] * exp(-im * bf.chi_curves[i])
            end
            bf.chunk_sum_buffer[j] = sum(@views bf.chunk_buffer[start_idx: end_idx])
        end
        res_u[α] = sum(bf.chunk_sum_buffer[1: nt])
    end
    
    for α in bf.alpha_list
        for β in bf.alpha_list
            res_u[α] += im * bf.Et_data[it] * bf.dipole_z_bb[α, β] * exp(im * (bf.eigen_states[β].Ip - bf.eigen_states[α].Ip) * t) * u[β]
        end
    end
end


function rk4_one_step(dt_half::Float64,
    u::Vector{ComplexF64},
    k1::Vector{ComplexF64},    # res: the destination arrays
    k2::Vector{ComplexF64},
    k3::Vector{ComplexF64},
    k4::Vector{ComplexF64},
    buffer::Vector{ComplexF64},
    
    t::Float64, it::Int64,
    bf::sfa_buffer_t
    )
    h = 2 * dt_half

    sfa_rhs(u, k1, t, it, bf)
    
    @. buffer = u + (h / 2) * k1
    sfa_rhs(buffer, k2, t + dt_half, it + 1, bf)

    @. buffer = u + (h / 2) * k2
    sfa_rhs(buffer, k3, t + dt_half, it + 1, bf)
    
    @. buffer = u + (h) * k3
    sfa_rhs(buffer, k4, t + h, it + 2, bf)

    @. u += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end



###########################

# init state
u = zeros(ComplexF64, N_alpha + N_kappa)
u_buffer = [zeros(ComplexF64, N_alpha + N_kappa) for i = 1: 5]
u[1] = 1.0

chunk_buffer = zeros(ComplexF64, N_kappa)
chunk_sum_buffer = zeros(ComplexF64, N_kappa)

##########################

alpha_list_selected = alpha_list[1: end]

sfa_buffer = sfa_buffer_t(N_alpha, alpha_list_selected,
    eigen_states, kappa_grid_p, kappa_grid_theta, N_kappa, kappa_delta_p, kappa_delta_theta,
    Δp, Δtheta, Et_data, eta_data, eta_1_data, eta_2_data, dipole_z_cb, dipole_z_bb, p_curves_p, p_curves_theta,
    chi_curves, chunk_buffer, chunk_sum_buffer)


# RK4 propagation
bf = sfa_buffer
t0 = 0.0
nt = Threads.nthreads()
chunk_size = cld(bf.N_kappa, nt)

mainloop_ts = 1: 2: (length(ts) - 2)

D_BB = zeros(Float64, length(mainloop_ts))
D_BC = zeros(Float64, length(mainloop_ts))
D_CC = zeros(Float64, length(mainloop_ts))
D_total = zeros(Float64, length(mainloop_ts))

P_bound_sum = zeros(Float64, length(mainloop_ts))
P_1s = zeros(Float64, length(mainloop_ts))
P_2s = zeros(Float64, length(mainloop_ts))
P_2p = zeros(Float64, length(mainloop_ts))
P_free_sum = zeros(Float64, length(mainloop_ts))
P_total = zeros(Float64, length(mainloop_ts))

# pre-calculation for dipole
kappa_w = @. 2 * pi * bf.kappa_grid_p ^ 2 * sin(bf.kappa_grid_theta) * bf.kappa_delta_p * bf.kappa_delta_theta
kappa_sqrt_w = sqrt.(kappa_w)
kappa_inv_sqrt_w = 1.0 ./ kappa_sqrt_w
kappa_cos = cos.(bf.kappa_grid_theta)
kappa_sin = sin.(bf.kappa_grid_theta)
kappa_p_cos = bf.kappa_grid_p .* kappa_cos
kappa_sin_over_p = kappa_sin ./ bf.kappa_grid_p

p_id_buffer = zeros(Int64, bf.N_kappa)
theta_id_buffer = zeros(Int64, bf.N_kappa)

kp_r_buffer = zeros(Int64, bf.N_kappa)
km_r_buffer = zeros(Int64, bf.N_kappa)
kp_t_buffer = zeros(Int64, bf.N_kappa)
km_t_buffer = zeros(Int64, bf.N_kappa)
inv_Δκ_buffer = zeros(Float64, bf.N_kappa)
inv_Δθ_buffer = zeros(Float64, bf.N_kappa)

for ip in 1: kappa_N_p
    ip_p = min(ip + 1, kappa_N_p)
    ip_m = max(ip - 1, 1)

    for iq in 1: kappa_N_theta
        iq_p = min(iq + 1, kappa_N_theta)
        iq_m = max(iq - 1, 1)

        k = (ip - 1) * kappa_N_theta + iq

        kp_r_buffer[k] = (ip_p - 1) * kappa_N_theta + iq
        km_r_buffer[k] = (ip_m - 1) * kappa_N_theta + iq
        kp_t_buffer[k] = (ip - 1) * kappa_N_theta + iq_p
        km_t_buffer[k] = (ip - 1) * kappa_N_theta + iq_m

        inv_Δκ_buffer[k] = 1.0 / ((ip_p - ip_m) * bf.kappa_delta_p)
        inv_Δθ_buffer[k] = 1.0 / ((iq_p - iq_m) * bf.kappa_delta_theta)
    end
end

for (i, it) in enumerate(mainloop_ts)
    t = ts[it]

    rk4_one_step(Δt, u, u_buffer[1], u_buffer[2], u_buffer[3], u_buffer[4], u_buffer[5], t, it, bf);

    # after rk4_one_step, u corresponds to it + 2
    it2 = it + 2
    t = ts[it2]

    # calculate the curves; keep your original method
    @. bf.p_curves_p = sqrt(bf.kappa_grid_p ^ 2 + 2 * bf.eta_data[it2] * bf.kappa_grid_p * cos(bf.kappa_grid_theta) + bf.eta_data[it2] ^ 2)
    @. bf.p_curves_theta = acos((bf.kappa_grid_p * cos(bf.kappa_grid_theta) + bf.eta_data[it2]) / (bf.p_curves_p + 1e-8))
    @. bf.chi_curves = (bf.kappa_grid_p ^ 2) * (t - t0) / 2 + bf.kappa_grid_p * cos(bf.kappa_grid_theta) * bf.eta_1_data[it2] + bf.eta_2_data[it2] / 2  

    # pre-calculate p_id, theta_id, and sqrt(W) * C * exp(-iχ)
    Threads.@threads for k in 1: bf.N_kappa
        @inbounds begin
            p_id_buffer[k] = floor(Int64, bf.p_curves_p[k] / bf.p_grid_delta) + 1
            theta_id_buffer[k] = floor(Int64, bf.p_curves_theta[k] / bf.theta_delta) + 1
            bf.chunk_buffer[k] = kappa_sqrt_w[k] * u[bf.N_alpha + k] * cis(-bf.chi_curves[k])
        end
    end

    # absorption of bound states (only for test)
    for α in bf.alpha_list
        l = eigen_states[α].l
        # if l == 2
        #     u[α] *= 0.95
        # elseif l == 3
        #     u[α] *= 0.8
        # elseif l == 4
        #     u[α] *= 0.2
        # end
    end

    # D_BB
    for α in bf.alpha_list
        for β in bf.alpha_list
            @inbounds D_BB[i] += real(conj(u[α]) * u[β] *
                                      cis((bf.eigen_states[β].Ip - bf.eigen_states[α].Ip) * t) *
                                      bf.dipole_z_bb[α, β])
        end 
    end

    # D_BC
    for α in bf.alpha_list
        bf.chunk_sum_buffer[1: nt] .= 0.0 + 0.0im

        dipole_mat = bf.coarse_dipole_z_cb[α]
        Np_dipole = size(dipole_mat, 1)
        Nθ_dipole = size(dipole_mat, 2)

        Threads.@threads for j in 1: nt
            s = 0.0 + 0.0im
            start_idx = (j - 1) * chunk_size + 1
            end_idx = min(j * chunk_size, bf.N_kappa)

            for k in start_idx: end_idx
                @inbounds begin
                    p_id = p_id_buffer[k]

                    if 1 <= p_id <= Np_dipole
                        theta_id = min(max(theta_id_buffer[k], 1), Nθ_dipole)
                        s += conj(dipole_mat[p_id, theta_id]) * bf.chunk_buffer[k]
                    end
                end
            end

            bf.chunk_sum_buffer[j] = s
        end

        D_BC[i] += 2 * real(conj(u[α]) * cis(-bf.eigen_states[α].Ip * t) * sum(@views bf.chunk_sum_buffer[1: nt]))
    end

    # D_CC
    bf.chunk_sum_buffer[1:nt] .= 0.0 + 0.0im

    Threads.@threads for j in 1:nt
        s = 0.0 + 0.0im
        start_idx = (j - 1) * chunk_size + 1
        end_idx = min(j * chunk_size, bf.N_kappa)

        for k in start_idx:end_idx
            @inbounds begin
                kp_r = kp_r_buffer[k]
                km_r = km_r_buffer[k]
                kp_t = kp_t_buffer[k]
                km_t = km_t_buffer[k]

                Ck   = u[bf.N_alpha + k]    * kappa_inv_sqrt_w[k]
                Cr_p = u[bf.N_alpha + kp_r] * kappa_inv_sqrt_w[kp_r]
                Cr_m = u[bf.N_alpha + km_r] * kappa_inv_sqrt_w[km_r]
                Ct_p = u[bf.N_alpha + kp_t] * kappa_inv_sqrt_w[kp_t]
                Ct_m = u[bf.N_alpha + km_t] * kappa_inv_sqrt_w[km_t]

                dC_dκ = (Cr_p - Cr_m) * inv_Δκ_buffer[k]
                dC_dθ = (Ct_p - Ct_m) * inv_Δθ_buffer[k]

                dC_dκz = kappa_cos[k] * dC_dκ - kappa_sin_over_p[k] * dC_dθ

                dχ_dκz = kappa_p_cos[k] * (t - t0) + bf.eta_1_data[it2]

                s += kappa_w[k] * (im * conj(Ck) * dC_dκz + abs2(Ck) * dχ_dκz)
            end
        end
        bf.chunk_sum_buffer[j] = s
    end
    D_CC[i] = real(sum(@views bf.chunk_sum_buffer[1:nt]))
    
    # get total dipole
    D_total[i] = D_BB[i] + D_BC[i] + D_CC[i]

    # get population
    P_bound_sum[i] = sum(norm.(u[1: N_alpha]) .^ 2)
    P_1s[i] = norm(u[1]) ^ 2
    P_2s[i] = norm(u[2]) ^ 2
    P_2p[i] = norm(u[3]) ^ 2
    P_free_sum[i] = sum(norm.(u[N_alpha + 1: end]) .^ 2)
    P_total[i] = P_bound_sum[i] + P_free_sum[i]

    if it % 100 == 1
        println("it = $it")

        @printf "Total = %0.4f, 1s = %0.4f, 2s = %0.4f, 2p = %0.4f, Free = %0.4f\n"  P_total[i] P_1s[i] P_2s[i]  P_2p[i]  P_free_sum[i]
        @printf "D_BB = %+0.6e, D_BC = %+0.6e, D_CC = %+0.6e, D_total = %+0.6e\n" D_BB[i] D_BC[i] D_CC[i] D_total[i]
    end
end

# save everything
# example_name = "2026_6_2_$(E_fs)_$(ω_fs)_$(nc)_short_range"
example_name = "2026_6_2_$(E_fs)_$(ω_fs)_$(nc)"
h5open("./data/$example_name.h5", "w") do h
    write(h, "D_total", D_total)
    write(h, "D_CC", D_CC)
    write(h, "D_BB", D_BB)
    write(h, "D_BC", D_BC)
end

# example_name = "2026_6_2_$(E_fs)_$(ω_fs)_$(nc)"
# D_total = retrieve_mat(example_name, "D_total")
# D_CC = retrieve_mat(example_name, "D_CC")
# D_BB = retrieve_mat(example_name, "D_BB")
# D_BC = retrieve_mat(example_name, "D_BC")


plot(ts[mainloop_ts], [D_total, D_BB, D_BC, D_CC], label=["D_total" "D_BB" "D_BC" "D_CC"])

# get harmonic spectrum, including data, and k axis (frequency axis)
# n_cut_off_estim = floor((-en + 3.17 * (E_fs ^ 2.0 / (4.0 * (ω_fs ^ 2.0)))) / ω_fs) * 1
n_cut_off_estim = 20

hg1, ks = get_hg_spectrum(ts[mainloop_ts], D_total, ω_fs * (n_cut_off_estim + 20))
hg1_free, _ = get_hg_spectrum(ts[mainloop_ts], D_CC, ω_fs * (n_cut_off_estim + 20))
hg1_bound, _ = get_hg_spectrum(ts[mainloop_ts], D_BB, ω_fs * (n_cut_off_estim + 20))
hg1_cross, _ = get_hg_spectrum(ts[mainloop_ts], D_BC, ω_fs * (n_cut_off_estim + 20))

# r
p3 = plot(ks ./ ω_fs, [(ks .^ 3) .* hg1, (ks .^ 3) .* hg1_free, (ks .^ 3) .* hg1_bound, (ks .^ 3) .* hg1_cross],
    yscale=:log10, xaxis=1:20, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-8, 1e3),
    label=["Total" "Free" "Bound" "Cross"], xlabel="Harmonic Order", ylabel="HG Intensity", title="HHG Spectrum (E0=$E_fs, ω=$ω_fs)")


# get a 2D slice of u_free in x-z plane (ϕ = 0) (Spherical)
u_free_xz = zeros(ComplexF64, length(kappa_p_subgrid), length(kappa_theta_subgrid))
for i in 1: length(kappa_p_subgrid), j in 1: length(kappa_theta_subgrid)
    kappa_index = (i - 1) * length(kappa_theta_subgrid) + j
    kappa_p = kappa_p_subgrid[i]
    kappa_theta = kappa_theta_subgrid[j]
    w = (2 * pi * kappa_p^2) * sin(kappa_theta) * kappa_delta_p * kappa_delta_theta
    u_free_xz[i, j] = norm.(u[N_alpha + kappa_index]) .^ 2.0 / w
end
# colormap = cgrad([:white, :black, :red, :blue, :white])
colormap = cgrad([:white, palette(:jet1, 6)...], rev = true)
display_data = clamp.(abs.(log10.(norm.(u_free_xz) ./ maximum(norm.(u_free_xz)))), 0, 3.0)
# display_data = (display_data ./ 5.0) .^ 0.5 * 5.0
p2_heat = heatmap([kappa_theta_subgrid; π .+ kappa_theta_subgrid], kappa_p_subgrid, [display_data display_data[:, end:-1:1]], projection=:polar, color=colormap)

colormap_2 = cgrad(["#710000", "#C60001", "#FB0001", "#FE3A01", "#FF7A00", "#FEAB01", "#FEAB01", "#FEAB01", "#D5FF17", "#91FF60", "#4AFFAE", "#02FFFC", "#00A3FE", "#003DFF", "#2120FF", "#8C8CFE", :white], rev=true)
p2 = heatmap([kappa_theta_subgrid .+ π/2; kappa_theta_subgrid .+ 3π/2], kappa_p_subgrid[1: end ÷ 2], [norm.(u_free_xz) norm.(u_free_xz)[:, end:-1:1]][1: end ÷ 2, :], projection=:polar, color=colormap_2)


