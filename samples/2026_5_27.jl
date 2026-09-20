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
const dot_t = Tuple{Float64, Float64, Float64}

# store the dipole results
_write_complex(h, name::String, x) = begin
    h[name * "/real"] = real.(x)
    h[name * "/imag"] = imag.(x)
end

_read_complex(h, name::String) = read(h[name * "/real"]) .+ im .* read(h[name * "/imag"])


# Basic Parameters
rate = 1
Nr =            5000 * rate            # number of radial grid points
Δr =            0.2 / rate            # radial grid step size
l_num =         5               # number of angular momentum components
Δt =            0.2            # time step size
Z =             1.0             # nuclear charge
# po_func(r) =    -1 / r        # potential function
po_func(r) =    -1 / r * exp(- r * r / (5.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
# po_func(r) =    -1 / r * exp(- r * r / (20.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
rmax =          Nr * Δr     
# absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function
absorb_func    = r -> 0


# create pw, rt
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func);
rt = create_tdse_rt_sh(pw, m_zero_flag=true);
rs = get_linspace(pw.shgrid.rgrid)


# mapping eigen label (n, l) with α (suitable for all eigenstates)
function get_eigen_label(n, l, m=0)
    id = get_index_from_lm(l, m, n)
    return id + (n - 1) * (n) * (2*n - 1) ÷ 6
end

# calculate eigen_states for m = 0 special case
struct eigen_state_t
    n::Int64
    l::Int64
    Ip::Float64
    data::Vector{ComplexF64}
end


function build_radial_box_continuum(
    L::Int,
    rs::Vector{Float64},
    rt::JuliaProp.tdse_sh_rt,
    pmax::Float64,
)
    H = -0.5 .* rt.D2_boost .+ Diagonal(pw.po_data_r) .+ Diagonal(@. (L * (L + 1.0) / 2.0) * (1.0 / (rs ^ 2)))
    F = eigen(H)
    E = F.values
    U = F.vectors

    # keep positive-energy states up to pmax
    free_ids = findall(e -> e > 0.0 && sqrt(2.0 * e) <= pmax, E)
    bound_ids = findall(e -> e <= 0.0, E)
    p_free = sqrt.(2.0 .* E[free_ids])

    return E[bound_ids], U[:, bound_ids], p_free, U[:, free_ids]
end

#################################

pseudo_box_p_max = 5.5
eigen_state_list = eigen_state_t[]
phi_by_l = Matrix{Float64}[]
pval_by_l = Vector{Float64}[]
alpha_list = Int64[]

l = 0
n_upper_limit = 3
eigen_l_num_upper_limit = 3
eigen_max_l_num = 0
while true
    E_bound, phi_bound, p_free, phi_free = build_radial_box_continuum(l, rs, rt, pseudo_box_p_max)
    push!(phi_by_l, phi_free)
    push!(pval_by_l, p_free)
    if isempty(E_bound) || l > eigen_l_num_upper_limit - 1
        eigen_max_l_num = l
        println("eigen_max_l_num = $l, since then no bound states exist")
        break
    else
        for (i, e) in enumerate(E_bound)
            n = i + l
            if n > n_upper_limit
                break
            end
            println("For l = $l, n = $n, energy = $e")
            push!(eigen_state_list, eigen_state_t(n, l, -e, phi_bound[:, i]))
            push!(alpha_list, get_eigen_label(n, l))
        end
    end
    @printf "Finished l = %d \n" l
    l += 1
end

N_alpha = eigen_max_l_num * (eigen_max_l_num + 1) * (2 * eigen_max_l_num + 1) ÷ 6
eigen_states = Vector{eigen_state_t}(undef, N_alpha)
for (i, α) in enumerate(alpha_list)
    eigen_states[α] = eigen_state_list[i]
end

using LinearAlgebra
using SpecialFunctions

# ---------- basic special functions ----------

_sphj(l, x) = sqrt(pi / (2x)) * besselj(l + 0.5, x)
_sphn(l, x) = sqrt(pi / (2x)) * bessely(l + 0.5, x)

_ricj(l, x) = x * _sphj(l, x)
_ricn(l, x) = x * _sphn(l, x)

function _legendreP(l::Int, x::Real)
    l == 0 && return one(float(x))
    l == 1 && return float(x)

    p0 = one(float(x))
    p1 = float(x)

    for n in 2:l
        p = ((2n - 1) * x * p1 - (n - 1) * p0) / n
        p0, p1 = p1, p
    end

    return p1
end

_y_l0(l::Int, θ::Real) =
    sqrt((2l + 1) / (4pi)) * _legendreP(l, cos(θ))

function _z_ang_l0(lc::Int, lb::Int)
    if lc == lb + 1
        return (lb + 1) / sqrt((2lb + 1) * (2lb + 3))
    elseif lc == lb - 1 && lb ≥ 1
        return lb / sqrt((2lb - 1) * (2lb + 1))
    else
        return 0.0
    end
end

# ---------- scattering radial construction ----------

function _local_dk(kvals::AbstractVector{<:Real})
    N = length(kvals)
    N ≥ 2 || error("At least two positive-k box states are required.")

    dk = similar(float.(kvals))
    dk[1] = kvals[2] - kvals[1]
    dk[N] = kvals[N] - kvals[N-1]

    @inbounds for n in 2:N-1
        dk[n] = 0.5 * (kvals[n+1] - kvals[n-1])
    end

    all(>(0), dk) || error("k values must be strictly increasing.")
    return dk
end

function _fit_delta_l(l::Int, k::Real, phi_col, r, match_inds)
    imax = match_inds[argmax(abs.(phi_col[match_inds]))]
    phase = abs(phi_col[imax]) == 0 ? one(eltype(phi_col)) : phi_col[imax] / abs(phi_col[imax])

    y = real.(phi_col[match_inds] ./ phase)
    A = Matrix{Float64}(undef, length(match_inds), 2)

    @inbounds for (q, i) in pairs(match_inds)
        x = k * r[i]
        A[q, 1] = _ricj(l, x)
        A[q, 2] = _ricn(l, x)
    end

    c = A \ y
    a, b = c[1], c[2]

    return atan(-b, a), phase
end

function _build_Fbox_l(l::Int, phi, kvals, r, match_inds, dr)
    keep = findall(>(0), kvals)
    length(keep) ≥ 2 || error("Need at least two positive-k states for l = $l.")

    kvals = collect(float.(kvals[keep]))
    phi = phi[:, keep]

    ord = sortperm(kvals)
    kvals = kvals[ord]
    phi = phi[:, ord]

    dk = _local_dk(kvals)

    Nr, Nkbox = size(phi)
    Fbox = Matrix{ComplexF64}(undef, Nr, Nkbox)

    @inbounds for n in 1:Nkbox
        δ, phase = _fit_delta_l(l, kvals[n], phi[:, n], r, match_inds)

        # phi[:,n] has discrete normalization sum_i |phi_i|^2 = 1.
        # Dividing by sqrt(dr) converts it to continuous radial samples.
        Fbox[:, n] .= cis(-δ) .* (phi[:, n] ./ phase) ./ sqrt(dr * dk[n])
    end

    return kvals, Fbox
end

function _interp_F_to_kgrid(kvals, Fbox, k_grid)
    Nr = size(Fbox, 1)
    Fk = Matrix{ComplexF64}(undef, Nr, length(k_grid))

    @inbounds for (q, k) in pairs(k_grid)
        kvals[1] ≤ k ≤ kvals[end] ||
            error("Target k = $k is outside available box-state k range.")

        if k == kvals[end]
            Fk[:, q] .= Fbox[:, end]
        else
            n = searchsortedlast(kvals, k)
            n = max(1, min(n, length(kvals) - 1))
            t = (k - kvals[n]) / (kvals[n+1] - kvals[n])
            Fk[:, q] .= (1 - t) .* Fbox[:, n] .+ t .* Fbox[:, n+1]
        end
    end

    return Fk
end

# ---------- main function ----------

"""
    dipole_z_scattering_bound(
        phi_scat_by_l, pval_by_l,
        phi_bound_by_alpha, l_bound,
        r, k_grid, theta_grid;
        match = (0.6r[end], 0.85r[end])
    )

Compute

    <Ψ_k^(-) | z | Ψ_α^B>

for bound states with m = 0.

Inputs:
- `phi_scat_by_l[l+1]`: box-normalized continuum radial states for angular momentum `l`.
  Each matrix has indices `[i, n]`, with `sum(abs2, phi[:,n]) = 1`.
- `pval_by_l[l+1]`: momentum values corresponding to columns of `phi_scat_by_l[l+1]`.
- `phi_bound_by_alpha[α]`: box-normalized bound radial state, with
  `sum(abs2, phi_bound) = 1`.
- `l_bound[α]`: angular momentum `l_α` of bound state `α`.
- `r`: radial grid.
- `k_grid`: sampled values of `|k|`.
- `theta_grid`: sampled polar angles of `k` with respect to the positive z-axis.
- `match`: radial interval for extracting scattering phase shifts.

Output:
- `D_by_alpha[α][ik, itheta] = <Ψ_k^(-)|z|Ψ_α^B>`.
"""
function dipole_z_scattering_bound(
    phi_scat_by_l::AbstractVector{<:AbstractMatrix},
    pval_by_l::AbstractVector{<:AbstractVector},
    phi_bound_by_alpha::AbstractVector{<:AbstractVector},
    l_bound::AbstractVector{<:Integer},
    r::AbstractVector{<:Real},
    k_grid::AbstractVector{<:Real},
    theta_grid::AbstractVector{<:Real};
    match = (0.6 * r[end], 0.85 * r[end]),
)
    Lnum = length(phi_scat_by_l)
    Lnum == length(pval_by_l) || error("phi_scat_by_l and pval_by_l size mismatch.")
    length(phi_bound_by_alpha) == length(l_bound) || error("Bound-state data size mismatch.")

    Nr = length(r)
    dr = r[2] - r[1]

    match_inds = findall(i -> match[1] ≤ r[i] ≤ match[2], eachindex(r))
    length(match_inds) ≥ 3 || error("Matching interval contains too few grid points.")

    Nk = length(k_grid)
    Nθ = length(theta_grid)

    # Build F_l^(-)(k,r) for all available l.
    F_by_l = Vector{Matrix{ComplexF64}}(undef, Lnum)

    for l in 0:Lnum-1
        phi = phi_scat_by_l[l+1]
        size(phi, 1) == Nr || error("Radial size mismatch for scattering l = $l.")

        kvals, Fbox = _build_Fbox_l(l, phi, pval_by_l[l+1], r, match_inds, dr)
        F_by_l[l+1] = _interp_F_to_kgrid(kvals, Fbox, k_grid)
    end

    D_by_alpha = Vector{Matrix{ComplexF64}}(undef, length(phi_bound_by_alpha))

    for α in eachindex(phi_bound_by_alpha)
        lb = Int(l_bound[α])
        phiB = phi_bound_by_alpha[α]
        length(phiB) == Nr || error("Radial size mismatch for bound state α = $α.")

        # Convert discrete radial samples to continuous radial samples.
        phiB_cont = phiB ./ sqrt(dr)

        D = zeros(ComplexF64, Nk, Nθ)

        for lc in (lb - 1, lb + 1)
            0 ≤ lc ≤ Lnum - 1 || continue

            Cang = _z_ang_l0(lc, lb)
            Cang == 0 && continue

            Fk = F_by_l[lc+1]

            # Radial part:
            # ∫ conj(F_l^(-)(k,r)) * phi_B(r) * r dr
            Rk = Vector{ComplexF64}(undef, Nk)

            @inbounds for ik in 1:Nk
                s = zero(ComplexF64)

                for i in 1:Nr
                    s += conj(Fk[i, ik]) * phiB_cont[i] * r[i]
                end

                Rk[ik] = dr * s
            end

            @inbounds for ik in 1:Nk
                pref = (-im)^lc / k_grid[ik] * Rk[ik] * Cang

                for it in 1:Nθ
                    D[ik, it] += pref * _y_l0(lc, theta_grid[it])
                end
            end
        end

        D_by_alpha[α] = D
    end

    return D_by_alpha
end


# pre-calculate d^z_{α1, α2} (m = 0)
dipole_z_bb = zeros(ComplexF64, N_alpha, N_alpha)
for α1 in alpha_list
    for α2 in alpha_list
        l1 = eigen_states[α1].l
        l2 = eigen_states[α2].l
        if abs(l1 - l2) != 1
            continue
        end
        for (j, r) in enumerate(rs)
            # dipole_z_bb[α1, α2] += eigen_states[α1].data[j] * r * eigen_states[α2].data[j]
            dipole_z_bb[α1, α2] += conj(eigen_states[α1].data[j]) * r * eigen_states[α2].data[j]
        end
        if l1 - l2 == -1
            dipole_z_bb[α1, α2] *= (l1 + 1) / sqrt((2 * l1 + 1) * (2 * l1 + 3))
        elseif l1 - l2 == 1
            dipole_z_bb[α1, α2] *= (l1) / sqrt((2 * l1 - 1) * (2 * l1 + 1))
        end
    end
end

p_max = 5.0
p_min = 0.02
Np = 1000 * 5 ÷ 2
Δp = (p_max - p_min) / Np
N_theta = 180
Δtheta = π / N_theta
p_grid = [p_min + (i - 1 + 0.5) * Δp for i = 1: Np]                 # use mid point grid
theta_grid = [(i - 1 + 0.5) * Δtheta for i = 1: N_theta]

dipole_z_cb_raw = dipole_z_scattering_bound(
    phi_by_l,
    pval_by_l,
    [e.data for e in eigen_states[alpha_list]],
    [e.l for e in eigen_states[alpha_list]],
    rs,
    p_grid,
    theta_grid;
    # match = (0.6 * rmax, 0.8 * rmax)
    match = (0.8 * rmax, 0.9 * rmax)
)

dipole_z_cb = Vector{Matrix{ComplexF64}}(undef, N_alpha)
for (i, α) in enumerate(alpha_list)
    dipole_z_cb[α] = dipole_z_cb_raw[i]
end

target = real.(dipole_z_cb[1])
heatmap([theta_grid; theta_grid .+ pi], p_grid, [target target[:, end:-1:1]], projection=:polar, color=:cork)


real_dipole_res = h5open("./data/2026_6_2_dipole_z_cb.h5", "r") do h
    _read_complex(h, "dipole_z_cb_1")
end
des2 = real.(real_dipole_res)
plot(p_grid[1: end ÷ 1], [target[1: end ÷ 1, end ÷ 1], des2[1: end ÷ 1, end ÷ 1]])


# h5open("./data/2026_5_27_dipole_z_cb.h5", "w") do h
#     _write_complex(h, "dipole_z_cb_1", dipole_z_cb[1])
#     _write_complex(h, "dipole_z_cb_2", dipole_z_cb[2])
#     _write_complex(h, "dipole_z_cb_3", dipole_z_cb[3])
#     _write_complex(h, "dipole_z_cb_6", dipole_z_cb[6])
#     _write_complex(h, "dipole_z_cb_7", dipole_z_cb[7])
#     _write_complex(h, "dipole_z_cb_8", dipole_z_cb[8])
# end


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
E_fs =          0.05               # peak electric field of the fs pulse
ω_fs =          0.057 * 0.8            # angular frequency of the fs pulse
nc =            12                  # number of optical cycles in the fs pulse
Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, 0, ellipticity=0.0, phase1=0.5pi)        # create the light pulse from the given parameters (+ ellipticity)
E_field(t) = Ex_fs(t)
Tp = 2 * nc * pi / ω_fs


# create time grid and auxiliaries for characteristic curves
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

coupling_amplitude = pre_judgement(p_curves_p, p_curves_theta, chi_curves,
    kappa_grid_p, kappa_grid_theta, N_kappa, Δp, Δtheta, dipole_z_cb, Et_data, eta_data, eta_1_data, eta_2_data, eigen_states, N_theta, ts[1: (length(ts))])

# get a 2D slice of coupling_amplitude in x-z plane (ϕ = 0) (Spherical)
coupling_amplitude_xz = zeros(Float64, length(kappa_p_subgrid), length(kappa_theta_subgrid))
for i in 1: length(kappa_p_subgrid), j in 1: length(kappa_theta_subgrid)
    kappa_index = (i - 1) * length(kappa_theta_subgrid) + j
    coupling_amplitude_xz[i, j] = real(norm(coupling_amplitude[kappa_index]) ^ 2)
end

colormap = cgrad([:white, palette(:jet1, 6)...], rev = true)
display_data = clamp.(abs.(log10.(norm.(coupling_amplitude_xz) ./ maximum(norm.(coupling_amplitude_xz)))), 0, 3.0)
heatmap([kappa_theta_subgrid; π .+ kappa_theta_subgrid], kappa_p_subgrid, [display_data display_data[:, end:-1:1]], projection=:polar, color=colormap)

colormap_2 = cgrad(["#710000", "#C60001", "#FB0001", "#FE3A01", "#FF7A00", "#FEAB01", "#FEAB01", "#FEAB01", "#D5FF17", "#91FF60", "#4AFFAE", "#02FFFC", "#00A3FE", "#003DFF", "#2120FF", "#8C8CFE", :white], rev=true)
p1 = heatmap([kappa_theta_subgrid .+ π/2; kappa_theta_subgrid .+ 3π/2], kappa_p_subgrid[1: end ÷ 2], [coupling_amplitude_xz coupling_amplitude_xz[:, end:-1:1]][1: end ÷ 2, :], projection=:polar, color=colormap_2)




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
for it in 1: 2: (length(ts) - 2)
    t = ts[it]

    rk4_one_step(Δt, u, u_buffer[1], u_buffer[2], u_buffer[3], u_buffer[4], u_buffer[5], t, it, sfa_buffer);

    # krylov_method_one_step
    
    if it % 100 == 1
        rog = it / length(ts) * 100
        @printf "it = %d, rate of progress = %3.1f %% \n" it rog
        
        # get population
        P_bound_sum = sum(norm.(u[1: N_alpha]) .^ 2)
        P_1s = norm.(u[1]) .^ 2
        P_2s = norm.(u[2]) .^ 2
        P_2p = norm.(u[3]) .^ 2
        P_free_sum = sum(norm.(u[N_alpha + 1: end]) .^ 2)
        P_total = P_bound_sum + P_free_sum

        @printf "Total = %0.4f, 1s = %0.4f, 2s = %0.4f, 2p = %0.4f, Free = %0.4f\n"  P_total  P_1s  P_2s  P_2p  P_free_sum
    end
end

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
heatmap([kappa_theta_subgrid; π .+ kappa_theta_subgrid], kappa_p_subgrid, [display_data display_data[:, end:-1:1]], projection=:polar, color=colormap)

colormap_2 = cgrad(["#710000", "#C60001", "#FB0001", "#FE3A01", "#FF7A00", "#FEAB01", "#FEAB01", "#FEAB01", "#D5FF17", "#91FF60", "#4AFFAE", "#02FFFC", "#00A3FE", "#003DFF", "#2120FF", "#8C8CFE", :white], rev=true)
p2 = heatmap([kappa_theta_subgrid .+ π/2; kappa_theta_subgrid .+ 3π/2], kappa_p_subgrid[1: end ÷ 2], [norm.(u_free_xz) norm.(u_free_xz)[:, end:-1:1]][1: end ÷ 2, :], projection=:polar, color=colormap_2)











