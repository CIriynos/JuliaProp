module ExtendedSFA

using LinearAlgebra
using SpecialFunctions
using HDF5
import Base.Threads: @threads, threadid, nthreads

export RadialGrid, BoundState, RadialTransformTable, KappaGrid, SFAModel, SFAScratch,
       ObservableBuffer,
       radial_grid, make_bound_state, normalize_radial!, build_zBB,
       precompute_radial_transforms, build_kappa_grid, make_scratch,
       state_slices, initial_state, recover_B!, update_dipoles!, sfa_rhs!,
       dipole_channels, populations, init_observable_buffer, set_observable!,
       channel_spectrum, save_model_hdf5, load_model_hdf5,
       save_scratch_hdf5, load_scratch_hdf5,
       save_observables_hdf5, load_observables_hdf5

const IM = 1im

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

"""
    RadialGrid

Uniform radial grid with the convention required here:

```julia
r[i] = (i - 1) * dr
```

Fields
- `r`: radial grid points, including `r[1] == 0`.
- `dr`: radial spacing.
"""
struct RadialGrid
    r::Vector{Float64}
    dr::Float64
end

"""
    BoundState

One supplied bound eigenstate labeled by `α = (n, l)`, with `m = 0` assumed.
No eigenstate solver is provided in this module.

Fields
- `n`: user's radial/principal-like eigenstate label within the given `l` sector.
- `l`: orbital angular momentum.
- `I`: ionization potential, `I = -E_bound`, in atomic units.
- `u`: complex radial function `u_nl(r)` on `RadialGrid.r`, normalized as
  `sum(abs2, u) * dr ≈ 1`.
"""
struct BoundState
    n::Int
    l::Int
    I::Float64
    u::Vector{ComplexF64}
end

"""
    RadialTransformTable

Precomputed plane-wave radial transforms used in `d^z_{p,α}`.

For each bound state `α`, the nonzero channels are `L = l - 1` and `L = l + 1`:

```julia
R_{α,L}(p) = ∫ r^2 u_α(r) j_L(p*r) dr
```

Fields
- `pgrid`: positive momentum grid for tabulating transforms.
- `Rminus[:, α]`: transform for `L = lα - 1`; zero if `lα == 0`.
- `Rplus[:, α]`: transform for `L = lα + 1`.
"""
struct RadialTransformTable
    pgrid::Vector{Float64}
    Rminus::Matrix{ComplexF64}
    Rplus::Matrix{ComplexF64}
end

"""
    KappaGrid

Fixed characteristic-label grid `κ_j`. The physical continuum momentum is

```julia
p_j(t) = κ_j + η(t) * zhat
```

Fields
- `kx_axis`, `ky_axis`, `kz_axis`: uniform Cartesian axes.
- `kx`, `ky`, `kz`: flattened arrays.
- `w`: flattened quadrature weights, usually `dkx*dky*dkz`.
- `nx`, `ny`, `nz`: grid sizes.
- `dkz`: `κ_z` spacing for finite differences in `D_CC`.
"""
struct KappaGrid
    kx_axis::Vector{Float64}
    ky_axis::Vector{Float64}
    kz_axis::Vector{Float64}
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}
    w::Vector{Float64}
    nx::Int
    ny::Int
    nz::Int
    dkz::Float64
end

"""
    SFAModel

Numerical model for the direct extended-SFA equations in atomic units.

Fields
- `grid`: radial grid used by supplied eigenstates.
- `states`: supplied bound eigenstates labeled by `(n, l)`.
- `I`: cached ionization potentials.
- `zBB`: bound-bound dipole matrix `z_{αβ}`.
- `kgrid`: characteristic-label continuum grid.
- `rtab`: radial transform table for plane-wave continuum dipoles.
- `Efield`: callable laser field `Efield(t)::Real`.
"""
struct SFAModel{F}
    grid::RadialGrid
    states::Vector{BoundState}
    I::Vector{Float64}
    zBB::Matrix{ComplexF64}
    kgrid::KappaGrid
    rtab::RadialTransformTable
    Efield::F
end

"""
    SFAScratch

Reusable work buffer. Allocate once by `make_scratch(model)` and pass it to
RHS/observable routines to avoid repeated allocation.
"""
mutable struct SFAScratch
    pabs::Vector{Float64}
    costh::Vector{Float64}
    dPN::Matrix{ComplexF64}       # dPN[j, α] = d^z_{p_j, α}
    B::Vector{ComplexF64}         # B_j = C_j * exp(-iχ_j)
    dCdkz::Vector{ComplexF64}
    feedback_tls::Matrix{ComplexF64}
end

"""
    ObservableBuffer

Time traces for dipole channels and populations.

Fields
- `D_BB`: bound-bound polarization channel.
- `D_BC`: bound-continuum/recombination channel.
- `D_CC`: continuum-continuum/free-electron channel.
"""
mutable struct ObservableBuffer
    t::Vector{Float64}
    D_BB::Vector{Float64}
    D_BC::Vector{Float64}
    D_CC::Vector{Float64}
    P_B::Vector{Float64}
    P_C::Vector{Float64}
    P_tot::Vector{Float64}
end

# -----------------------------------------------------------------------------
# Grid and supplied-state utilities
# -----------------------------------------------------------------------------

"""
    radial_grid(dr, Nr) -> RadialGrid

Construct a radial grid satisfying `r[i] = (i - 1) * dr`. Thus `r[1] == 0`.

Arguments
- `dr`: radial spacing.
- `Nr`: number of radial grid points, including the origin.
"""
function radial_grid(dr::Real, Nr::Integer)
    Nr ≥ 2 || error("Nr must be at least 2.")
    dr > 0 || error("dr must be positive.")
    r = [Float64(i - 1) * Float64(dr) for i in 1:Nr]
    return RadialGrid(r, Float64(dr))
end

"""
    make_bound_state(n, l, I, u) -> BoundState

Create a `BoundState` from a user-supplied complex radial eigenfunction.
The function does not normalize `u`; call `normalize_radial!` explicitly if needed.
"""
function make_bound_state(n::Integer, l::Integer, I::Real, u::AbstractVector{<:Number})
    return BoundState(Int(n), Int(l), Float64(I), ComplexF64.(u))
end

"""
    normalize_radial!(state, grid) -> state

Normalize a complex radial eigenfunction using `sum(abs2, u) * dr = 1`.
The origin point is included; for regular radial functions its contribution is usually zero.
"""
function normalize_radial!(state::BoundState, grid::RadialGrid)
    length(state.u) == length(grid.r) || error("state.u and grid.r lengths do not match.")
    norm2 = real(sum(abs2, state.u) * grid.dr)
    norm2 > 0 || error("Cannot normalize a zero radial function.")
    state.u ./= sqrt(norm2)
    return state
end

@inline function idx3(ix::Int, iy::Int, iz::Int, g::KappaGrid)
    return ((ix - 1) * g.ny + (iy - 1)) * g.nz + iz
end

@inline function _uniform_step(x::AbstractVector{<:Real}, name::String)
    length(x) ≥ 2 || error("$name must contain at least two points.")
    dx = Float64(x[2] - x[1])
    dx != 0 || error("$name spacing cannot be zero.")
    tol = 100eps(Float64) * max(1.0, maximum(abs, x))
    for i in 2:(length(x) - 1)
        abs((x[i + 1] - x[i]) - dx) ≤ tol || error("$name must be uniform.")
    end
    return dx
end

"""
    build_kappa_grid(kx_axis, ky_axis, kz_axis) -> KappaGrid

Build a flattened uniform characteristic-label grid.
The flattening order is `ix -> iy -> iz`, with `kz` as the fastest index.
"""
function build_kappa_grid(
    kx_axis::AbstractVector{<:Real},
    ky_axis::AbstractVector{<:Real},
    kz_axis::AbstractVector{<:Real},
)
    kxv = Float64.(kx_axis)
    kyv = Float64.(ky_axis)
    kzv = Float64.(kz_axis)

    dkx = abs(_uniform_step(kxv, "kx_axis"))
    dky = abs(_uniform_step(kyv, "ky_axis"))
    dkz = abs(_uniform_step(kzv, "kz_axis"))

    nx, ny, nz = length(kxv), length(kyv), length(kzv)
    Nk = nx * ny * nz
    kx = Vector{Float64}(undef, Nk)
    ky = Vector{Float64}(undef, Nk)
    kz = Vector{Float64}(undef, Nk)
    w  = fill(dkx * dky * dkz, Nk)

    @inbounds for ix in 1:nx, iy in 1:ny, iz in 1:nz
        j = ((ix - 1) * ny + (iy - 1)) * nz + iz
        kx[j] = kxv[ix]
        ky[j] = kyv[iy]
        kz[j] = kzv[iz]
    end

    return KappaGrid(kxv, kyv, kzv, kx, ky, kz, w, nx, ny, nz, dkz)
end

# -----------------------------------------------------------------------------
# Dipole and transform construction
# -----------------------------------------------------------------------------

"""
    angular_cos_coeff(l, lp) -> Float64

Return `<Y_{l0}|cosθ|Y_{lp,0}>`. It is nonzero only for `lp = l ± 1`.
"""
@inline function angular_cos_coeff(l::Int, lp::Int)
    if lp == l + 1
        return (l + 1) / sqrt((2l + 1) * (2l + 3))
    elseif lp == l - 1 && l > 0
        return l / sqrt((2l - 1) * (2l + 1))
    else
        return 0.0
    end
end

"""
    build_zBB(states, grid) -> Matrix{ComplexF64}

Construct the bound-bound dipole matrix

```julia
zBB[α, β] = <ψ_α|z|ψ_β>
```

for `m = 0` states. The radial integral is
`∫ conj(u_α) * r * u_β dr`, and the angular factor enforces `lβ = lα ± 1`.
"""
function build_zBB(states::Vector{BoundState}, grid::RadialGrid)
    Nb = length(states)
    zBB = zeros(ComplexF64, Nb, Nb)

    @inbounds for α in 1:Nb, β in 1:Nb
        C = angular_cos_coeff(states[α].l, states[β].l)
        C == 0.0 && continue

        acc = 0.0 + 0.0im
        @simd for ir in eachindex(grid.r)
            acc += conj(states[α].u[ir]) * grid.r[ir] * states[β].u[ir]
        end
        zBB[α, β] = C * acc * grid.dr
    end
    return zBB
end

@inline function spherical_besselj_l(l::Int, x::Float64)
    l < 0 && return 0.0
    if abs(x) < 1e-12
        return l == 0 ? 1.0 : 0.0
    end
    return sqrt(pi / (2x)) * besselj(l + 0.5, x)
end

@inline function legendreP_l(l::Int, x::Float64)
    l == 0 && return 1.0
    l == 1 && return x
    p0 = 1.0
    p1 = x
    @inbounds for n in 2:l
        p = ((2n - 1) * x * p1 - (n - 1) * p0) / n
        p0, p1 = p1, p
    end
    return p1
end

@inline function Y_L0(L::Int, costh::Float64)
    return sqrt((2L + 1) / (4π)) * legendreP_l(L, clamp(costh, -1.0, 1.0))
end

"""
    precompute_radial_transforms(states, grid, pgrid) -> RadialTransformTable

Precompute the radial transforms entering the plane-wave continuum dipole:

```julia
R_{α,L}(p) = ∫ r^2 u_α(r) j_L(p*r) dr
```

The supplied eigenstates may be complex arrays. No eigenstate solver is called.
"""
function precompute_radial_transforms(
    states::Vector{BoundState},
    grid::RadialGrid,
    pgrid::AbstractVector{<:Real},
)
    Nb = length(states)
    pg = Float64.(pgrid)
    length(pg) ≥ 2 || error("pgrid must contain at least two points.")

    Rminus = zeros(ComplexF64, length(pg), Nb)
    Rplus  = zeros(ComplexF64, length(pg), Nb)

    @threads for α in 1:Nb
        l = states[α].l
        u = states[α].u
        length(u) == length(grid.r) || error("State $α radial length does not match grid.")

        for (ip, p) in pairs(pg)
            # Channel L = l + 1
            Lp = l + 1
            accp = 0.0 + 0.0im
            @inbounds @simd for ir in eachindex(grid.r)
                r = grid.r[ir]
                accp += r^2 * u[ir] * spherical_besselj_l(Lp, p * r)
            end
            Rplus[ip, α] = accp * grid.dr

            # Channel L = l - 1, absent for l = 0
            if l > 0
                Lm = l - 1
                accm = 0.0 + 0.0im
                @inbounds @simd for ir in eachindex(grid.r)
                    r = grid.r[ir]
                    accm += r^2 * u[ir] * spherical_besselj_l(Lm, p * r)
                end
                Rminus[ip, α] = accm * grid.dr
            end
        end
    end

    return RadialTransformTable(pg, Rminus, Rplus)
end

@inline function interp1_linear_col(xgrid::Vector{Float64}, Y::Matrix{ComplexF64}, col::Int, x::Float64)
    x < xgrid[1] && return 0.0 + 0.0im
    x > xgrid[end] && return 0.0 + 0.0im
    x == xgrid[end] && return Y[end, col]

    i = searchsortedlast(xgrid, x)
    i = clamp(i, 1, length(xgrid) - 1)
    x0, x1 = xgrid[i], xgrid[i + 1]
    s = (x - x0) / (x1 - x0)
    return (1 - s) * Y[i, col] + s * Y[i + 1, col]
end

"""
    continuum_dipole_z(pabs, costh, α, states, rtab) -> ComplexF64

Evaluate the plane-wave continuum dipole `d^z_{p,α}` for linearly polarized `z` field.
The formula uses the precomputed `R_{α,l±1}(p)` transforms and `Y_{L0}(p̂)`.
"""
function continuum_dipole_z(
    pabs::Float64,
    costh::Float64,
    α::Int,
    states::Vector{BoundState},
    rtab::RadialTransformTable,
)
    l = states[α].l
    val = 0.0 + 0.0im

    # L = l + 1 contribution
    Lp = l + 1
    Cp = angular_cos_coeff(l, Lp)
    if Cp != 0.0
        Rp = interp1_linear_col(rtab.pgrid, rtab.Rplus, α, pabs)
        val += Cp * (-IM)^Lp * Y_L0(Lp, costh) * Rp
    end

    # L = l - 1 contribution
    if l > 0
        Lm = l - 1
        Cm = angular_cos_coeff(l, Lm)
        if Cm != 0.0
            Rm = interp1_linear_col(rtab.pgrid, rtab.Rminus, α, pabs)
            val += Cm * (-IM)^Lm * Y_L0(Lm, costh) * Rm
        end
    end

    return sqrt(2 / π) * val
end

# -----------------------------------------------------------------------------
# Model state, scratch, and RHS
# -----------------------------------------------------------------------------

"""
    make_scratch(model) -> SFAScratch

Allocate reusable arrays for momentum magnitudes, continuum dipoles, recovered `B_j`,
and thread-local feedback accumulators.
"""
function make_scratch(model::SFAModel)
    Nk = length(model.kgrid.kx)
    Nb = length(model.states)
    nt = max(1, nthreads())
    return SFAScratch(
        zeros(Float64, Nk),
        zeros(Float64, Nk),
        zeros(ComplexF64, Nk, Nb),
        zeros(ComplexF64, Nk),
        zeros(ComplexF64, Nk),
        zeros(ComplexF64, Nb, nt),
    )
end

"""
    state_slices(model) -> NamedTuple

Return index ranges for a flattened ODE state vector:

```julia
u = [a; C; eta; chi; zeta]
```

where `C_j` is the phase-removed continuum amplitude,
`B_j = C_j * exp(-iχ_j)`, and `zeta_j = ∂χ_j/∂κ_z`.
"""
function state_slices(model::SFAModel)
    Nb = length(model.states)
    Nk = length(model.kgrid.kx)
    a    = 1:Nb
    C    = (last(a) + 1):(last(a) + Nk)
    eta  = (last(C) + 1):(last(C) + 1)
    chi  = (last(eta) + 1):(last(eta) + Nk)
    zeta = (last(chi) + 1):(last(chi) + Nk)
    return (a = a, C = C, eta = eta, chi = chi, zeta = zeta)
end

"""
    initial_state(model; ground_index=1) -> Vector{ComplexF64}

Construct initial state with `a[ground_index] = 1`, all other bound amplitudes zero,
`C_j = 0`, `η = 0`, `χ_j = 0`, and `ζ_j = 0`.
"""
function initial_state(model::SFAModel; ground_index::Int = 1)
    s = state_slices(model)
    N = last(s.zeta)
    u = zeros(ComplexF64, N)
    1 ≤ ground_index ≤ length(model.states) || error("ground_index out of range.")
    u[s.a[ground_index]] = 1.0 + 0.0im
    return u
end

"""
    recover_B!(B, C, chi) -> B

Recover the physical characteristic-grid continuum amplitude

```julia
B_j = C_j * exp(-iχ_j)
```

from the phase-removed variable `C_j`.
"""
function recover_B!(B::Vector{ComplexF64}, C, chi)
    length(B) == length(C) == length(chi) || error("B, C, chi lengths must match.")
    @threads for j in eachindex(B)
        @inbounds B[j] = C[j] * exp(-IM * real(chi[j]))
    end
    return B
end

"""
    update_dipoles!(scratch, model, eta) -> scratch

Update `p_j(t)`, `|p_j(t)|`, `cosθ_j`, and `dPN[j,α] = d^z_{p_j(t),α}`
for the current characteristic shift `eta`.
"""
function update_dipoles!(scratch::SFAScratch, model::SFAModel, eta::Real)
    g = model.kgrid
    states = model.states
    rtab = model.rtab
    Nb = length(states)
    Nk = length(g.kx)

    @threads for j in 1:Nk
        @inbounds begin
            px = g.kx[j]
            py = g.ky[j]
            pz = g.kz[j] + eta
            p2 = px * px + py * py + pz * pz
            pabs = sqrt(p2)
            costh = pabs > 0 ? pz / pabs : 0.0
            scratch.pabs[j] = pabs
            scratch.costh[j] = costh
            @simd for α in 1:Nb
                scratch.dPN[j, α] = continuum_dipole_z(pabs, costh, α, states, rtab)
            end
        end
    end
    return scratch
end

"""
    sfa_rhs!(du, u, model, t, scratch) -> du

Right-hand side for the phase-removed direct extended-SFA system:

```julia
p_j(t) = κ_j + η(t) zhat
B_j(t) = C_j(t) * exp(-iχ_j(t))
χ'_j(t) = |p_j(t)|^2 / 2
ζ'_j(t) = p_{j,z}(t)
η'(t) = E(t)
```

The propagated continuum variable is `C_j`, not `B_j`, which removes the fast
Volkov phase from the continuum ODE.
"""
function sfa_rhs!(
    du::Vector{ComplexF64},
    u::Vector{ComplexF64},
    model::SFAModel,
    t::Real,
    scratch::SFAScratch,
)
    s = state_slices(model)
    a    = @view u[s.a]
    C    = @view u[s.C]
    eta  = real(u[first(s.eta)])
    chi  = @view u[s.chi]

    da    = @view du[s.a]
    dC    = @view du[s.C]
    deta  = @view du[s.eta]
    dchi  = @view du[s.chi]
    dzeta = @view du[s.zeta]

    Nb = length(model.states)
    Nk = length(model.kgrid.kx)
    E = Float64(model.Efield(t))

    update_dipoles!(scratch, model, eta)
    recover_B!(scratch.B, C, chi)

    fill!(du, 0.0 + 0.0im)
    fill!(scratch.feedback_tls, 0.0 + 0.0im)

    # Continuum-to-bound feedback: thread-local reductions over κ-grid.
    @threads for j in 1:Nk
        tid = threadid()
        Bj = scratch.B[j]
        wj = model.kgrid.w[j]
        @inbounds @simd for α in 1:Nb
            scratch.feedback_tls[α, tid] +=
                wj * conj(scratch.dPN[j, α]) * exp(-IM * model.I[α] * t) * Bj
        end
    end

    # Bound amplitudes: bound-bound coupling plus continuum feedback.
    @threads for α in 1:Nb
        bb = 0.0 + 0.0im
        @inbounds @simd for β in 1:Nb
            bb += model.zBB[α, β] * exp(IM * (model.I[β] - model.I[α]) * t) * a[β]
        end

        fb = 0.0 + 0.0im
        @inbounds @simd for q in 1:size(scratch.feedback_tls, 2)
            fb += scratch.feedback_tls[α, q]
        end
        da[α] = IM * E * (bb + fb)
    end

    # Phase-removed continuum amplitude C_j and Volkov auxiliaries χ_j, ζ_j.
    @threads for j in 1:Nk
        src = 0.0 + 0.0im
        @inbounds @simd for α in 1:Nb
            src += scratch.dPN[j, α] * a[α] * exp(IM * model.I[α] * t)
        end

        @inbounds begin
            dC[j] = IM * E * src * exp(IM * real(chi[j]))
            dchi[j] = 0.5 * scratch.pabs[j]^2
            dzeta[j] = model.kgrid.kz[j] + eta
        end
    end

    deta[1] = E + 0.0im
    return du
end

# -----------------------------------------------------------------------------
# Observables and spectra
# -----------------------------------------------------------------------------

"""
    finite_diff_kz!(out, C, kgrid) -> out

Compute `∂C/∂κ_z` on the fixed characteristic grid using second-order central
finite differences in the interior and one-sided boundaries.
"""
function finite_diff_kz!(out::Vector{ComplexF64}, C, g::KappaGrid)
    length(out) == length(C) == length(g.kz) || error("Length mismatch in finite_diff_kz!.")
    nz = g.nz
    dk = g.dkz

    if nz == 2
        @threads for ix in 1:g.nx
            @inbounds for iy in 1:g.ny
                j1 = idx3(ix, iy, 1, g)
                j2 = idx3(ix, iy, 2, g)
                d = (C[j2] - C[j1]) / dk
                out[j1] = d
                out[j2] = d
            end
        end
        return out
    end

    nz ≥ 3 || error("kz grid must contain at least two points.")

    @threads for ix in 1:g.nx
        @inbounds for iy in 1:g.ny
            j1 = idx3(ix, iy, 1, g)
            j2 = idx3(ix, iy, 2, g)
            j3 = idx3(ix, iy, 3, g)
            out[j1] = (-3C[j1] + 4C[j2] - C[j3]) / (2dk)

            for iz in 2:(nz - 1)
                jm = idx3(ix, iy, iz - 1, g)
                j0 = idx3(ix, iy, iz,     g)
                jp = idx3(ix, iy, iz + 1, g)
                out[j0] = (C[jp] - C[jm]) / (2dk)
            end

            jn2 = idx3(ix, iy, nz - 2, g)
            jn1 = idx3(ix, iy, nz - 1, g)
            jn  = idx3(ix, iy, nz,     g)
            out[jn] = (3C[jn] - 4C[jn1] + C[jn2]) / (2dk)
        end
    end
    return out
end

"""
    dipole_channels(u, t, model, scratch) -> NamedTuple

Compute the three dipole channels from a flattened state vector:

- `D_BB`: bound-bound polarization.
- `D_BC`: bound-continuum/recombination contribution.
- `D_CC`: continuum-continuum/free-electron contribution using the smoother `C_j` form.
"""
function dipole_channels(u::Vector{ComplexF64}, t::Real, model::SFAModel, scratch::SFAScratch)
    s = state_slices(model)
    a = @view u[s.a]
    C = @view u[s.C]
    eta = real(u[first(s.eta)])
    chi = @view u[s.chi]
    zeta = @view u[s.zeta]

    Nb = length(model.states)
    Nk = length(model.kgrid.kx)

    update_dipoles!(scratch, model, eta)
    recover_B!(scratch.B, C, chi)

    D_BB = 0.0 + 0.0im
    @inbounds for α in 1:Nb, β in 1:Nb
        D_BB += conj(a[α]) * a[β] *
                exp(IM * (model.I[β] - model.I[α]) * t) * model.zBB[α, β]
    end

    bc = 0.0 + 0.0im
    @threads for α in 1:Nb
        local_acc = 0.0 + 0.0im
        @inbounds @simd for j in 1:Nk
            local_acc += model.kgrid.w[j] * conj(scratch.dPN[j, α]) * scratch.B[j]
        end
        # This atomic update avoids an extra Nb × nthreads buffer for a diagnostic path.
        Threads.atomic_add!(Ref(bc), conj(a[α]) * exp(-IM * model.I[α] * t) * local_acc)
    end

    # Fallback deterministic accumulation, because atomic Complex is not supported on all Julia versions.
    bc = 0.0 + 0.0im
    @inbounds for α in 1:Nb
        local_acc = 0.0 + 0.0im
        @simd for j in 1:Nk
            local_acc += model.kgrid.w[j] * conj(scratch.dPN[j, α]) * scratch.B[j]
        end
        bc += conj(a[α]) * exp(-IM * model.I[α] * t) * local_acc
    end
    D_BC = bc + conj(bc)

    finite_diff_kz!(scratch.dCdkz, C, model.kgrid)
    D_CC = 0.0 + 0.0im
    @inbounds @simd for j in 1:Nk
        D_CC += model.kgrid.w[j] * (IM * conj(C[j]) * scratch.dCdkz[j] + abs2(C[j]) * real(zeta[j]))
    end

    return (
        D_BB = real(D_BB),
        D_BC = real(D_BC),
        D_CC = real(D_CC),
        D_tot = real(D_BB + D_BC + D_CC),
    )
end

"""
    populations(u, model) -> NamedTuple

Return bound, continuum, and total populations. Since `B_j = C_j exp(-iχ_j)`,
`|B_j|² = |C_j|²`.
"""
function populations(u::Vector{ComplexF64}, model::SFAModel)
    s = state_slices(model)
    a = @view u[s.a]
    C = @view u[s.C]

    PB = sum(abs2, a)
    PC = 0.0
    @inbounds @simd for j in eachindex(C)
        PC += model.kgrid.w[j] * abs2(C[j])
    end
    return (bound = PB, continuum = PC, total = PB + PC)
end

"""
    init_observable_buffer(Nt) -> ObservableBuffer

Allocate storage for `Nt` samples of dipoles and populations.
"""
function init_observable_buffer(Nt::Integer)
    Nt > 0 || error("Nt must be positive.")
    return ObservableBuffer(
        zeros(Float64, Nt),
        zeros(Float64, Nt),
        zeros(Float64, Nt),
        zeros(Float64, Nt),
        zeros(Float64, Nt),
        zeros(Float64, Nt),
        zeros(Float64, Nt),
    )
end

"""
    set_observable!(buf, i, t, u, model, scratch) -> buf

Store dipole channels and populations into the `i`-th slot of `buf`.
"""
function set_observable!(
    buf::ObservableBuffer,
    i::Integer,
    t::Real,
    u::Vector{ComplexF64},
    model::SFAModel,
    scratch::SFAScratch,
)
    1 ≤ i ≤ length(buf.t) || error("Observable index out of range.")
    D = dipole_channels(u, t, model, scratch)
    P = populations(u, model)

    buf.t[i] = Float64(t)
    buf.D_BB[i] = D.D_BB
    buf.D_BC[i] = D.D_BC
    buf.D_CC[i] = D.D_CC
    buf.P_B[i] = P.bound
    buf.P_C[i] = P.continuum
    buf.P_tot[i] = P.total
    return buf
end

"""
    channel_spectrum(tgrid, D, omega; window=ones(length(tgrid))) -> (amps, spec)

Compute the dipole-form spectrum by trapezoidal quadrature:

```julia
D̃(Ω) = ∫ W(t) D(t) exp(iΩt) dt
S(Ω) = Ω^4 * abs2(D̃(Ω))
```
"""
function channel_spectrum(
    tgrid::AbstractVector{<:Real},
    D::AbstractVector{<:Real},
    omega::AbstractVector{<:Real};
    window::AbstractVector{<:Real} = ones(length(tgrid)),
)
    length(tgrid) == length(D) == length(window) ||
        error("tgrid, D, and window must have the same length.")

    amps = zeros(ComplexF64, length(omega))
    spec = zeros(Float64, length(omega))

    @threads for k in eachindex(omega)
        Ω = Float64(omega[k])
        acc = 0.0 + 0.0im
        @inbounds for n in 1:(length(tgrid) - 1)
            t0 = Float64(tgrid[n])
            t1 = Float64(tgrid[n + 1])
            f0 = window[n]     * D[n]     * exp(IM * Ω * t0)
            f1 = window[n + 1] * D[n + 1] * exp(IM * Ω * t1)
            acc += 0.5 * (t1 - t0) * (f0 + f1)
        end
        amps[k] = acc
        spec[k] = Ω^4 * abs2(acc)
    end

    return amps, spec
end

# -----------------------------------------------------------------------------
# HDF5 storage
# -----------------------------------------------------------------------------

_write_complex(h, name::String, x) = begin
    h[name * "/real"] = real.(x)
    h[name * "/imag"] = imag.(x)
end

_read_complex(h, name::String) = read(h[name * "/real"]) .+ IM .* read(h[name * "/imag"])

"""
    save_model_hdf5(path, model)

Store the numerical model core in an HDF5 file. The laser callback `Efield` is not
serializable and is therefore not stored; pass it back to `load_model_hdf5`.
"""
function save_model_hdf5(path::AbstractString, model::SFAModel)
    h5open(path, "w") do h
        h["grid/r"] = model.grid.r
        h["grid/dr"] = model.grid.dr

        h["states/n"] = [s.n for s in model.states]
        h["states/l"] = [s.l for s in model.states]
        h["states/I"] = model.I
        U = hcat([s.u for s in model.states]...)
        _write_complex(h, "states/u", U)

        _write_complex(h, "zBB", model.zBB)

        h["rtab/pgrid"] = model.rtab.pgrid
        _write_complex(h, "rtab/Rminus", model.rtab.Rminus)
        _write_complex(h, "rtab/Rplus", model.rtab.Rplus)

        h["kgrid/kx_axis"] = model.kgrid.kx_axis
        h["kgrid/ky_axis"] = model.kgrid.ky_axis
        h["kgrid/kz_axis"] = model.kgrid.kz_axis
    end
    return path
end

"""
    load_model_hdf5(path, Efield) -> SFAModel

Load a model core stored by `save_model_hdf5`. Since functions cannot be stored
safely in HDF5, the laser callback must be supplied again.
"""
function load_model_hdf5(path::AbstractString, Efield)
    h5open(path, "r") do h
        r = read(h["grid/r"])
        dr = read(h["grid/dr"])
        grid = RadialGrid(Vector{Float64}(r), Float64(dr))

        ns = Vector{Int}(read(h["states/n"]))
        ls = Vector{Int}(read(h["states/l"]))
        Is = Vector{Float64}(read(h["states/I"]))
        U = _read_complex(h, "states/u")
        states = [BoundState(ns[α], ls[α], Is[α], Vector{ComplexF64}(U[:, α])) for α in eachindex(ns)]

        zBB = Matrix{ComplexF64}(_read_complex(h, "zBB"))
        pgrid = Vector{Float64}(read(h["rtab/pgrid"]))
        Rminus = Matrix{ComplexF64}(_read_complex(h, "rtab/Rminus"))
        Rplus  = Matrix{ComplexF64}(_read_complex(h, "rtab/Rplus"))
        rtab = RadialTransformTable(pgrid, Rminus, Rplus)

        kx_axis = Vector{Float64}(read(h["kgrid/kx_axis"]))
        ky_axis = Vector{Float64}(read(h["kgrid/ky_axis"]))
        kz_axis = Vector{Float64}(read(h["kgrid/kz_axis"]))
        kgrid = build_kappa_grid(kx_axis, ky_axis, kz_axis)

        return SFAModel(grid, states, Is, zBB, kgrid, rtab, Efield)
    end
end

"""
    save_scratch_hdf5(path, scratch)

Store the current scratch buffer, including the current continuum dipoles `dPN`.
This is mainly useful for debugging or for reusing a fixed-η precomputed buffer.
"""
function save_scratch_hdf5(path::AbstractString, scratch::SFAScratch)
    h5open(path, "w") do h
        h["pabs"] = scratch.pabs
        h["costh"] = scratch.costh
        _write_complex(h, "dPN", scratch.dPN)
        _write_complex(h, "B", scratch.B)
        _write_complex(h, "dCdkz", scratch.dCdkz)
    end
    return path
end

"""
    load_scratch_hdf5(path) -> NamedTuple

Load a scratch snapshot stored by `save_scratch_hdf5`. It returns arrays rather than
an `SFAScratch`, because thread-local buffer sizes depend on the current session.
"""
function load_scratch_hdf5(path::AbstractString)
    h5open(path, "r") do h
        return (
            pabs = Vector{Float64}(read(h["pabs"])),
            costh = Vector{Float64}(read(h["costh"])),
            dPN = Matrix{ComplexF64}(_read_complex(h, "dPN")),
            B = Vector{ComplexF64}(_read_complex(h, "B")),
            dCdkz = Vector{ComplexF64}(_read_complex(h, "dCdkz")),
        )
    end
end

"""
    save_observables_hdf5(path, buf)

Store dipole-channel and population histories. This is the recommended storage path
for future reuse of `D_BB`, `D_BC`, `D_CC`, and population diagnostics.
"""
function save_observables_hdf5(path::AbstractString, buf::ObservableBuffer)
    h5open(path, "w") do h
        h["t"] = buf.t
        h["D_BB"] = buf.D_BB
        h["D_BC"] = buf.D_BC
        h["D_CC"] = buf.D_CC
        h["P_B"] = buf.P_B
        h["P_C"] = buf.P_C
        h["P_tot"] = buf.P_tot
    end
    return path
end

"""
    load_observables_hdf5(path) -> ObservableBuffer

Load dipole-channel and population histories stored by `save_observables_hdf5`.
"""
function load_observables_hdf5(path::AbstractString)
    h5open(path, "r") do h
        return ObservableBuffer(
            Vector{Float64}(read(h["t"])),
            Vector{Float64}(read(h["D_BB"])),
            Vector{Float64}(read(h["D_BC"])),
            Vector{Float64}(read(h["D_CC"])),
            Vector{Float64}(read(h["P_B"])),
            Vector{Float64}(read(h["P_C"])),
            Vector{Float64}(read(h["P_tot"])),
        )
    end
end

# -----------------------------------------------------------------------------
# Optional dynamic grid-point-number RK4 layer
# -----------------------------------------------------------------------------

export DynamicGridConfig, DynamicGridScratch, DynamicRK4Cache,
       make_dynamic_scratch, make_dynamic_rk4_cache,
       select_active_indices!, update_dipoles_active!, recover_B_active!,
       sfa_rhs_dynamic!, rk4_step_dynamic!, dipole_channels_dynamic

"""
    DynamicGridConfig(; kwargs...)

Configuration for the optional dynamic grid-point-number strategy.

A continuum point is treated as active if either

```julia
abs(C[j]) > amp_threshold
```

or if it lies inside a source-admissible momentum region while the field is non-negligible.
The second criterion is essential: a point with tiny current occupation can still become
populated by the ionization source during the next RK4 step.

Keyword fields
- `amp_threshold`: occupation-amplitude threshold for keeping an already populated point.
- `field_threshold`: source gate is disabled when `abs(E(t)) <= field_threshold`.
- `kperp_max`: maximum transverse momentum `sqrt(κx^2 + κy^2)` for source admission.
- `pabs_max`: maximum physical momentum `abs(κ + η zhat)` for source admission.
- `pad_kz`: optional number of neighboring points added along the `κz` direction.
- `keep_previous`: if true, previously active points remain active after reselection.
"""
struct DynamicGridConfig
    amp_threshold::Float64
    field_threshold::Float64
    kperp_max::Float64
    pabs_max::Float64
    pad_kz::Int
    keep_previous::Bool
end

function DynamicGridConfig(;
    amp_threshold::Real = 1e-10,
    field_threshold::Real = 0.0,
    kperp_max::Real = Inf,
    pabs_max::Real = Inf,
    pad_kz::Integer = 0,
    keep_previous::Bool = false,
)
    amp_threshold ≥ 0 || error("amp_threshold must be non-negative.")
    field_threshold ≥ 0 || error("field_threshold must be non-negative.")
    kperp_max > 0 || error("kperp_max must be positive.")
    pabs_max > 0 || error("pabs_max must be positive.")
    pad_kz ≥ 0 || error("pad_kz must be non-negative.")
    return DynamicGridConfig(
        Float64(amp_threshold),
        Float64(field_threshold),
        Float64(kperp_max),
        Float64(pabs_max),
        Int(pad_kz),
        keep_previous,
    )
end

"""
    DynamicGridScratch

Active-set workspace for dynamic-grid propagation.

Fields
- `active`: active flattened continuum indices used in the current RK4 macrostep.
- `mask`: Boolean active mask over all `κ` points.
- `tmpmask`: temporary mask used for optional `κz` padding.
"""
mutable struct DynamicGridScratch
    active::Vector{Int}
    mask::Vector{Bool}
    tmpmask::Vector{Bool}
end

"""
    DynamicRK4Cache

Reusable RK4 stage arrays for `rk4_step_dynamic!`.
"""
mutable struct DynamicRK4Cache
    k1::Vector{ComplexF64}
    k2::Vector{ComplexF64}
    k3::Vector{ComplexF64}
    k4::Vector{ComplexF64}
    tmp::Vector{ComplexF64}
end

"""
    make_dynamic_scratch(model) -> DynamicGridScratch

Allocate the active-index workspace for the dynamic-grid strategy.
"""
function make_dynamic_scratch(model::SFAModel)
    Nk = length(model.kgrid.kx)
    return DynamicGridScratch(Int[], fill(false, Nk), fill(false, Nk))
end

"""
    make_dynamic_rk4_cache(model) -> DynamicRK4Cache

Allocate reusable RK4 stage arrays compatible with the flattened state vector.
"""
function make_dynamic_rk4_cache(model::SFAModel)
    N = last(state_slices(model).zeta)
    return DynamicRK4Cache(
        zeros(ComplexF64, N),
        zeros(ComplexF64, N),
        zeros(ComplexF64, N),
        zeros(ComplexF64, N),
        zeros(ComplexF64, N),
    )
end

@inline function _source_gate(g::KappaGrid, j::Int, eta::Float64, Eabs::Float64, cfg::DynamicGridConfig)
    Eabs > cfg.field_threshold || return false
    kperp2 = g.kx[j]^2 + g.ky[j]^2
    kperp2 ≤ cfg.kperp_max^2 || return false
    pz = g.kz[j] + eta
    pabs2 = kperp2 + pz^2
    return pabs2 ≤ cfg.pabs_max^2
end

function _pad_mask_kz!(dyn::DynamicGridScratch, g::KappaGrid, pad::Int)
    pad ≤ 0 && return dyn
    copyto!(dyn.tmpmask, dyn.mask)

    @inbounds for ix in 1:g.nx, iy in 1:g.ny, iz in 1:g.nz
        j = idx3(ix, iy, iz, g)
        dyn.mask[j] || continue
        iz0 = max(1, iz - pad)
        iz1 = min(g.nz, iz + pad)
        for izp in iz0:iz1
            dyn.tmpmask[idx3(ix, iy, izp, g)] = true
        end
    end

    copyto!(dyn.mask, dyn.tmpmask)
    return dyn
end

"""
    select_active_indices!(dyn, u, t, model, cfg; dt=0.0) -> active

Select active continuum indices for one RK4 macrostep.

The active set is frozen during the subsequent four RK4 stages. If `dt > 0`, the
source-admissible criterion is checked at rough beginning/middle/end estimates of
`η(t)` and `E(t)` to avoid losing points that become source-relevant inside the step.
"""
function select_active_indices!(
    dyn::DynamicGridScratch,
    u::Vector{ComplexF64},
    t::Real,
    model::SFAModel,
    cfg::DynamicGridConfig;
    dt::Real = 0.0,
)
    s = state_slices(model)
    C = @view u[s.C]
    eta0 = real(u[first(s.eta)])
    g = model.kgrid
    Nk = length(g.kx)

    if !cfg.keep_previous
        fill!(dyn.mask, false)
    end

    t0 = Float64(t)
    h = Float64(dt)
    E0 = Float64(model.Efield(t0))
    E1 = Float64(model.Efield(t0 + 0.5h))
    E2 = Float64(model.Efield(t0 + h))

    # Low-cost estimates sufficient for active-set admission; the RK4 state still
    # evolves η self-consistently inside the actual RHS calls.
    eta1 = eta0 + 0.5h * E1
    eta2 = eta0 + h * E2
    eabs0, eabs1, eabs2 = abs(E0), abs(E1), abs(E2)

    @threads for j in 1:Nk
        occupied = abs(C[j]) > cfg.amp_threshold
        source_ok = _source_gate(g, j, eta0, eabs0, cfg) ||
                    _source_gate(g, j, eta1, eabs1, cfg) ||
                    _source_gate(g, j, eta2, eabs2, cfg)
        if occupied || source_ok
            @inbounds dyn.mask[j] = true
        end
    end

    _pad_mask_kz!(dyn, g, cfg.pad_kz)

    empty!(dyn.active)
    sizehint!(dyn.active, count(identity, dyn.mask))
    @inbounds for j in 1:Nk
        dyn.mask[j] && push!(dyn.active, j)
    end
    return dyn.active
end

"""
    recover_B_active!(B, C, chi, active) -> B

Recover `B_j = C_j * exp(-iχ_j)` only on the active continuum set.
Inactive entries of `B` are left unchanged and should not be used by active-set routines.
"""
function recover_B_active!(B::Vector{ComplexF64}, C, chi, active::Vector{Int})
    @threads for q in eachindex(active)
        j = active[q]
        @inbounds B[j] = C[j] * exp(-IM * real(chi[j]))
    end
    return B
end

"""
    update_dipoles_active!(scratch, model, eta, active) -> scratch

Update `pabs`, `costh`, and `dPN[j,α]` only for active continuum points.
This avoids evaluating expensive radial-transform interpolations on inactive points.
"""
function update_dipoles_active!(
    scratch::SFAScratch,
    model::SFAModel,
    eta::Real,
    active::Vector{Int},
)
    g = model.kgrid
    states = model.states
    rtab = model.rtab
    Nb = length(states)

    @threads for q in eachindex(active)
        j = active[q]
        @inbounds begin
            px = g.kx[j]
            py = g.ky[j]
            pz = g.kz[j] + eta
            p2 = px * px + py * py + pz * pz
            pabs = sqrt(p2)
            costh = pabs > 0 ? pz / pabs : 0.0
            scratch.pabs[j] = pabs
            scratch.costh[j] = costh
            @simd for α in 1:Nb
                scratch.dPN[j, α] = continuum_dipole_z(pabs, costh, α, states, rtab)
            end
        end
    end
    return scratch
end

"""
    sfa_rhs_dynamic!(du, u, model, t, scratch, dyn) -> du

Active-set RHS for the phase-removed direct extended-SFA equations.

Only active continuum indices are used for

```julia
C'_j, continuum-to-bound feedback, and d^z_{p_j,α}
```

while `χ'_j`, `ζ'_j`, and `η'` are still updated for all grid points. This is important:
if a previously inactive point becomes active later, its Volkov phase is still correct.
"""
function sfa_rhs_dynamic!(
    du::Vector{ComplexF64},
    u::Vector{ComplexF64},
    model::SFAModel,
    t::Real,
    scratch::SFAScratch,
    dyn::DynamicGridScratch,
)
    s = state_slices(model)
    a    = @view u[s.a]
    C    = @view u[s.C]
    eta  = real(u[first(s.eta)])
    chi  = @view u[s.chi]

    da    = @view du[s.a]
    dC    = @view du[s.C]
    deta  = @view du[s.eta]
    dchi  = @view du[s.chi]
    dzeta = @view du[s.zeta]

    Nb = length(model.states)
    Nk = length(model.kgrid.kx)
    E = Float64(model.Efield(t))

    fill!(du, 0.0 + 0.0im)
    fill!(scratch.feedback_tls, 0.0 + 0.0im)

    update_dipoles_active!(scratch, model, eta, dyn.active)
    recover_B_active!(scratch.B, C, chi, dyn.active)

    # Volkov auxiliaries are cheap and must remain correct for all points.
    @threads for j in 1:Nk
        @inbounds begin
            px = model.kgrid.kx[j]
            py = model.kgrid.ky[j]
            pz = model.kgrid.kz[j] + eta
            dchi[j] = 0.5 * (px * px + py * py + pz * pz)
            dzeta[j] = pz
        end
    end

    # Active continuum feedback into bound amplitudes.
    @threads for q in eachindex(dyn.active)
        tid = threadid()
        j = dyn.active[q]
        Bj = scratch.B[j]
        wj = model.kgrid.w[j]
        @inbounds @simd for α in 1:Nb
            scratch.feedback_tls[α, tid] +=
                wj * conj(scratch.dPN[j, α]) * exp(-IM * model.I[α] * t) * Bj
        end
    end

    # Bound amplitudes remain dense in α.
    @threads for α in 1:Nb
        bb = 0.0 + 0.0im
        @inbounds @simd for β in 1:Nb
            bb += model.zBB[α, β] * exp(IM * (model.I[β] - model.I[α]) * t) * a[β]
        end

        fb = 0.0 + 0.0im
        @inbounds @simd for q in 1:size(scratch.feedback_tls, 2)
            fb += scratch.feedback_tls[α, q]
        end
        da[α] = IM * E * (bb + fb)
    end

    # Active continuum source. Inactive C_j are held fixed in this RHS call.
    @threads for q in eachindex(dyn.active)
        j = dyn.active[q]
        src = 0.0 + 0.0im
        @inbounds @simd for α in 1:Nb
            src += scratch.dPN[j, α] * a[α] * exp(IM * model.I[α] * t)
        end
        @inbounds dC[j] = IM * E * src * exp(IM * real(chi[j]))
    end

    deta[1] = E + 0.0im
    return du
end

"""
    rk4_step_dynamic!(u, t, dt, model, scratch, dyn, cfg, cache) -> u

Advance one fixed-step RK4 macrostep using the dynamic grid-point-number strategy.
The active continuum set is selected once at the beginning of the macrostep and then
kept fixed for the four RK4 stages.

This routine is intended for explicit custom RK4 propagation. It is not a drop-in RHS
for adaptive ODE solvers, because adaptive solvers control their own stage structure.
"""
function rk4_step_dynamic!(
    u::Vector{ComplexF64},
    t::Real,
    dt::Real,
    model::SFAModel,
    scratch::SFAScratch,
    dyn::DynamicGridScratch,
    cfg::DynamicGridConfig,
    cache::DynamicRK4Cache,
)
    h = Float64(dt)
    select_active_indices!(dyn, u, t, model, cfg; dt = h)

    sfa_rhs_dynamic!(cache.k1, u, model, t, scratch, dyn)

    @. cache.tmp = u + 0.5 * h * cache.k1
    sfa_rhs_dynamic!(cache.k2, cache.tmp, model, t + 0.5 * h, scratch, dyn)

    @. cache.tmp = u + 0.5 * h * cache.k2
    sfa_rhs_dynamic!(cache.k3, cache.tmp, model, t + 0.5 * h, scratch, dyn)

    @. cache.tmp = u + h * cache.k3
    sfa_rhs_dynamic!(cache.k4, cache.tmp, model, t + h, scratch, dyn)

    @. u = u + (h / 6) * (cache.k1 + 2 * cache.k2 + 2 * cache.k3 + cache.k4)
    return u
end

"""
    finite_diff_kz_active!(out, C, kgrid, active) -> out

Compute `∂C/∂κ_z` only at active flattened continuum indices.
The stencil still reads neighboring `C` values from the full fixed grid, but it avoids
constructing derivatives for inactive points.
"""
function finite_diff_kz_active!(
    out::Vector{ComplexF64},
    C,
    g::KappaGrid,
    active::Vector{Int},
)
    nz = g.nz
    dk = g.dkz
    nz ≥ 2 || error("kz grid must contain at least two points.")

    @threads for q in eachindex(active)
        j = active[q]
        iz = ((j - 1) % nz) + 1
        @inbounds begin
            if nz == 2
                if iz == 1
                    out[j] = (C[j + 1] - C[j]) / dk
                else
                    out[j] = (C[j] - C[j - 1]) / dk
                end
            elseif iz == 1
                out[j] = (-3C[j] + 4C[j + 1] - C[j + 2]) / (2dk)
            elseif iz == nz
                out[j] = (3C[j] - 4C[j - 1] + C[j - 2]) / (2dk)
            else
                out[j] = (C[j + 1] - C[j - 1]) / (2dk)
            end
        end
    end
    return out
end

"""
    dipole_channels_dynamic(u, t, model, scratch, dyn) -> NamedTuple

Parallel active-set diagnostic for the dipole channels.

Optimization relative to the simple dynamic version:
- updates continuum dipoles only on `dyn.active`;
- recovers `B_j` only on `dyn.active`;
- computes `∂C/∂κ_z` only on `dyn.active`;
- accumulates `D_BC` and `D_CC` through thread-local reductions.

Call `select_active_indices!` before this function if the active set is stale.
"""
function dipole_channels_dynamic(
    u::Vector{ComplexF64},
    t::Real,
    model::SFAModel,
    scratch::SFAScratch,
    dyn::DynamicGridScratch,
)
    s = state_slices(model)
    a = @view u[s.a]
    C = @view u[s.C]
    eta = real(u[first(s.eta)])
    chi = @view u[s.chi]
    zeta = @view u[s.zeta]

    Nb = length(model.states)
    nt = max(1, nthreads())

    update_dipoles_active!(scratch, model, eta, dyn.active)
    recover_B_active!(scratch.B, C, chi, dyn.active)
    finite_diff_kz_active!(scratch.dCdkz, C, model.kgrid, dyn.active)

    # D_BB is usually small compared with continuum work, but still parallelized
    # over α through the existing thread-local buffer when possible.
    fill!(scratch.feedback_tls, 0.0 + 0.0im)
    @threads for α in 1:Nb
        tid = threadid()
        acc = 0.0 + 0.0im
        @inbounds @simd for β in 1:Nb
            acc += conj(a[α]) * a[β] *
                   exp(IM * (model.I[β] - model.I[α]) * t) * model.zBB[α, β]
        end
        scratch.feedback_tls[1, tid] += acc
    end
    D_BB = 0.0 + 0.0im
    @inbounds for tid in 1:nt
        D_BB += scratch.feedback_tls[1, tid]
    end

    # D_BC: thread-local reduction over active κ-points.
    fill!(scratch.feedback_tls, 0.0 + 0.0im)
    @threads for q in eachindex(dyn.active)
        tid = threadid()
        j = dyn.active[q]
        Bj = scratch.B[j]
        wj = model.kgrid.w[j]
        @inbounds @simd for α in 1:Nb
            scratch.feedback_tls[α, tid] += wj * conj(scratch.dPN[j, α]) * Bj
        end
    end

    bc = 0.0 + 0.0im
    @inbounds for α in 1:Nb
        accα = 0.0 + 0.0im
        @simd for tid in 1:nt
            accα += scratch.feedback_tls[α, tid]
        end
        bc += conj(a[α]) * exp(-IM * model.I[α] * t) * accα
    end
    D_BC = bc + conj(bc)

    # D_CC: scalar thread-local reduction. The small vector allocation is negligible
    # compared with active continuum dipole evaluation; keep it local to avoid
    # changing the SFAScratch struct.
    dcc_tls = zeros(ComplexF64, nt)
    @threads for q in eachindex(dyn.active)
        tid = threadid()
        j = dyn.active[q]
        @inbounds dcc_tls[tid] += model.kgrid.w[j] *
            (IM * conj(C[j]) * scratch.dCdkz[j] + abs2(C[j]) * real(zeta[j]))
    end
    D_CC = sum(dcc_tls)

    return (
        D_BB = real(D_BB),
        D_BC = real(D_BC),
        D_CC = real(D_CC),
        D_tot = real(D_BB + D_BC + D_CC),
        N_active = length(dyn.active),
    )
end

# -----------------------------------------------------------------------------
# Allocation-reduced dynamic RK4 path
# -----------------------------------------------------------------------------

export DynamicFastRK4Cache, make_dynamic_fast_rk4_cache,
       sfa_rhs_dynamic_fast!, rk4_step_dynamic_fast!

"""
    DynamicFastRK4Cache

Allocation-reduced RK4 cache for the dynamic-grid strategy.

Compared with `DynamicRK4Cache`, this cache also stores bound-state phase factors
so that `exp(±im*Iα*t)` is evaluated once per bound state and RK4 stage, not once
per active continuum point.
"""
mutable struct DynamicFastRK4Cache
    k1::Vector{ComplexF64}
    k2::Vector{ComplexF64}
    k3::Vector{ComplexF64}
    k4::Vector{ComplexF64}
    tmp::Vector{ComplexF64}
    phase_plus::Vector{ComplexF64}   # exp(+i Iα t)
    phase_minus::Vector{ComplexF64}  # exp(-i Iα t)
end

"""
    make_dynamic_fast_rk4_cache(model) -> DynamicFastRK4Cache

Allocate reusable stage arrays and phase buffers for `rk4_step_dynamic_fast!`.
"""
function make_dynamic_fast_rk4_cache(model::SFAModel)
    N = last(state_slices(model).zeta)
    Nb = length(model.states)
    return DynamicFastRK4Cache(
        zeros(ComplexF64, N),
        zeros(ComplexF64, N),
        zeros(ComplexF64, N),
        zeros(ComplexF64, N),
        zeros(ComplexF64, N),
        zeros(ComplexF64, Nb),
        zeros(ComplexF64, Nb),
    )
end

@inline function _layout(model::SFAModel)
    Nb = length(model.states)
    Nk = length(model.kgrid.kx)
    a0 = 0
    c0 = Nb
    eta_idx = Nb + Nk + 1
    chi0 = eta_idx
    zeta0 = chi0 + Nk
    return Nb, Nk, a0, c0, eta_idx, chi0, zeta0
end

@inline function _fill_bound_phases!(cache::DynamicFastRK4Cache, model::SFAModel, t::Real)
    @inbounds @simd for α in eachindex(model.I)
        θ = model.I[α] * t
        cp = cis(θ)
        cache.phase_plus[α] = cp
        cache.phase_minus[α] = conj(cp)
    end
    return cache
end

"""
    sfa_rhs_dynamic_fast!(du, u, model, t, scratch, dyn, cache) -> du

Allocation-reduced active-set RHS.

Differences from `sfa_rhs_dynamic!`:
- no full-state `fill!(du, 0)`;
- no `state_slices` or `@view` construction inside the hot path;
- one fused active-grid loop computes `dPN`, `B_j`, continuum feedback, and `dC_j`;
- bound phase factors `exp(±iIαt)` are cached once per RHS call;
- inactive `C_j` derivatives are not touched;
- `χ_j` and `ζ_j` are still updated for all grid points, because they are cheap and
  must remain correct if a point becomes active later.

This function assumes `dyn.active` has already been selected.
"""
function sfa_rhs_dynamic_fast!(
    du::Vector{ComplexF64},
    u::Vector{ComplexF64},
    model::SFAModel,
    t::Real,
    scratch::SFAScratch,
    dyn::DynamicGridScratch,
    cache::DynamicFastRK4Cache,
)
    Nb, Nk, a0, c0, eta_idx, chi0, zeta0 = _layout(model)
    g = model.kgrid
    E = Float64(model.Efield(t))
    eta = real(u[eta_idx])
    nt = max(1, nthreads())

    _fill_bound_phases!(cache, model, t)
    fill!(scratch.feedback_tls, 0.0 + 0.0im)

    # Volkov auxiliaries for all points. This loop is memory-bandwidth bound but cheap
    # compared with continuum dipole evaluation. It also keeps inactive phases valid.
    @threads :static for j in 1:Nk
        @inbounds begin
            px = g.kx[j]
            py = g.ky[j]
            pz = g.kz[j] + eta
            du[chi0 + j] = 0.5 * (px * px + py * py + pz * pz)
            du[zeta0 + j] = pz + 0.0im
        end
    end

    # Fused active-grid work: dipole update, B recovery, feedback reduction, and dC.
    @threads :static for q in eachindex(dyn.active)
        tid = threadid()
        j = dyn.active[q]

        @inbounds begin
            px = g.kx[j]
            py = g.ky[j]
            pz = g.kz[j] + eta
            p2 = px * px + py * py + pz * pz
            pabs = sqrt(p2)
            costh = pabs > 0 ? pz / pabs : 0.0

            scratch.pabs[j] = pabs
            scratch.costh[j] = costh

            Bj = u[c0 + j] * exp(-IM * real(u[chi0 + j]))
            scratch.B[j] = Bj

            src = 0.0 + 0.0im
            wj = g.w[j]

            @simd for α in 1:Nb
                d = continuum_dipole_z(pabs, costh, α, model.states, model.rtab)
                scratch.dPN[j, α] = d
                src += d * u[a0 + α] * cache.phase_plus[α]
                scratch.feedback_tls[α, tid] += wj * conj(d) * cache.phase_minus[α] * Bj
            end

            du[c0 + j] = IM * E * src * exp(IM * real(u[chi0 + j]))
        end
    end

    # Bound amplitudes: dense in α, but small.
    @threads :static for α in 1:Nb
        bb = 0.0 + 0.0im
        @inbounds @simd for β in 1:Nb
            bb += model.zBB[α, β] * cache.phase_plus[β] * cache.phase_minus[α] * u[a0 + β]
        end

        fb = 0.0 + 0.0im
        @inbounds @simd for tid in 1:nt
            fb += scratch.feedback_tls[α, tid]
        end

        @inbounds du[a0 + α] = IM * E * (bb + fb)
    end

    @inbounds du[eta_idx] = E + 0.0im
    return du
end

@inline function _stage_state_dynamic_fast!(
    tmp::Vector{ComplexF64},
    u::Vector{ComplexF64},
    k::Vector{ComplexF64},
    scale::Float64,
    model::SFAModel,
    dyn::DynamicGridScratch,
)
    Nb, Nk, a0, c0, eta_idx, chi0, zeta0 = _layout(model)

    @inbounds @simd for α in 1:Nb
        tmp[a0 + α] = u[a0 + α] + scale * k[a0 + α]
    end

    @threads :static for q in eachindex(dyn.active)
        j = dyn.active[q]
        @inbounds tmp[c0 + j] = u[c0 + j] + scale * k[c0 + j]
    end

    @inbounds tmp[eta_idx] = u[eta_idx] + scale * k[eta_idx]

    # RHS reads χ for active points and computes χ'/ζ' for all points.
    @threads :static for j in 1:Nk
        @inbounds tmp[chi0 + j] = u[chi0 + j] + scale * k[chi0 + j]
    end

    return tmp
end

@inline function _final_update_dynamic_fast!(
    u::Vector{ComplexF64},
    cache::DynamicFastRK4Cache,
    h::Float64,
    model::SFAModel,
    dyn::DynamicGridScratch,
)
    Nb, Nk, a0, c0, eta_idx, chi0, zeta0 = _layout(model)
    h6 = h / 6

    @inbounds @simd for α in 1:Nb
        u[a0 + α] += h6 * (cache.k1[a0 + α] + 2 * cache.k2[a0 + α] +
                           2 * cache.k3[a0 + α] + cache.k4[a0 + α])
    end

    @threads :static for q in eachindex(dyn.active)
        j = dyn.active[q]
        @inbounds u[c0 + j] += h6 * (cache.k1[c0 + j] + 2 * cache.k2[c0 + j] +
                                     2 * cache.k3[c0 + j] + cache.k4[c0 + j])
    end

    @inbounds u[eta_idx] += h6 * (cache.k1[eta_idx] + 2 * cache.k2[eta_idx] +
                                  2 * cache.k3[eta_idx] + cache.k4[eta_idx])

    @threads :static for j in 1:Nk
        @inbounds begin
            u[chi0 + j] += h6 * (cache.k1[chi0 + j] + 2 * cache.k2[chi0 + j] +
                                 2 * cache.k3[chi0 + j] + cache.k4[chi0 + j])
            u[zeta0 + j] += h6 * (cache.k1[zeta0 + j] + 2 * cache.k2[zeta0 + j] +
                                  2 * cache.k3[zeta0 + j] + cache.k4[zeta0 + j])
        end
    end

    return u
end

"""
    rk4_step_dynamic_fast!(u, t, dt, model, scratch, dyn, cfg, cache) -> u

Fast fixed-step RK4 macrostep for the dynamic-grid strategy.

The active set is selected once at the beginning of the macrostep and frozen during
all four RK4 stages. Unlike `rk4_step_dynamic!`, this version does not broadcast over
the entire flattened state at every stage. It updates only:

- all bound amplitudes `a_α`;
- active continuum amplitudes `C_j`;
- scalar `η`;
- all Volkov phases `χ_j` and phase gradients `ζ_j`.

Use this routine when `Nactive << Nk` or when full-state broadcasts dominate runtime.
"""
function rk4_step_dynamic_fast!(
    u::Vector{ComplexF64},
    t::Real,
    dt::Real,
    model::SFAModel,
    scratch::SFAScratch,
    dyn::DynamicGridScratch,
    cfg::DynamicGridConfig,
    cache::DynamicFastRK4Cache,
)
    h = Float64(dt)
    select_active_indices!(dyn, u, t, model, cfg; dt = h)

    sfa_rhs_dynamic_fast!(cache.k1, u, model, t, scratch, dyn, cache)

    _stage_state_dynamic_fast!(cache.tmp, u, cache.k1, 0.5 * h, model, dyn)
    sfa_rhs_dynamic_fast!(cache.k2, cache.tmp, model, t + 0.5 * h, scratch, dyn, cache)

    _stage_state_dynamic_fast!(cache.tmp, u, cache.k2, 0.5 * h, model, dyn)
    sfa_rhs_dynamic_fast!(cache.k3, cache.tmp, model, t + 0.5 * h, scratch, dyn, cache)

    _stage_state_dynamic_fast!(cache.tmp, u, cache.k3, h, model, dyn)
    sfa_rhs_dynamic_fast!(cache.k4, cache.tmp, model, t + h, scratch, dyn, cache)

    _final_update_dynamic_fast!(u, cache, h, model, dyn)
    return u
end

# -----------------------------------------------------------------------------
# Further optimized dynamic RHS: no dPN storage in propagation
# -----------------------------------------------------------------------------

export sfa_rhs_dynamic_nostore!, rk4_step_dynamic_nostore!

@inline function _chunk_bounds(n::Int, slot::Int, nslot::Int)
    qlo = fld((slot - 1) * n, nslot) + 1
    qhi = fld(slot * n, nslot)
    return qlo, qhi
end

@inline function interp1_linear_col_uniform(
    xgrid::Vector{Float64},
    Y::Matrix{ComplexF64},
    col::Int,
    x::Float64,
)
    x < xgrid[1] && return 0.0 + 0.0im
    x > xgrid[end] && return 0.0 + 0.0im

    n = length(xgrid)
    x == xgrid[end] && return Y[n, col]

    dx = xgrid[2] - xgrid[1]
    y = (x - xgrid[1]) / dx
    i = floor(Int, y) + 1
    i = clamp(i, 1, n - 1)
    s = y - (i - 1)
    @inbounds return (1 - s) * Y[i, col] + s * Y[i + 1, col]
end

"""
    continuum_dipole_z_uniform(pabs, costh, α, states, rtab) -> ComplexF64

Fast version of `continuum_dipole_z` for uniformly spaced `rtab.pgrid`.
Use this when `pgrid` was generated by `range` or another uniform grid constructor.
"""
function continuum_dipole_z_uniform(
    pabs::Float64,
    costh::Float64,
    α::Int,
    states::Vector{BoundState},
    rtab::RadialTransformTable,
)
    l = states[α].l
    val = 0.0 + 0.0im

    Lp = l + 1
    Cp = angular_cos_coeff(l, Lp)
    if Cp != 0.0
        Rp = interp1_linear_col_uniform(rtab.pgrid, rtab.Rplus, α, pabs)
        val += Cp * (-IM)^Lp * Y_L0(Lp, costh) * Rp
    end

    if l > 0
        Lm = l - 1
        Cm = angular_cos_coeff(l, Lm)
        if Cm != 0.0
            Rm = interp1_linear_col_uniform(rtab.pgrid, rtab.Rminus, α, pabs)
            val += Cm * (-IM)^Lm * Y_L0(Lm, costh) * Rm
        end
    end

    return sqrt(2 / π) * val
end

"""
    sfa_rhs_dynamic_nostore!(du, u, model, t, scratch, dyn, cache) -> du

Lowest-memory active-set RHS currently provided.

This variant is intended to replace `sfa_rhs_dynamic_fast!` when propagation speed is
limited by the active-grid block. It does not store `scratch.dPN[j,α]`, `scratch.pabs[j]`,
or `scratch.costh[j]` during propagation. Those arrays are needed by diagnostics, but
not by the RK4 RHS itself.

It also removes `threadid()` from the active-grid loop. Instead, the active list is
manually divided into fixed chunks, and each thread-slot writes only to
`scratch.feedback_tls[:, slot]`.

Assumption
- `model.rtab.pgrid` is uniformly spaced. If it is not, use `sfa_rhs_dynamic_fast!`.
"""
function sfa_rhs_dynamic_nostore!(
    du::Vector{ComplexF64},
    u::Vector{ComplexF64},
    model::SFAModel,
    t::Real,
    scratch::SFAScratch,
    dyn::DynamicGridScratch,
    cache::DynamicFastRK4Cache,
)
    Nb, Nk, a0, c0, eta_idx, chi0, zeta0 = _layout(model)
    g = model.kgrid
    E = Float64(model.Efield(t))
    eta = real(u[eta_idx])
    nt = size(scratch.feedback_tls, 2)
    nact = length(dyn.active)

    _fill_bound_phases!(cache, model, t)
    fill!(scratch.feedback_tls, 0.0 + 0.0im)

    # println("start sfa_rhs_dynamic_nostore!")

    # @time begin
    # Keep all Volkov auxiliaries valid, including currently inactive points.
    @threads :static for j in 1:Nk
        @inbounds begin
            px = g.kx[j]
            py = g.ky[j]
            pz = g.kz[j] + eta
            du[chi0 + j] = 0.5 * (px * px + py * py + pz * pz)
            du[zeta0 + j] = pz + 0.0im
        end
    end
    # end

    # @time begin
    # Active continuum source and feedback. Manual chunking removes the need for
    # `threadid()` inside the hot loop and gives each slot an exclusive reduction column.
    @threads :static for slot in 1:nt
        qlo, qhi = _chunk_bounds(nact, slot, nt)
        @inbounds for q in qlo:qhi
            j = dyn.active[q]

            px = g.kx[j]
            py = g.ky[j]
            pz = g.kz[j] + eta
            p2 = px * px + py * py + pz * pz
            pabs = sqrt(p2)
            costh = pabs > 0 ? pz / pabs : 0.0

            chi_phase = cis(real(u[chi0 + j]))
            Bj = u[c0 + j] * conj(chi_phase)
            src = 0.0 + 0.0im
            wj = g.w[j]

            @simd for α in 1:Nb
                d = continuum_dipole_z_uniform(pabs, costh, α, model.states, model.rtab)
                src += d * u[a0 + α] * cache.phase_plus[α]
                scratch.feedback_tls[α, slot] += wj * conj(d) * cache.phase_minus[α] * Bj
            end

            du[c0 + j] = IM * E * src * chi_phase
        end
    end
    # end

    # @time begin
    # Bound amplitudes.
    @threads :static for α in 1:Nb
        bb = 0.0 + 0.0im
        @inbounds @simd for β in 1:Nb
            bb += model.zBB[α, β] * cache.phase_plus[β] * cache.phase_minus[α] * u[a0 + β]
        end

        fb = 0.0 + 0.0im
        @inbounds @simd for slot in 1:nt
            fb += scratch.feedback_tls[α, slot]
        end

        @inbounds du[a0 + α] = IM * E * (bb + fb)
    end
    # end
    # println("end sfa_rhs_dynamic_nostore!")

    @inbounds du[eta_idx] = E + 0.0im
    return du
end

"""
    rk4_step_dynamic_nostore!(u, t, dt, model, scratch, dyn, cfg, cache) -> u

Fixed-step RK4 using `sfa_rhs_dynamic_nostore!`.
This is the fastest dynamic-grid RK4 path in this module when diagnostics do not need
`dPN` during the propagation stage.
"""
function rk4_step_dynamic_nostore!(
    u::Vector{ComplexF64},
    t::Real,
    dt::Real,
    model::SFAModel,
    scratch::SFAScratch,
    dyn::DynamicGridScratch,
    cfg::DynamicGridConfig,
    cache::DynamicFastRK4Cache,
)
    h = Float64(dt)
    select_active_indices!(dyn, u, t, model, cfg; dt = h)

    sfa_rhs_dynamic_nostore!(cache.k1, u, model, t, scratch, dyn, cache)

    _stage_state_dynamic_fast!(cache.tmp, u, cache.k1, 0.5 * h, model, dyn)
    sfa_rhs_dynamic_nostore!(cache.k2, cache.tmp, model, t + 0.5 * h, scratch, dyn, cache)

    _stage_state_dynamic_fast!(cache.tmp, u, cache.k2, 0.5 * h, model, dyn)
    sfa_rhs_dynamic_nostore!(cache.k3, cache.tmp, model, t + 0.5 * h, scratch, dyn, cache)

    _stage_state_dynamic_fast!(cache.tmp, u, cache.k3, h, model, dyn)
    sfa_rhs_dynamic_nostore!(cache.k4, cache.tmp, model, t + h, scratch, dyn, cache)

    _final_update_dynamic_fast!(u, cache, h, model, dyn)
    return u
end

end # module
