include("./SFA.jl")

import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW
using SparseArrays
using SpecialFunctions

using .ExtendedSFA

# ============================================================
# 0. User-provided inputs
# ============================================================

# You already have this:
# E_field(t) = ...
@expo E_fs =          0.04                 # peak electric field of the fs pulse
@expo ω_fs =          0.057 * 1           # angular frequency of the fs pulse
@expo nc =            6                  # number of optical cycles in the fs pulse
Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, 0, ellipticity=0.0, phase1=0.0)        # create the light pulse from the given parameters (+ ellipticity)
E_field(t) = Ex_fs(t)

# Required user-side data:
# dr::Float64
# Nr::Int
# eigen_data = [
#     (n = 1, l = 0, I = I10, u = u10_complex),
#     (n = 2, l = 1, I = I21, u = u21_complex),
#     ...
# ]
ratio = 1
Nr =            20000 * ratio        # number of radial grid points
Δr =            0.2 / ratio          # radial grid step size
l_num =         5                    # number of angular momentum components
Δt =            0.05 / ratio         # time step size
Z =             1.0                  # nuclear charge
# po_func(r) =    -1 / r             # potential function
po_func(r) =    -1 / r * exp(- r * r / (20.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
rmax =          Nr * Δr     
absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function
Ri_tsurf        = rmax * 0.7        # radius for t-surf method

dr::Float64 = Δr

# Momentum transform grid for radial Bessel transforms
pmax = 5.0
Np = 2000
pgrid = collect(range(0.0, pmax, length = Np))

# Characteristic κ-grid
# kx_axis = collect(range(-kxmax, kxmax, length = Nkx))
# ky_axis = collect(range(-kymax, kymax, length = Nky))
# kz_axis = collect(range(-kzmax, kzmax, length = Nkz))
kx_axis = collect(range(-2.5, 2.5, length = 100))
ky_axis = collect(range(-2.5, 2.5, length = 100))
kz_axis = collect(range(-2.5, 2.5, length = 100))

# Time grid
# t0, tf, dt must be given in atomic units
t0 = 0.0
dt = 0.2
tf = (2 * nc * pi / ω_fs)
tgrid = collect(t0:dt:tf)
Nt = length(tgrid)

# Initial bound state index in `states`
ground_index = 1

# Spectrum grid
omega_min = 0
omega_max = ω_fs * 20
Nω = 1000
omega_grid = collect(range(omega_min, omega_max, length = Nω))


# ============================================================
# 1. Build model
# ============================================================

grid = radial_grid(dr, Nr)

# states = [
#     make_bound_state(n, l, I, u_complex, grid; normalize = true),
#     # ...
# ]
states = BoundState[]

# get the Ip of the system
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func);
rt = create_tdse_rt_sh(pw, m_zero_flag=true);
init_wave = create_empty_shwave(pw.shgrid)

# max_k = 20
max_k = 4

ek_list = []
for k = 1: max_k
    init_wave = create_empty_mzero_shwave(pw.shgrid)
    rs = get_linspace(pw.shgrid.rgrid)
    @. init_wave[k] = rs * exp(-rs * k)
    itp_fdsh_single(pw, rt, init_wave, k, err=1e-15, log_info=false)
    ek = get_energy_sh_so(init_wave, rt, k)

    # ek = -0.5 / (k ^ 2)
    push!(ek_list, ek)
    println("Energy of the state with k=$k: ", ek)

    init_wave = nothing 
    GC.gc(true)
    ccall(:malloc_trim, Cint, (Csize_t,), 0)
end

rs = get_linspace(pw.shgrid.rgrid)
pw = nothing
rt = nothing
GC.gc(true)
ccall(:malloc_trim, Cint, (Csize_t,), 0)

m_zero_flag = true

k = 1
for (n, ek) in enumerate(ek_list)
    pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func, delta_t_im = 2 / (-ek))
    rt = create_tdse_rt_sh(pw, m_zero_flag=true);

    for l in 0: min(n-1, l_num-1)
        for m in -l: l
            if m_zero_flag == true && m != 0
                id = get_index_from_lm(l, m, l_num)
                init_wave = create_empty_shwave(pw.shgrid)
            else
                id = get_index_from_lm(l, m, l_num)
                init_wave = create_empty_shwave(pw.shgrid)
                @. init_wave[id] = rs * exp(-rs * n)
                itp_fdsh_single(pw, rt, init_wave, id, log_info=false, mininum_loop_times=2000)
                en = get_energy_sh(init_wave, rt, pw.shgrid)

                # push!(states, BoundState(n, l, en, copy(init_wave[id])))
                push!(states, make_bound_state(n, l, -en, copy(init_wave[id]) ./ sqrt(Δr)))

                # write(file, "energy_state_k_$k", (init_wave[id]))
                println("(k = $k) Energy of the state with n=$n, l=$l, m=$m: ", en)
            end
            k += 1

            init_wave = nothing 
            GC.gc(true)
            ccall(:malloc_trim, Cint, (Csize_t,), 0)
        end
    end

    println("ended procedure.")
    pw = nothing
    rt = nothing
    GC.gc(true)
    ccall(:malloc_trim, Cint, (Csize_t,), 0)
    println("finished memory clear.")
end

foreach(s -> normalize_radial!(s, grid), states)

zBB = build_zBB(states, grid)

rtab = precompute_radial_transforms(states, grid, pgrid)

kgrid = build_kappa_grid(kx_axis, ky_axis, kz_axis)

model = SFAModel(
    grid,
    states,
    [s.I for s in states],
    zBB,
    kgrid,
    rtab,
    E_field,
)

save_model_hdf5("./data/sfa_model.h5", model)

model = load_model_hdf5("./data/sfa_model.h5", E_field)


# ============================================================
# 2. Allocate propagation buffers
# ============================================================

u = initial_state(model; ground_index = ground_index)

scratch = make_scratch(model)

dyn = make_dynamic_scratch(model)

cache = make_dynamic_fast_rk4_cache(model)

obs = init_observable_buffer(Nt)


# ============================================================
# 3. Dynamic-grid RK4 configuration
# ============================================================

kperp_cut = 1.0
pabs_cut = 3.0
cfg = DynamicGridConfig(
    amp_threshold = 1e-8,   # ignore points with very small C_j
    field_threshold = 1e-10, # disable source gate when field is negligible
    kperp_max = kperp_cut,   # transverse κ cutoff, chosen by you
    pabs_max = pabs_cut,     # physical |p_j(t)| cutoff, chosen by you
    pad_kz = 1,              # add one neighbor along κz
    keep_previous = false,
)


# ============================================================
# 4. RK4 propagation and observable storage
# ============================================================

for it in 1: Nt
    t = tgrid[it]

    # Select active continuum points for diagnostics at this time.
    select_active_indices!(dyn, u, t, model, cfg; dt = dt)

    D = dipole_channels_dynamic(u, t, model, scratch, dyn)
    P = populations(u, model)

    obs.t[it] = t
    obs.D_BB[it] = D.D_BB
    obs.D_BC[it] = D.D_BC
    obs.D_CC[it] = D.D_CC
    obs.P_B[it] = P.bound
    obs.P_C[it] = P.continuum
    obs.P_tot[it] = P.total

    if it % 100 == 0
        println(
            "step = $it / $Nt, ",
            "t = $(round(t, digits=4)), ",
            "Nactive = $(D.N_active), ",
            "Ptot = $(round(P.total, digits=8))"
        )
    end

    # Do not step after the final stored point.
    if it < Nt
        rk4_step_dynamic_nostore!(u, t, dt, model, scratch, dyn, cfg, cache)
    end
end


# ============================================================
# 5. Save polarization / population results
# ============================================================

save_observables_hdf5("sfa_observables.h5", obs)

# Optional: save reusable model precomputations.
# Note: E_field itself is not saved; supply it again when loading.
save_model_hdf5("sfa_model.h5", model)


# ============================================================
# 6. Spectrum analysis
# ============================================================

D_tot = obs.D_BB .+ obs.D_BC .+ obs.D_CC

# Smooth Hann-like window
τ = (obs.t .- obs.t[1]) ./ (obs.t[end] - obs.t[1])
window = @. sin(pi * τ)^2

_, S_BB  = channel_spectrum(obs.t, obs.D_BB, omega_grid; window = window)
_, S_BC  = channel_spectrum(obs.t, obs.D_BC, omega_grid; window = window)
_, S_CC  = channel_spectrum(obs.t, obs.D_CC, omega_grid; window = window)
_, S_tot = channel_spectrum(obs.t, D_tot,    omega_grid; window = window)


# ============================================================
# 7. Visualization
# ============================================================

p1 = plot(
    obs.t,
    [obs.D_BB obs.D_BC obs.D_CC D_tot],
    label = ["D_BB" "D_BC" "D_CC" "D_total"],
    xlabel = "t / a.u.",
    ylabel = "D(t)",
    title = "Dipole channels",
)

savefig(p1, "dipole_channels.png")

p2 = plot(
    omega_grid,
    log10.(S_tot .+ eps()),
    label = "total",
    xlabel = "ω / a.u.",
    ylabel = "log10 spectrum",
    title = "Total harmonic spectrum",
)

plot!(p2, omega_grid, log10.(S_BB .+ eps()), label = "BB")
plot!(p2, omega_grid, log10.(S_BC .+ eps()), label = "BC")
plot!(p2, omega_grid, log10.(S_CC .+ eps()), label = "CC")

savefig(p2, "spectrum_channels.png")

p3 = plot(
    obs.t,
    [obs.P_B obs.P_C obs.P_tot],
    label = ["P_B" "P_C" "P_total"],
    xlabel = "t / a.u.",
    ylabel = "population",
    title = "Population check",
)

savefig(p3, "population_check.png")

display(p1)
display(p2)
display(p3)