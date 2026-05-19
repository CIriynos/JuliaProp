import Pkg
Pkg.activate(".")
# using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW
using SparseArrays
using SpecialFunctions

using LinearAlgebra
BLAS.set_num_threads(1)     # recommended when using Julia threads
include("SFA.jl")
using .ExtendedSFA  


#################################################


# Basic Parameters
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


grid = radial_grid(Δr, Nr)

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


# Define the Laser pulse
@expo E_fs =          0.05                 # peak electric field of the fs pulse
@expo E_thz =         0.00005 * 0         # peak electric field of the THz pulse
@expo E_dc =          0.001 * 0           # static electric field
@expo ω_fs =          0.057 * 1           # angular frequency of the fs pulse
@expo ω_thz =         ω_fs / 30           # angular frequency of the THz pulse
@expo nc =            12                  # number of optical cycles in the fs pulse
@expo tau_fs =        0                   # delay of the fs pulse
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz)
                                    # A helper function to get the delays of the THz pulse, 
                                    # which ensures consecutively scanning through the whole duration of the fs pulse.
tau_thz = tau_list[1]               # select one delay from the delay list

# create the Laser pulse data
Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.0, phase1=0.0)        # create the light pulse from the given parameters (+ ellipticity)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="sin2", phase1=0.5pi)       # create the THz pulse from the given parameters
E_applied(t) = (Ex_thz(t) + E_dc) * flap_top_windows_f(t, 0, tmax, 1/2)                 # define the applied electric field (THz + DC) with a flattop window
At_datas, Et_datas, ts, steps = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light, appendix_steps=1)
                                                                                        # create the vector potential and electric field data for propagation
plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts, thz_ratio=100)                                         # Visualize the superimposed electric field (fs + THz + DC)


foreach(s -> normalize_radial!(s, grid), states)


# Define the Laser pulse
@expo E_fs =          0.1                 # peak electric field of the fs pulse
@expo E_thz =         0.00005 * 0         # peak electric field of the THz pulse
@expo E_dc =          0.001 * 0           # static electric field
@expo ω_fs =          0.057 * 1           # angular frequency of the fs pulse
@expo ω_thz =         ω_fs / 30           # angular frequency of the THz pulse
@expo nc =            6                  # number of optical cycles in the fs pulse
@expo tau_fs =        0                   # delay of the fs pulse
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz)
                                    # A helper function to get the delays of the THz pulse, 
                                    # which ensures consecutively scanning through the whole duration of the fs pulse.
tau_thz = tau_list[1]               # select one delay from the delay list

# create the Laser pulse data
Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.0, phase1=0.0)        # create the light pulse from the given parameters (+ ellipticity)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="sin2", phase1=0.5pi)       # create the THz pulse from the given parameters
E_applied(t) = (Ex_thz(t) + E_dc) * flap_top_windows_f(t, 0, tmax, 1/2)                 # define the applied electric field (THz + DC) with a flattop window
At_datas, Et_datas, ts, steps = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light, appendix_steps=1)
                                                                                        # create the vector potential and electric field data for propagation
plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts, thz_ratio=100)                                         # Visualize the superimposed electric field (fs + THz + DC)


zBB = build_zBB(states, grid)

pmax = 5.0
Np = 2000
pgrid = collect(range(0.0, pmax, length = Np))
rtab = precompute_radial_transforms(states, grid, pgrid)

delta_kappa = 0.05
kx_axis = -2.5: delta_kappa: 2.5
ky_axis = -2.5: delta_kappa: 2.5
kz_axis = -2.5: delta_kappa: 2.5
kgrid = build_kappa_grid(kx_axis, ky_axis, kz_axis)

Efield(t) = Ex_fs(t)

model = SFAModel(
    grid,
    states,
    [s.I for s in states],
    zBB,
    kgrid,
    rtab,
    Efield,
)

dyn = DynamicGridScratch = make_dynamic_scratch(model)
cache = make_dynamic_rk4_cache(model)

cfg = DynamicGridConfig(
    amp_threshold = 1e-10,
    field_threshold = 1e-8,
    kperp_max = 0.5,
    pabs_max = 3.0,
    pad_kz = 1,
    keep_previous = false,
)

t = 0
dt = 0.2
Nt = Int64(floor((2 * pi * nc) / (ω_fs) / dt)) + 1
for it in 1: Nt
    rk4_step_dynamic!(u, t, dt, model, scratch, dyn, cfg, cache)
    t += dt
    if it % 100 == 0
        println("it = $it")
    end
end

select_active_indices!(dyn, u, t, model, cfg; dt = dt)
D = dipole_channels_dynamic(u, t, model, scratch, dyn)

populations(u, model)

println(D.N_active)


# get harmonic spectrum, including data, and k axis (frequency axis)
n_cut_off_estim = floor((-ek_list[1] + 3.17 * (E_fs ^ 2.0 / (4.0 * (ω_fs ^ 2.0)))) / ω_fs) * 1 + 20
# n_cut_off_estim = 20

hg1, ks = get_hg_spectrum(ts, Dtot, ω_fs * (n_cut_off_estim + 20))
hg1_free, _ = get_hg_spectrum(ts, DCC, ω_fs * (n_cut_off_estim + 20))
hg1_bound, _ = get_hg_spectrum(ts, DBB, ω_fs * (n_cut_off_estim + 20))
hg1_cross, _ = get_hg_spectrum(ts, DBC, ω_fs * (n_cut_off_estim + 20))

# r
plot(ks ./ ω_fs, [(ks .^ 3) .* hg1, (ks .^ 3) .* hg1_free, (ks .^ 3) .* hg1_bound],
    yscale=:log10, xaxis=1:20, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-12, 1e3),
    label=["Total" "Free" "Bound"], xlabel="Harmonic Order", ylabel="HG Intensity", title="HHG Spectrum (E0=$E_fs, ω=$ω_fs)")

# plot(norm.(u[1: 1000]))