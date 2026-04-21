import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW


# Basic Parameters
ratio = 1
Nr =            20000 * ratio           # number of radial grid points
Δr =            0.2 / ratio             # radial grid step size
l_num =         100                  # number of angular momentum components
Δt =            0.05 / ratio            # time step size
Z =             1.0                  # nuclear charge
# po_func(r) =    -1 / r              # potential function
po_func(r) =    -1 / r * exp(- r * r / (20.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
rmax =          Nr * Δr     
absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function
Ri_tsurf        = rmax * 0.7        # radius for t-surf method


# Create Physical World & Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func)
rt = create_tdse_rt_sh(pw);


# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-30);           # get initial wavefunctions by imaginary time propagation
crt_shwave = deepcopy(init_wave_list[1]);               # set the current wavefunction as the ground state
en = get_energy_sh(init_wave_list[1], rt, pw.shgrid)    # get energy of the initial wavefunction (For H atom, should be -0.5 a.u.)

# Define the Laser pulse
E_fs =          0.1                 # peak electric field of the fs pulse
E_thz =         0.00005 * 0         # peak electric field of the THz pulse
E_dc =          0.001 * 0           # static electric field
ω_fs =          0.057 * 1           # angular frequency of the fs pulse
ω_thz =         ω_fs / 30           # angular frequency of the THz pulse
nc =            12                  # number of optical cycles in the fs pulse
tau_fs =        0                   # delay of the fs pulse
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

###########################

# bound_state_num = sum([i^2 for i = 1: 10])
# example_name = "2026_3_28_test_itp_new_approach"
# eigen_states = retrieve_compact_bound_states(example_name, bound_state_num, pw.shgrid.l_num)

bound_state_num = sum([i^2 for i = 1: 4])
# example_name = "2026_4_20_test_itp_short_range"
example_name = "2026_4_20_test_itp_short_range" # ratio = 2
eigen_states = retrieve_compact_bound_states(example_name, bound_state_num, pw.shgrid.l_num)
eigen_states = []

# hhg_integral_t, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw, rt, At_datas[1], steps, Ri_tsurf)

# hhg_integral_t, hhg_integral_t_free, hhg_integral_t_bound = tdseln_sh_mainloop_velocity_gauge_hhg_analysis(crt_shwave, pw, rt, At_datas[1], steps, Ri_tsurf, eigen_states)

# hhg_integral_t, hhg_integral_t_free, hhg_integral_t_bound = tdseln_sh_mainloop_length_gauge_hhg_analysis(crt_shwave, pw, rt, Et_datas[1], steps, Ri_tsurf, eigen_states)

# hhg_integral_t, hhg_integral_t_free, hhg_integral_t_bound = tdseln_sh_mainloop_length_gauge_hhg_analysis_dipole(crt_shwave, pw, rt, Et_datas[1], steps, Ri_tsurf, eigen_states)

# crt_shwave_1 = deepcopy(init_wave_list[1]);   
# hhg_integral_t_1, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave_1, pw, rt, At_datas[1], steps, Ri_tsurf)

crt_shwave_2 = deepcopy(init_wave_list[1]);   
hhg_integral_t_2, hhg_integral_t_free, hhg_integral_t_bound = tdseln_sh_mainloop_length_gauge_hhg_analysis(crt_shwave_2, pw, rt, Et_datas[1], steps, Ri_tsurf, eigen_states)

crt_shwave_3 = deepcopy(init_wave_list[1]);   
hhg_integral_t_3, hhg_integral_t_free, hhg_integral_t_bound = tdseln_sh_mainloop_length_gauge_hhg_analysis_dipole(crt_shwave_3, pw, rt, Et_datas[1], steps, Ri_tsurf, eigen_states)


# # Store Data Manually
# example_name = "2026_4_20_hg"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "hhg_integral_t", hhg_integral_t)
#     write(file, "hhg_integral_t_free", hhg_integral_t_free)
#     write(file, "hhg_integral_t_bound", hhg_integral_t_bound)
# end

# # Store Data Manually
# example_name = "2026_4_20_hg"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "hhg_integral_t_1", hhg_integral_t_1)
#     write(file, "hhg_integral_t_2", hhg_integral_t_2)
#     write(file, "hhg_integral_t_3", hhg_integral_t_3)
# end

# # ###########################

# # Retrieve Data
# example_name = "2026_4_20_hg"
# hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")
# hhg_integral_t_free = retrieve_mat(example_name, "hhg_integral_t_free")
# hhg_integral_t_bound = retrieve_mat(example_name, "hhg_integral_t_bound")

# get harmonic spectrum, including data, and k axis (frequency axis)
n_cut_off_estim = floor((-en + 3.17 * (E_fs ^ 2.0 / (4.0 * (ω_fs ^ 2.0)))) / ω_fs) * 1
n_cut_off_estim = 1

# hhg_integral_t = hhg_integral_t_1 .+ Et_datas[1]
hhg_integral_t = hhg_integral_t_3
hg1, ks = get_hg_spectrum(ts, hhg_integral_t, ω_fs * (n_cut_off_estim + 20))
hg1_free, _ = get_hg_spectrum(ts, hhg_integral_t_free, ω_fs * (n_cut_off_estim + 20))
hg1_bound, _ = get_hg_spectrum(ts, hhg_integral_t_bound, ω_fs * (n_cut_off_estim + 20))
hg1_cross, _ = get_hg_spectrum(ts, hhg_integral_t .- hhg_integral_t_free .- hhg_integral_t_bound, ω_fs * (n_cut_off_estim + 20))

# r
plot(ks ./ ω_fs, [(ks .^ 3) .* hg1, (ks .^ 3) .* hg1_free, (ks .^ 3) .* hg1_bound],
    yscale=:log10, xaxis=1:20, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-8, 1e3),
    label=["Total" "Free" "Bound"], xlabel="Harmonic Order", ylabel="HG Intensity", title="HHG Spectrum (E0=$E_fs, ω=$ω_fs)")

# a
# plot(ks ./ ω_fs, [hg1 .* (1 ./ ks), hg1_free .* (1 ./ ks), hg1_bound .* (1 ./ ks)], yscale=:log10, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-8, 1e3), label=["Total" "Free" "Bound"], xlabel="Harmonic Order", ylabel="HG Intensity", title="HHG Spectrum")

# A
# plot(ks ./ ω_fs, [hg1 .* (1 ./ ks)], yscale=:log10, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-8, 1e3), label=["Total" "Free" "Bound"], xlabel="Harmonic Order", ylabel="HG Intensity", title="HHG Spectrum (E0=$E_fs, ω=$ω_fs)")

# r
plot(ks ./ ω_fs, [hg1 .* (ks .^ 3)], yscale=:log10, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-8, 1e3), label=["Total" "Free" "Bound"], xlabel="Harmonic Order", ylabel="HG Intensity", title="HHG Spectrum (E0=$E_fs, ω=$ω_fs)")


# convergence check
hg1, ks = get_hg_spectrum(ts, hhg_integral_t_1 .+ Et_datas[1], ω_fs * (n_cut_off_estim + 20))
hg2, _ = get_hg_spectrum(ts, hhg_integral_t_2 .+ Et_datas[1], ω_fs * (n_cut_off_estim + 20))
hg3, _ = get_hg_spectrum(ts, hhg_integral_t_3, ω_fs * (n_cut_off_estim + 20))
plot(ks ./ ω_fs, [hg1 .* (1 ./ ks), hg2 .* (1 ./ ks), hg3 .* (ks .^ 3)], yscale=:log10, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-8, 1e3), label=["Optimized" "Length Gauge" "Length Gauge Dipole"], xlabel="Harmonic Order", ylabel="HG Intensity", title="Convergence Check (E0=$E_fs, ω=$ω_fs)")