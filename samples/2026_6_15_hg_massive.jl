import Pkg
Pkg.activate(".")
# using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW


# Basic Parameters
@expo grid_ratio = 1
@expo Nr =            20000 * grid_ratio           # number of radial grid points
@expo Δr =            0.2 / grid_ratio             # radial grid step size
@expo l_num =         50                  # number of angular momentum components
@expo Δt =            0.05 / grid_ratio            # time step size
@expo Z =             1.0                  # nuclear charge
@expo po_func(r) =    -1 / r              # potential function
# po_func(r) =    -1 / r * exp(- r * r / (20.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
# po_func(r) =    -1 / r * exp(- r * r / (5.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
@expo rmax =          Nr * Δr     
@expo absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function
@expo Ri_tsurf        = rmax * 0.7        # radius for t-surf method


# Create Physical World & Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func)
rt = create_tdse_rt_sh(pw, m_zero_flag=true);

# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-30);           # get initial wavefunctions by imaginary time propagation
crt_shwave = deepcopy(init_wave_list[1]);               # set the current wavefunction as the ground state
en = get_energy_sh(init_wave_list[1], rt, pw.shgrid)    # get energy of the initial wavefunction (For H atom, should be -0.5 a.u.)


# Define the Laser pulse
@expo E_fs =          0.04                 # peak electric field of the fs pulse
@expo E_thz =         0.00005 * 0         # peak electric field of the THz pulse
@expo E_dc =          0.001 * 0           # static electric field
@expo ω_fs =          0.057 * 1            # angular frequency of the fs pulse
@expo ω_thz =         ω_fs / 30           # angular frequency of the THz pulse
@expo nc =            6                  # number of optical cycles in the fs pulse
@expo tau_fs =        0                   # delay of the fs pulse
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz)
                                    # A helper function to get the delays of the THz pulse, 
                                    # which ensures consecutively scanning through the whole duration of the fs pulse.
tau_thz = tau_list[1]               # select one delay from the delay list

# create the Laser pulse data
Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, 0, ellipticity=0.0, phase1=0.5pi)        # create the light pulse from the given parameters (+ ellipticity)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="sin2", phase1=0.5pi)       # create the THz pulse from the given parameters
E_applied(t) = (Ex_thz(t) + E_dc) * flap_top_windows_f(t, 0, tmax, 1/2)                 # define the applied electric field (THz + DC) with a flattop window
At_datas, Et_datas, ts, steps = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light, appendix_steps=1)
                                                                                        # create the vector potential and electric field data for propagation
plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts, thz_ratio=100)                                         # Visualize the superimposed electric field (fs + THz + DC)

###########################

bound_state_num = sum([i^2 for i = 1: 5])
# example_name = "2026_6_15_itp_short_range_5"
example_name = "2026_6_15_itp"
eigen_states = retrieve_compact_bound_states(example_name, bound_state_num, pw.shgrid.l_num)

hhg_integral_t, hhg_integral_t_free, hhg_integral_t_bound, norm_value, bound_norm_value, free_norm_value = tdseln_sh_mainloop_length_gauge_hhg_analysis_dipole(crt_shwave, pw, rt, Et_datas[1], steps, Ri_tsurf, eigen_states)

# Store Data Manually
example_name = "2026_6_15_hg_massive_$(E_fs)_$(E_dc)_$(ω_fs)_$(nc)_short_range"
h5open("./data/$example_name.h5", "w") do file
    # write(file, "crt_shwave", hcat(crt_shwave...))
    write(file, "hhg_integral_t", hhg_integral_t)
    write(file, "hhg_integral_t_free", hhg_integral_t_free)
    write(file, "hhg_integral_t_bound", hhg_integral_t_bound)
    write(file, "norm_value", norm_value)
    write(file, "bound_norm_value", bound_norm_value)
    write(file, "free_norm_value", free_norm_value)
end


# ###########################

# # Retrieve Data
# example_name = "2026_6_15_hg_massive_$(E_fs)_$(E_dc)_$(ω_fs)_$(nc)_short_range"
# hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")
# hhg_integral_t_free = retrieve_mat(example_name, "hhg_integral_t_free")
# hhg_integral_t_bound = retrieve_mat(example_name, "hhg_integral_t_bound")
# retrieve_mat(example_name, "free_norm_value")

# get harmonic spectrum, including data, and k axis (frequency axis)
n_cut_off_estim = floor((-en + 3.17 * (E_fs ^ 2.0 / (4.0 * (ω_fs ^ 2.0)))) / ω_fs) * 1
n_cut_off_estim = 20

hg1, ks = get_hg_spectrum(ts, hhg_integral_t, ω_fs * (n_cut_off_estim + 20))
hg1_free, _ = get_hg_spectrum(ts, hhg_integral_t_free, ω_fs * (n_cut_off_estim + 20))
hg1_bound, _ = get_hg_spectrum(ts, hhg_integral_t_bound, ω_fs * (n_cut_off_estim + 20))
hg1_cross, _ = get_hg_spectrum(ts, hhg_integral_t .- hhg_integral_t_free .- hhg_integral_t_bound, ω_fs * (n_cut_off_estim + 20))

# r
plot(ks ./ ω_fs, [(ks .^ 3) .* hg1, (ks .^ 3) .* hg1_free, (ks .^ 3) .* hg1_bound],
    yscale=:log10, xaxis=1:20, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-8, 1e3),
    label=["Total" "Free" "Bound" "Cross"], xlabel="Harmonic Order", ylabel="HG Intensity", title="HHG Spectrum (E0=$E_fs, ω=$ω_fs)")