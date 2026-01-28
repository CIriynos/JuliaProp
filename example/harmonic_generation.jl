import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# 


# Basic Parameters
Nr =            5000           # number of radial grid points
Δr =            0.2             # radial grid step size
l_num =         50              # number of angular momentum components
Δt =            0.05            # time step size
Z =             1.0             # nuclear charge
po_func(r) =    -1 / r          # potential function
rmax =          Nr * Δr     
absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function


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
E_dc =          0.00005 * 0         # static electric field
ω_fs =          0.057               # angular frequency of the fs pulse
ω_thz =         ω_fs / 30           # angular frequency of the THz pulse
nc =            6                   # number of optical cycles in the fs pulse
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

# Main Propagation Loop of TDSE with HHG Recording
hhg_integral_t, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw, rt, At_datas[1], steps, Ri_tsurf);


# Store Data Manually
example_name = "harmonic_generation"
h5open("./data/$example_name.h5", "w") do file
    write(file, "crt_shwave", hcat(crt_shwave...))
    write(file, "hhg_integral_t", hhg_integral_t)
end

###########################

# Retrieve Data
example_name = "harmonic_generation"
hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")

# get harmonic spectrum, including data, and k axis (frequency axis)
hg1, ks = get_hg_spectrum(ts, -hhg_integral_t - Et_datas[1], ω_fs * 65)

plot(ks ./ ω_fs, hg1 .* ks .* 3e3, yscale=:log10, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-4, 1e4))

# The figure should be as the same as P74 Fig.9 (a)