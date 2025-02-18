import Pkg
Pkg.activate(".")

using Revise
using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using Printf
using FFTW
using DSP

task_id = 1

# Define the function of Movement
F1(E, Z, x, y, z, rj) = -E - Z * rj / (x^2 + y^2 + z^2 + 1e-5) ^ (3 / 2)

# Core Parameters
Z = 1.0
Ip = 0.5

# CTMC Parameters
Δt = 0.2
filter_threshold = 2.0
p_min = -1.0
p_max = 1.0
p_delta = 0.005

# Laser Parameters.
E_fs = 0.0533 #* 1.414
E_thz = 0.00002
E_dc = 0.00002
ω_fs = 0.05693
ω_thz = ω_fs / 30
nc = 12
tau_fs = 200
tau_thz = get_1c_thz_delay_list(ω_fs, tau_fs, nc, ω_thz)[task_id]

Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.8)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="none", phase1=0.5pi)
Ex_window = dc_bias(1.0, 0, tau_fs, tmax, tmax * 2)
E_applied(t) = (Ex_thz(t) + E_dc) * Ex_window(t)

At_data_xyz, Et_data, ts, t_num = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light, appendix_steps=2001)
At_data_xyz_hf, Et_data_hf, ts_hf, = create_tdata(tmax, Δt/2, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light, appendix_steps=2001)
plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts)


# create runtime for CTMC
trajs_num = t_num
rt = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)

# Preparation for Start Point 
m = 1
tid_cc, pv_cc, theta_cc = generate_start_point_uniform_special(trajs_num, 1)
update_start_point(rt, trajs_num, tid_cc, pv_cc, theta_cc, Et_data, Ip, Z)


# CTMC Mainloop
traj_filter_flag = ctmc_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, filter_threshold)

# Add to PMD
add_to_pmd(rt, trajs_num, t_num, Z)

# Add to HHG
add_to_hhg(rt, trajs_num, t_num, Δt, tid_cc, traj_filter_flag)

# Trajs Analyse
ap1, ap2, ap3 = trajs_analyse(rt, Et_data, trajs_num, tmax, t_num, ts, Δt, ω_fs, tid_cc, traj_filter_flag, nc)

# rad2deg(angle(tmp1))
# rad2deg(angle(tmp2))



############ <===3

# HHG
p2, hhg_spectrum_x, hhg_spectrum_y, base_id, hhg_k_linspace = ctmc_get_hhg_spectrum(rt, Et_data, tmax, t_num, ts, Δt, ω_fs)

# PMD
Plots.heatmap(p_axis, p_axis, rt.pxy_final, color=:jet1, size=(500, 420))