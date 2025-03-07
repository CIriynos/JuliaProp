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


ap1s = []
ap2s = []
ap3s = []
tau_list = []
hhg_plt_list_ctmc = []
hhg_data_list_ctmc = []
mid_Efs_record_ctmc = []
shg_yields = []
fs_thz_fig_list = []
hhg_k_linspace_ctmc = []

# Define the function of Movement
F1(E, Z, x, y, z, rj) = -E - Z * rj / (x^2 + y^2 + z^2 + 1e-5) ^ (3 / 2)

# Core Parameters
Z = 1.0
Ip = Z ^ 2 / 2

# CTMC Parameters
Δt = 0.2
filter_threshold = 1.0
p_min = -1.0
p_max = 1.0
p_delta = 0.005

# Define the Laser. 
E_fs = 0.0533
E_thz = 0.005
E_dc = 0.005
ω_fs = 0.05693
ω_thz = ω_fs / 20
nc = 15
tau_fs = 0


# create runtime for CTMC
t_num = Int64((tau_fs + 2 * nc * pi / ω_fs) ÷ Δt) + 1
trajs_num = t_num
rt = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)

# task_id = 1
for task_id = 1: 32

# t-data for Field
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz, samples_num=32)
tau_thz = tau_list[task_id]

Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.4, phase1=0.5pi, phase2=pi)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="sin2", phase1=0.5pi)
E_applied(t) = (Ex_thz(t) + E_dc) #* flap_top_windows_f(t, 0, tmax, 1/2)
At_data_xyz, Et_data, ts, t_num = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light)
At_data_xyz_hf, Et_data_hf, ts_hf, = create_tdata(tmax, Δt/2, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light)
At_data_xyz_no_thz, Et_data_no_thz, = create_tdata(tmax, 0, Δt, t -> Ex_fs(t), Ey_fs, no_light)
At_data_xyz_hf_no_thz, Et_data_hf_no_thz, = create_tdata(tmax, Δt/2, Δt, t -> Ex_fs(t), Ey_fs, no_light)

fs_thz_fig = plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts)
plot(sqrt.(Et_data[1] .^ 2 + Et_data[2] .^ 2 + Et_data[3] .^ 2))

# Preparation for Start Point 
m = 1
tid_cc, pv_cc, theta_cc = generate_start_point_uniform_special(trajs_num, 1)
update_start_point(rt, trajs_num, tid_cc, pv_cc, theta_cc, Et_data_no_thz, Ip, Z, Δt)

# Mainloop for CTMC
traj_filter_flag = ctmc_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, filter_threshold)
add_to_pmd(rt, trajs_num, t_num, Z, traj_filter_flag)
add_to_hhg(rt, trajs_num, t_num, Δt, tid_cc, traj_filter_flag)

# # Trajs Analyse
# ap1, ap2, ap3, shg_yield_data = trajs_analyse(rt, Et_data, trajs_num, tmax, t_num, ts, Δt, ω_fs, tau_fs, tid_cc, traj_filter_flag, nc)
# push!(ap1s, ap1)
# push!(ap2s, ap2)
# push!(ap3s, ap3)

# HHG
p1, p2, hhg_spectrum_x, hhg_spectrum_y, base_id, hhg_k_linspace_ctmc = ctmc_get_hhg_spectrum(rt, Et_data, tmax, t_num, ts, Δt, ω_fs, tau_fs)

# Record
push!(hhg_plt_list_ctmc, p2)
push!(shg_yields, norm.(hhg_spectrum_x[base_id * 2]))
push!(fs_thz_fig_list, fs_thz_fig)
push!(hhg_data_list_ctmc, hhg_spectrum_x)

At_THz_datas, Et_THz_datas, = create_tdata(tmax, 0, Δt, t -> E_applied(t), no_light, no_light, appendix_steps=1)
push!(mid_Efs_record_ctmc, E_applied(tau_fs + nc * pi / ω_fs))

# Clear all
clear_ctmc_rt(rt)

end

plot(tau_list, shg_yields)

unify(data) = (data .- minimum(data)) ./ (maximum(data) - minimum(data))
p = plot(unify(tau_list), unify(shg_yields))
plot!(p, unify(tau_list), unify(mid_Efs_record_ctmc))

# hhg_plt_list_ctmc[1]

# 0.755
# bd1 = 150
# bd2 = 200
# ppp = plot(range(1, bd1) .* 1.0, norm.(hhg_data_x_list[1])[1:bd1], yscale=:log10)
# plot!(ppp, range(1, bd2) .* 0.755, norm.(hhg_data_list_ctmc[1])[1:bd2] .* 1e1, yscale=:log10)
# plot!(ppp, range(1, bd2) .* 0.755, norm.(theory_hhg_data_list[1])[1:bd2] .* 1e1, yscale=:log10)
# ppp

# f = open("phase_fig_3.txt", "w+")
# for i = 1: length(shg_yield_data)
#     write(f, "$(angle.(shg_yield_data[i])) $(norm.(shg_yield_data[i]))\n")
# end
# close(f)


# # PMD
# Plots.heatmap(p_axis, p_axis, rt.pxy_final, color=:jet1, size=(500, 420))