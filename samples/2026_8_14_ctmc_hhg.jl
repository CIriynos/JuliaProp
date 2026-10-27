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
phase_data_list = []
thz_data = []
traj_ana_list = []
ft_ana_list = []
hhg_t_data = []

# Define the function of Movement
a0 = 0.5
F1(E, Z, x, y, z, rj) = -E - Z * rj / (x^2 + y^2 + z^2 + 1e-5) ^ (3 / 2) #* exp(-0.1 * (x^2 + y^2 + z^2 + a0) ^ (0.5))

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
E_thz = 0.00002 * 10 * 0
E_dc = 0.00002 * 10
ω_fs = 0.05693
ω_thz = ω_fs / 25
nc = 15
tau_fs = 0
appendix_time = 0

# create runtime for CTMC
tmax = tau_fs + 2 * nc * pi / ω_fs + appendix_time
t_num = Int64(tmax ÷ Δt) + 1
trajs_num = Int64(tmax ÷ Δt) + 1
rt = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)

# task_id = 1
for E_dc_scale_id in 1: 10

E_dc = 0.0002 * (10 ^ range(-3, 3, 10)[E_dc_scale_id])

for task_id = [1, 2, 3, 4]

# t-data for Field
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz, samples_num=32 + 2)
tau_thz = tau_list[1]

Ex_fs, Ey_fs, Ez_fs, fs_tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.5, phase1=0.5pi, phase2=0)
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
if task_id == 1 || task_id == 2
    update_start_point(rt, trajs_num, tid_cc, pv_cc, theta_cc, Et_data, Ip, Z, Δt)
else
    update_start_point(rt, trajs_num, tid_cc, pv_cc, theta_cc, Et_data_no_thz, Ip, Z, Δt)
end

# Mainloop for CTMC
if task_id == 1 || task_id == 3
    traj_filter_flag = ctmc_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, filter_threshold)
else
    traj_filter_flag = ctmc_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data_no_thz, Et_data_hf_no_thz, Z, filter_threshold)
end
add_to_pmd(rt, trajs_num, t_num, Z, traj_filter_flag)
add_to_hhg(rt, trajs_num, t_num, Δt, tid_cc, traj_filter_flag)

# Trajs Analyse
ap1, ap2, ap3, shg_yield_data, traj_data = trajs_analyse(rt, Et_data, trajs_num, tmax, t_num, ts, Δt, ω_fs, tau_fs, tid_cc, traj_filter_flag, nc)
push!(ap1s, ap1)
push!(ap2s, ap2)
push!(ap3s, ap3)
push!(phase_data_list, shg_yield_data)
push!(traj_ana_list, traj_data)
# push!(ft_ana_list, ft_ana)

# HHG
p1, p2, hhg_spectrum_x, hhg_spectrum_y, base_id, hhg_k_linspace_ctmc = ctmc_get_hhg_spectrum(rt, Et_data, tmax, t_num, ts, Δt, ω_fs, tau_fs)

# Record
push!(hhg_plt_list_ctmc, p2)
push!(shg_yields, norm.(hhg_spectrum_x[base_id * 2]))
push!(fs_thz_fig_list, fs_thz_fig)
push!(hhg_data_list_ctmc, hhg_spectrum_x)
push!(hhg_t_data, deepcopy(rt.ax_data))

# At_THz_datas, Et_THz_datas, = create_tdata(tmax, 0, Δt, t -> E_applied(t), no_light, no_light, appendix_steps=1)
# push!(mid_Efs_record_ctmc, E_applied(tau_fs + nc * pi / ω_fs))

Tp = 2 * nc * pi / ω_fs
Ex_thz_tmp, Ey_thz_tmp, Ez_thz_tmp, tmax_tmp = light_pulse(ω_thz, E_thz, 1, 0, pulse_shape="sin2", phase1=0.5pi)
At_datas_tmp, Et_datas_tmp, ts_tmp = create_tdata(tmax_tmp + Tp/2, -Tp/2, 10*Δt, Ex_thz_tmp, no_light, no_light)
thz_data = [Et_datas_tmp[1], ts_tmp]

# Clear all
clear_ctmc_rt(rt)

end

l1 = norm.(hhg_data_list_ctmc[1][1:100])
l2 = norm.(hhg_data_list_ctmc[2][1:100])
l3 = norm.(hhg_data_list_ctmc[3][1:100])
l4 = norm.(hhg_data_list_ctmc[4][1:100])
ks = hhg_k_linspace_ctmc[1:100]
plot(hhg_k_linspace_ctmc[1:100] ./ ω_fs, [l1 l2 l3 l4], yscale=:log10)

f = open("fig3_b_data_$E_dc_scale_id.txt", "w+")
for i = 1: length(l1)
    write(f, "$(ks[i]) $(l1[i]) $(l2[i]) $(l3[i]) $(l4[i])\n")
end
close(f)

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
phase_data_list = []
thz_data = []
traj_ana_list = []
ft_ana_list = []
hhg_t_data = []

end


# f = open("single_traj_ctmc_first_2.txt", "w+")
# for i = 1: length(traj_ana_list[2][:, 4])
#     write(f, "$(real(traj_ana_list[2][:, 4][i]))\n")
# end
# close(f)

# plot([traj_ana_list[1][:, 5] traj_ana_list[2][:, 5]])

