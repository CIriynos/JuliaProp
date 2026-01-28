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
using Statistics


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
ω_fs = 0.05693
nc = 15
tau_fs = 0
appendix_time = 0
Tp = 2 * nc * pi / ω_fs

# create runtime for CTMC
tmax = tau_fs + 2 * nc * pi / ω_fs + appendix_time
t_num = Int64(tmax ÷ Δt) + 1
trajs_num = Int64(tmax ÷ Δt) + 1
rt = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)

# Scanning E_thz_ratio and freq_ratio
# E_thz_ratio_list = 5: 5: 150
# thz_freq_list = 5: 5: 150      # unit: THz
E_thz_ratio_list = [1, 50, 100, 150, 200]
thz_freq_list = [10, 20, 50, 100]
mse_data = zeros(Float64, length(E_thz_ratio_list), length(thz_freq_list))
R2_data = zeros(Float64, length(E_thz_ratio_list), length(thz_freq_list))
record_data = []
plot_mat = []

for (j, E_thz_ratio) in enumerate(E_thz_ratio_list)
for (k, thz_freq) in enumerate(thz_freq_list)

println("Start Scanning. E_thz_ratio=$E_thz_ratio, thz_freq=$thz_freq")

# Define the THz wave
E_thz = 0.00002 * E_thz_ratio
E_dc = 0.00002 * E_thz_ratio
ω_thz = ω_fs * (thz_freq / 375)

# Get THz waveform for plotting and t1, t2
Ex_thz_tmp, _, _, tmax_tmp = light_pulse(ω_thz, E_thz, 1, 0, pulse_shape="sin2", phase1=0.5pi)
At_datas_tmp, Et_datas_tmp, ts_tmp = create_tdata(tmax_tmp + Tp/2, -Tp/2, Δt, Ex_thz_tmp, no_light, no_light)
thz_data = [Et_datas_tmp[1], ts_tmp]
thz_maxv, t1 = findmax(Et_datas_tmp[1])
thz_minv, t2 = findmin(Et_datas_tmp[1])
t1 = (t1 * Δt - Tp/2)
t2 = (t2 * Δt - Tp/2)

# Scanning 32 + 2 samples (the lastest 2 samples are max/min values, convenient for unifying)
for task_id = 1: (32 + 2)

# t-data for Field
tau_list = get_1c_thz_delay_list_combined(ω_fs, tau_fs, nc, ω_thz, 32 + 2, t1, t2)
tau_thz = tau_list[task_id]

Ex_fs, Ey_fs, Ez_fs, fs_tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.5, phase1=0.5pi, phase2=pi)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="sin2", phase1=0.5pi)
E_applied(t) = (Ex_thz(t) + E_dc) #* flap_top_windows_f(t, 0, tmax, 1/2)
At_data_xyz, Et_data, ts, t_num = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light)
At_data_xyz_hf, Et_data_hf, ts_hf, = create_tdata(tmax, Δt/2, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light)
At_data_xyz_no_thz, Et_data_no_thz, = create_tdata(tmax, 0, Δt, t -> Ex_fs(t), Ey_fs, no_light)
At_data_xyz_hf_no_thz, Et_data_hf_no_thz, = create_tdata(tmax, Δt/2, Δt, t -> Ex_fs(t), Ey_fs, no_light)

fs_thz_fig = plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts)
# plot(sqrt.(Et_data[1] .^ 2 + Et_data[2] .^ 2 + Et_data[3] .^ 2))

# Preparation for Start Point 
m = 1
tid_cc, pv_cc, theta_cc = generate_start_point_uniform_special(trajs_num, 1)
update_start_point(rt, trajs_num, tid_cc, pv_cc, theta_cc, Et_data, Ip, Z, Δt)

# Mainloop for CTMC
traj_filter_flag = ctmc_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, filter_threshold)
add_to_pmd(rt, trajs_num, t_num, Z, traj_filter_flag)
add_to_hhg(rt, trajs_num, t_num, Δt, tid_cc, traj_filter_flag)

# # Trajs Analyse
# ap1, ap2, ap3, shg_yield_data, traj_data, ft_ana = trajs_analyse(rt, Et_data, trajs_num, tmax, t_num, ts, Δt, ω_fs, tau_fs, tid_cc, traj_filter_flag, nc)
# push!(ap1s, ap1)
# push!(ap2s, ap2)
# push!(ap3s, ap3)
# push!(phase_data_list, shg_yield_data)
# push!(traj_ana_list, traj_data)
# push!(ft_ana_list, ft_ana)

# HHG
p1, p2, hhg_spectrum_x, hhg_spectrum_y, base_id, hhg_k_linspace_ctmc = ctmc_get_hhg_spectrum(rt, Et_data, tmax, t_num, ts, Δt, ω_fs, tau_fs)

# Record
# push!(hhg_plt_list_ctmc, p2)
push!(shg_yields, norm.(hhg_spectrum_x[base_id * 2]))
# push!(fs_thz_fig_list, fs_thz_fig)
# push!(hhg_data_list_ctmc, hhg_spectrum_x)

At_THz_datas, Et_THz_datas, = create_tdata(tmax, 0, Δt, t -> E_applied(t), no_light, no_light, appendix_steps=1)
push!(mid_Efs_record_ctmc, E_applied(tau_fs + nc * pi / ω_fs))

# Clear all
clear_ctmc_rt(rt)

end

unify(data) = (data .- minimum(data)) ./ (maximum(data) - minimum(data))

l1 = unify(shg_yields)
l2 = unify(mid_Efs_record_ctmc)
l1 .+= (l2[1] - l1[1])
p = scatter(unify(tau_list), l1)
scatter!(p, unify(tau_list), l2)
plot!(p, unify(thz_data[2]), unify(reverse(thz_data[1])))
p
push!(plot_mat, p)
push!(record_data, [l1, l2])

mse = mean((l2 .- l1) .^ 2)     # 均方误差
ss_res = sum((l2 .- l1) .^ 2)   # 残差平方和
ss_tot = sum((l2 .- mean(l1)) .^ 2) # 总平方和
R2 = 1 - ss_res / ss_tot

mse_data[j, k] = mse
R2_data[j, k] = R2

# clear buffer
shg_yields = []
mid_Efs_record_ctmc = []

end
end

contourf(thz_freq_list, E_thz_ratio_list, max.(0.89, R2_data),
    xlabel="ω_thz (THz)", ylabel="E_THz (arb.unit)",
    fontsize=16, tickfontsize=10)
contourf((mse_data))
heatmap(mse_data)