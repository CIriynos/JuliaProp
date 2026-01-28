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

L1 = []
L2 = []
L3 = []

k = 1
for nc = 1: 0.25: 10
for cep = 0: 0.25pi: 2pi
# nc = 6
# cep = 0.0pi
println("nc = $nc, cep = $cep")

ap1s = []
ap2s = []
ap3s = []
tau_list = []
hhg_plt_list_ctmc = []
hhg_data_list_ctmc = []
mid_Efs_record_ctmc = []
shg_yields = []
thg_yields = []
fs_thz_fig_list = []
hhg_k_linspace_ctmc = []
phase_data_list = []
thz_data = []
traj_ana_list = []
ft_ana_list = []
hhg_t_data = []
Et_data_list = []

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
E_thz = 0.00002 * 0
E_dc = 0.00005 * 0
nc_thz = 2
ω_fs = 0.05693
ω_thz = ω_fs / 4

# make sure the time grid is the same.
Tp_fs = 2 * nc * pi / ω_fs
max_Tp_fs = 2 * 12 * pi / ω_fs
tau_fs = (max_Tp_fs - Tp_fs) / 2
appendix_time = (max_Tp_fs - Tp_fs) / 2

# harmonics of THz, for wide-spectrum detection
Tp_thz = 2pi * nc_thz / ω_thz
ω_thz_harmonic_list = [k * ω_thz for k = 2: 15]
nc_thz_harmonic_list = (Tp_thz / 2pi) .* ω_thz_harmonic_list
# phase_thz_harmonic_list = rand(length(ω_thz_harmonic_list)) .* 2pi
phase_thz_harmonic_list = ones(length(ω_thz_harmonic_list)) .* 0.5pi


# create runtime for CTMC
tmax = tau_fs + 2 * nc * pi / ω_fs + appendix_time
t_num = Int64(tmax ÷ Δt) + 1
trajs_num = Int64(tmax ÷ Δt) + 1
rt = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)
println("tnum = $t_num")

tasks_num = 100
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz, samples_num=tasks_num, nc_thz=nc_thz)
for task_id = [1]

# t-data for Field
tau_thz = tau_list[task_id]

Ex_fs, Ey_fs, Ez_fs = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.5, phase1=0.5pi + cep, phase2=0 + cep)
mid_point = tau_fs + nc*pi/ω_fs - pi/ω_thz
E_thz_1, E_thz_2, E_thz_3, tmax_thz = light_pulse(ω_thz, E_thz, nc_thz, tau_thz, pulse_shape="sin2", phase1=0.5pi)
E_thz_k1, _, _, _ = light_pulse(ω_thz_harmonic_list[1], E_thz / 1, nc_thz_harmonic_list[1], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[1])
E_thz_k2, _, _, _ = light_pulse(ω_thz_harmonic_list[2], E_thz / 1, nc_thz_harmonic_list[2], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[2])
E_thz_k3, _, _, _ = light_pulse(ω_thz_harmonic_list[3], E_thz / 1, nc_thz_harmonic_list[3], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[3])
# E_thz_k4, _, _, _ = light_pulse(ω_thz_harmonic_list[4], E_thz / 1, nc_thz_harmonic_list[4], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[4])
# E_thz_k5, _, _, _ = light_pulse(ω_thz_harmonic_list[5], E_thz / 1, nc_thz_harmonic_list[5], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[5])
# E_thz_k6, _, _, _ = light_pulse(ω_thz_harmonic_list[6], E_thz / 1, nc_thz_harmonic_list[6], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[6])
# E_thz_k7, _, _, _ = light_pulse(ω_thz_harmonic_list[7], E_thz / 1, nc_thz_harmonic_list[7], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[7])
# E_thz_k8, _, _, _ = light_pulse(ω_thz_harmonic_list[8], E_thz / 1, nc_thz_harmonic_list[8], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[8])
# E_thz_k9, _, _, _ = light_pulse(ω_thz_harmonic_list[9], E_thz / 1, nc_thz_harmonic_list[9], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[9])
# E_thz_k10, _, _, _ = light_pulse(ω_thz_harmonic_list[10], E_thz / 1, nc_thz_harmonic_list[10], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[10])
# E_thz_k11, _, _, _ = light_pulse(ω_thz_harmonic_list[11], E_thz / 1, nc_thz_harmonic_list[11], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[11])

E_applied(t) = ((E_thz_1(t) + E_thz_k1(t) + E_thz_k2(t) + E_thz_k3(t)) / 2 + E_dc) #* E_window(t)
# E_applied(t) = ((E_thz_1(t) + E_thz_k1(t) + E_thz_k2(t) + E_thz_k3(t) +
#     E_thz_k4(t) + E_thz_k5(t) + E_thz_k6(t) + E_thz_k7(t) + E_thz_k8(t) +
#     E_thz_k9(t) + E_thz_k10(t) + E_thz_k11(t)) / 6 + E_dc) #* E_window(t)

At_data_xyz, Et_data, ts, t_num = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light)
At_data_xyz_hf, Et_data_hf, ts_hf, = create_tdata(tmax, Δt/2, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light)
At_data_xyz_no_thz, Et_data_no_thz, = create_tdata(tmax, 0, Δt, t -> Ex_fs(t), Ey_fs, no_light)
At_data_xyz_hf_no_thz, Et_data_hf_no_thz, = create_tdata(tmax, Δt/2, Δt, t -> Ex_fs(t), Ey_fs, no_light)

fs_thz_fig = plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts)
plot(sqrt.(Et_data[1] .^ 2 + Et_data[2] .^ 2 + Et_data[3] .^ 2))

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
hhg_delta_k = 2pi / t_num / Δt
shg_id = Int64(floor((ω_fs * 2) ÷ hhg_delta_k) + 1)
thg_id = Int64(floor((ω_fs * 3) ÷ hhg_delta_k) + 1)
p1, p2, hhg_spectrum_x, hhg_spectrum_y, base_id, hhg_k_linspace_ctmc = ctmc_get_hhg_spectrum(rt, Et_data, tmax, t_num, ts, Δt, ω_fs, tau_fs)

# Record
push!(hhg_plt_list_ctmc, p2)
push!(shg_yields, hhg_spectrum_x[shg_id])
push!(thg_yields, hhg_spectrum_x[thg_id])
push!(fs_thz_fig_list, fs_thz_fig)
push!(hhg_data_list_ctmc, hhg_spectrum_x)
push!(hhg_t_data, deepcopy(rt.ax_data))
push!(Et_data_list, Et_data)

At_THz_datas, Et_THz_datas, = create_tdata(tmax, 0, Δt, t -> E_applied(t), no_light, no_light, appendix_steps=1)
push!(mid_Efs_record_ctmc, E_applied(tau_fs + nc * pi / ω_fs))

# Tp = 2 * nc * pi / ω_fs
# Ex_thz_tmp, Ey_thz_tmp, Ez_thz_tmp, tmax_tmp = light_pulse(ω_thz, E_thz, 1, 0, pulse_shape="sin2", phase1=0.5pi)
# At_datas_tmp, Et_datas_tmp, ts_tmp = create_tdata(tmax_tmp + Tp/2, -Tp/2, 10*Δt, Ex_thz_tmp, no_light, no_light)
# thz_data = [Et_datas_tmp[1], ts_tmp]

# Clear all
clear_ctmc_rt(rt)

end

push!(L1, shg_yields)
push!(L2, mid_Efs_record_ctmc)
push!(L3, thg_yields)
# plot(tau_list, [shg_yields .- shg_yields[1] mid_Efs_record_ctmc .- mid_Efs_record_ctmc[1]])

# using Interpolations
# a1 = ((shg_yields .- shg_yields[1]))
# a2 = (mid_Efs_record_ctmc .- mid_Efs_record_ctmc[1])
# plot([reverse(a1) .* 0.5 a2])
# interp_cub_1 = cubic_spline_interpolation(range(tau_list[1], tau_list[tasks_num], tasks_num), a1[1: (tasks_num)])
# interp_cub_2 = cubic_spline_interpolation(range(tau_list[1], tau_list[tasks_num], tasks_num), a2[1: (tasks_num)])

# ts = tau_list[1]: (Δt): tau_list[tasks_num]
# interp_res_1 = (interp_cub_1(ts))
# interp_res_2 = (interp_cub_2(ts))
# plot([reverse(interp_res_1) interp_res_2])

# hhg_delta_k = 2pi / length(ts) / Δt
# hhg_k_linspace = [hhg_delta_k * (i - 1) for i = 1: length(ts)]
# spec1 = (fft(interp_res_1 .* hanning(length(ts))))
# spec2 = (fft(interp_res_2 .* hanning(length(ts))))
# base_id = Int64((ω_thz) ÷ hhg_delta_k) + 1
# ratio = spec1[base_id] / spec2[base_id]
# spec2[1] = spec1[1] / ratio
# upper_limit_of_hhg = 10.0 * ω_thz
# rg = 1: (Int64((upper_limit_of_hhg) ÷ hhg_delta_k) + 1)
# display_rg = 1: (Int64((13.0 * ω_thz) ÷ hhg_delta_k) + 1)
# norm_cp_data = 20 .* log10.(norm.(spec1) ./ norm.(spec2 .* ratio))
# angle_cp_data = unwrap(angle.(spec1) .- angle.(spec2 * ratio))

# plot(hhg_k_linspace[display_rg] ./ (ω_fs / 375), [norm.(spec1)[display_rg] norm.(ratio .* spec2)[display_rg]], yscale=:log10, ylimits=(1e-5, 1e3))

# plot(hhg_k_linspace[display_rg] ./ (ω_fs / 375), [norm_cp_data[display_rg]], ylimits=(-10, 10))
# plot(hhg_k_linspace[display_rg] ./ (ω_fs / 375), [angle_cp_data[display_rg]], ylimits=(-pi, pi))

# interp_cub_f = cubic_spline_interpolation(0: hhg_delta_k: upper_limit_of_hhg, norm_cp_data[rg])
# xs = 0: (hhg_delta_k / 10): (upper_limit_of_hhg - hhg_delta_k)
# ys = interp_cub_f(xs)
# cutoff_id_list = []
# critical_cut_off_id = last(rg)
# for i = 2: length(xs)
#     if (ys[i] <= -3.0 && ys[i - 1] >= -3.0) || (ys[i] >= -3.0 && ys[i - 1] <= -3.0)
#         push!(cutoff_id_list, i)
#     end
#     if ys[i] <= -10.0
#         critical_cut_off_id = min(i ÷ 10, critical_cut_off_id)
#     end
# end
# cutoff_freq_list = @. (cutoff_id_list - 1) * (hhg_delta_k / 10) / (ω_fs / 375)

# # RSS
# rg_phi = 1: critical_cut_off_id
# a_phi = sum(hhg_k_linspace[rg_phi] .* angle_cp_data[rg_phi]) / sum(hhg_k_linspace[rg_phi] .^ 2)
# angle_data_linear_regression = hhg_k_linspace[rg_phi] .* a_phi
# rss_result = sum((angle_data_linear_regression .- angle_cp_data[rg_phi]) .^ 2)
# plot([angle_cp_data[rg_phi] angle_data_linear_regression])

# println("Nc = $nc, CEP = $cep: \n\tcutoff_freq(-3 dB) = $cutoff_freq_list, \n\tFR_angle_RSS = $rss_result,\n\tGroup Delay = $a_phi")

end
end

# Store Data Manually
example_name = "2025_6_6_zero"
res1 = hcat(L1...)
res2 = hcat(L2...)
res3 = hcat(L3...)

h5open("./data/$example_name.h5", "w") do file
    write(file, "L1", hcat(L1...))
    write(file, "L2", hcat(L2...))
    write(file, "L3", hcat(L3...))
end