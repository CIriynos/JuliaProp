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

py_ave_list_collection = []
actual_At_list_collection = []

for nc = 1: 0.5: 10
for cep = 0: 0.125pi: 2pi
# nc = 1
# cep = 0.5pi

ap1s = []
ap2s = []
ap3s = []
tau_list = []
hhg_plt_list = []
shg_yields = []
fs_thz_fig_list = []
pmd_fig_list = []
py_ave_list = Float64[]
actual_At_list = Float64[]
pdd_list = []
Et_data_list = []

# Define the function of Movement
F1(E, Z, x, y, z, rj) = -E - Z * rj / (x^2 + y^2 + z^2 + 0.0 + 1e-5) ^ (3 / 2)

# Core Parameters
Z = 1.0
Ip = 0.5

# CTMC Parameters
Δt = 0.2
filter_threshold = 2.0
p_min = -1.0
p_max = 1.0
p_delta = 0.005

# Define the Laser. 
E_fs = 0.0533 * 1
E_thz = 0.00002
E_dc = 0.0
# nc = 2
nc_thz = 2
ω_fs = 0.05693
ω_thz = ω_fs / 4
tau_fs = 500
# cep = 0.0pi

Tp_thz = 2pi * nc_thz / ω_thz
ω_thz_harmonic_list = [k * ω_thz for k = 2: 15]
nc_thz_harmonic_list = (Tp_thz / 2pi) .* ω_thz_harmonic_list
# phase_thz_harmonic_list = rand(length(ω_thz_harmonic_list)) .* 2pi
phase_thz_harmonic_list = ones(length(ω_thz_harmonic_list)) .* 0.5pi

# create runtime for CTMC
trajs_num = Int64((2 * nc * pi / ω_fs) ÷ Δt) + 1
tmax = last(get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz)) + 2 * nc_thz * pi / ω_thz
t_num = Int64(tmax ÷ Δt) + 1
rt = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)

tasks_num = 100
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz, samples_num=tasks_num, nc_thz=nc_thz)
for task_id = 1: tasks_num

# t-data for Field
tau_thz = tau_list[task_id]

Ex_fs, Ey_fs, Ez_fs = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.5, phase1=0.5pi + cep, phase2=0 + cep)
mid_point = tau_fs + nc*pi/ω_fs - pi/ω_thz
E_thz_1, E_thz_2, E_thz_3, tmax_thz = light_pulse(ω_thz, E_thz, nc_thz, tau_thz, pulse_shape="sin2", phase1=0.5pi)
E_thz_k1, _, _, _ = light_pulse(ω_thz_harmonic_list[1], E_thz / 1, nc_thz_harmonic_list[1], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[1])
E_thz_k2, _, _, _ = light_pulse(ω_thz_harmonic_list[2], E_thz / 1, nc_thz_harmonic_list[2], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[2])
E_thz_k3, _, _, _ = light_pulse(ω_thz_harmonic_list[3], E_thz / 1, nc_thz_harmonic_list[3], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[3])
E_thz_k4, _, _, _ = light_pulse(ω_thz_harmonic_list[4], E_thz / 1, nc_thz_harmonic_list[4], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[4])
E_thz_k5, _, _, _ = light_pulse(ω_thz_harmonic_list[5], E_thz / 1, nc_thz_harmonic_list[5], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[5])
E_thz_k6, _, _, _ = light_pulse(ω_thz_harmonic_list[6], E_thz / 1, nc_thz_harmonic_list[6], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[6])
E_thz_k7, _, _, _ = light_pulse(ω_thz_harmonic_list[7], E_thz / 1, nc_thz_harmonic_list[7], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[7])
E_thz_k8, _, _, _ = light_pulse(ω_thz_harmonic_list[8], E_thz / 1, nc_thz_harmonic_list[8], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[8])
E_thz_k9, _, _, _ = light_pulse(ω_thz_harmonic_list[9], E_thz / 1, nc_thz_harmonic_list[9], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[9])
E_thz_k10, _, _, _ = light_pulse(ω_thz_harmonic_list[10], E_thz / 1, nc_thz_harmonic_list[10], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[10])
E_thz_k11, _, _, _ = light_pulse(ω_thz_harmonic_list[11], E_thz / 1, nc_thz_harmonic_list[11], tau_thz, pulse_shape="sin2", phase1=phase_thz_harmonic_list[11])

# E_applied(t) = (E_thz_1(t)) #* E_window(t)
E_applied(t) = (E_thz_1(t) + E_thz_k1(t) + E_thz_k2(t) + E_thz_k3(t) +
    E_thz_k4(t) + E_thz_k5(t) + E_thz_k6(t) + E_thz_k7(t) + E_thz_k8(t) +
    E_thz_k9(t) + E_thz_k10(t) + E_thz_k11(t)) #* E_window(t)

At_data_xyz, Et_data, ts, = create_tdata(tmax, 0, Δt, Ex_fs, t -> Ey_fs(t) + E_applied(t), no_light)
At_data_xyz_hf, Et_data_hf, ts_hf, = create_tdata(tmax, Δt/2, Δt, Ex_fs, t -> Ey_fs(t) + E_applied(t), no_light)
At_data_xyz_no_thz, Et_data_no_thz, = create_tdata(tmax, 0, Δt, Ex_fs, t -> Ey_fs(t), no_light)
At_data_xyz_hf_no_thz, Et_data_hf_no_thz, = create_tdata(tmax, Δt/2, Δt, Ex_fs, t -> Ey_fs(t), no_light)
fs_thz_fig = plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts, thz_ratio=40)

plot(sqrt.(Et_data[1] .^ 2 + Et_data[2] .^ 2 + Et_data[3] .^ 2))

for m = 1: 1
    pv_max = 0.5
    tid_cc, pv_cc, theta_cc = generate_start_point_uniform_special(trajs_num, m, shift=Int64(tau_fs ÷ Δt + 1))
    # tid_cc, pv_cc, theta_cc = generate_start_point_random(trajs_num, t_num, pv_max, Et_data, Ip, Z)
    update_start_point(rt, trajs_num, tid_cc, pv_cc, theta_cc, Et_data, Ip, Z, Δt)

    # Mainloop for CTMC
    traj_filter_flag = ctmc_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, filter_threshold)
    # println((length(traj_filter_flag), sum(traj_filter_flag)))

    # Add data to PMD/HHG
    add_to_pmd(rt, trajs_num, t_num, Z, traj_filter_flag)
    add_to_hhg(rt, trajs_num, t_num, Δt, tid_cc, traj_filter_flag)
end

# HHG
p1, p2, hhg_spectrum_x, hhg_spectrum_y, base_id, hhg_k_linspace = ctmc_get_hhg_spectrum(rt, Et_data, tmax, t_num, ts, Δt, ω_fs, tau_fs)

# PMD
p_axis = p_min: p_delta: p_max
p3 = heatmap(p_axis, p_axis, rt.pxy_final, color=:jet1, size=(500, 420))
px_ave, py_ave, pz_ave = get_average_p_ctmc(rt)


# Record
push!(hhg_plt_list, p2)
push!(shg_yields, norm.(hhg_spectrum_x[base_id * 2 + 1]))
push!(fs_thz_fig_list, fs_thz_fig)
push!(Et_data_list, Et_data)
push!(pmd_fig_list, p3)
push!(py_ave_list, py_ave)

At_thz_xyz, Et_thz_data, = create_tdata(tmax, 0, Δt, no_light, t -> E_applied(t), no_light)
push!(actual_At_list, At_thz_xyz[2][Int64((tau_fs + nc*pi/ω_fs) ÷ Δt) + 1])

# Clear all (Important!!!)
clear_ctmc_rt(rt)

end

push!(py_ave_list_collection, py_ave_list)
push!(actual_At_list_collection, actual_At_list)
# py_ave_list = py_ave_list_collection[k]
# actual_At_list = actual_At_list_collection[k]
println("Nc = $nc, CEP = $cep")

# scatter(tau_list, [(py_ave_list .- py_ave_list[1]) -(actual_At_list)])

# hhg_delta_k = 2pi / t_num / Δt
# hhg_k_linspace = [hhg_delta_k * (i - 1) for i = 1: t_num]
# d2 = norm.(fft(Et_data_list[1][2] .* hanning(t_num)))
# rg = 1: (Int64((1.5 * ω_fs) ÷ hhg_delta_k))
# plot(hhg_k_linspace[rg] ./ ω_fs, d2[rg], yscale=:log10, ylimits=(1e-4, 1e2))

# using Interpolations
# a1 = ((py_ave_list .- py_ave_list[1]))
# a2 = (actual_At_list)
# plot([a1 -a2])
# interp_cub_1 = cubic_spline_interpolation(range(tau_list[2], tau_list[tasks_num-1], tasks_num-2), a1[2: (tasks_num-1)])
# interp_cub_2 = cubic_spline_interpolation(range(tau_list[2], tau_list[tasks_num-1], tasks_num-2), -a2[2: (tasks_num-1)])

# ts = tau_list[2]: Δt: tau_list[tasks_num - 1]
# interp_res_1 = zeros(Float64, t_num)
# interp_res_2 = zeros(Float64, t_num)
# interp_res_1[1: length(ts)] .= (interp_cub_1(ts))
# interp_res_2[1: length(ts)] .= (interp_cub_2(ts))
# plot(1:t_num, [interp_res_1 interp_res_2])
# plot([interp_cub_1(ts) interp_cub_2(ts)])

# spec1 = (fft(interp_res_1 .* hanning(t_num)))
# spec2 = (fft(interp_res_2 .* hanning(t_num)))
# upper_limit_of_hhg = 12.0 * ω_thz
# rg = 1: (Int64((upper_limit_of_hhg) ÷ hhg_delta_k) + 1)
# display_rg = 1: (Int64((13.0 * ω_thz) ÷ hhg_delta_k) + 1)
# norm_cp_data = 20 .* log10.(norm.(spec1) ./ norm.(spec2))
# angle_cp_data = unwrap(angle.(spec1) .- angle.(spec2))

# plot(hhg_k_linspace[rg] ./ (ω_fs / 375), [(d2)[rg] .* 1 norm.(spec1)[rg] norm.(spec2)[rg]], yscale=:log10, ylimits=(1e-5, 1e3))
# plot(hhg_k_linspace[rg] ./ (ω_fs / 375), [norm_cp_data[rg]], ylimits=(-10, 10))
# plot(hhg_k_linspace[rg] ./ (ω_fs / 375), [angle_cp_data[rg]], ylimits=(-5pi, pi))

# interp_cub_f = cubic_spline_interpolation(0: hhg_delta_k: upper_limit_of_hhg, norm_cp_data[rg])
# xs = 0: (hhg_delta_k / 10): (upper_limit_of_hhg - hhg_delta_k)
# ys = interp_cub_f(xs)
# cutoff_id_list = []
# critical_cut_off_id = last(rg)
# for i = 2: length(xs)
#     if (ys[i] <= -3.0 && ys[i - 1] >= -3.0) || (ys[i] >= -3.0 && ys[i - 1] <= -3.0)
#         push!(cutoff_id_list, i)
#     end
#     if ys[i] <= -8.0
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
example_name = "2025_5_25_ctmc_pc"
res1 = hcat(py_ave_list_collection...)
res2 = hcat(actual_At_list_collection...)

h5open("./data/$example_name.h5", "w") do file
    write(file, "py_ave_list_collection", hcat(py_ave_list_collection...))
    write(file, "actual_At_list_collection", hcat(actual_At_list_collection...))
end