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
E_thz = 0.00002 * 10
E_dc = 0.00002 * 0
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
for task_id = [1, 13]

# t-data for Field
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz, samples_num=32 + 2)
tau_thz = tau_list[task_id]

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
update_start_point(rt, trajs_num, tid_cc, pv_cc, theta_cc, Et_data, Ip, Z, Δt)

# Mainloop for CTMC
traj_filter_flag = ctmc_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, filter_threshold)
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


# f = open("single_traj_ctmc_first_2.txt", "w+")
# for i = 1: length(traj_ana_list[2][:, 4])
#     write(f, "$(real(traj_ana_list[2][:, 4][i]))\n")
# end
# close(f)

# plot([traj_ana_list[1][:, 5] traj_ana_list[2][:, 5]])




# l4 = norm.(hhg_data_list_ctmc[1][1:100])
# ks = hhg_k_linspace_ctmc[1:100]
# plot(hhg_k_linspace_ctmc[1:100] ./ ω_fs, [l1 l2 l3 l4], yscale=:log10)

# f = open("fig3_b.txt", "w+")
# for i = 1: length(l1)
#     write(f, "$(ks[i]) $(l1[i]) $(l2[i]) $(l3[i]) $(l4[i])\n")
# end
# close(f)


# unify(data) = (data .- minimum(data)) ./ (maximum(data) - minimum(data))

# p = plot(unify(tau_list), unify(shg_yields))
# plot!(p, unify(thz_data[2]), unify(reverse(thz_data[1])))
# p



# f = open("fig3_a_8.txt", "w+")
# for i = 1: length(tau_list)
#     write(f, "$(tau_list[i]) $(shg_yields_scaled[i])\n")
# end
# close(f)

# f = open("fig2_ctmc_point.txt", "w+")
# for i = 1: length(tau_list)
#     write(f, "$(tau_list[i]) $(shg_yields_scaled[i])\n")
# end
# close(f)

f = open("fig2_phase_data_1.txt", "w+")
for i = 1: length(phase_data_list[1])
    write(f, "$(angle(phase_data_list[1][i])) $(norm(phase_data_list[1][i]))\n")
end
close(f)

f = open("fig2_phase_data_2.txt", "w+")
for i = 1: length(phase_data_list[2])
    write(f, "$(angle(phase_data_list[2][i])) $(norm(phase_data_list[2][i]))\n")
end
close(f)

# hhg_plt_list_ctmc[1]


# using SignalAnalysis, SignalAnalysis.Units
# Tp = 2 * nc * pi / ω_fs
# mid_id = Int64((Tp / 2) ÷ Δt + 1)
# rg = mid_id - 500: mid_id + 3000
# hhg_delta_k = 2pi / t_num / Δt
# shg_id = Int64(floor((ω_fs * 2) ÷ hhg_delta_k) + 1)
# x1 = signal(traj_ana_list[1][:, 5] .+ 1e-5, 1kHz)
# x2 = signal(traj_ana_list[1][:, 4] .+ 1e-5, 1kHz)
# y1 = tfd(x1, Spectrogram(nfft=200, noverlap=200 - 1, window=hamming));
# y2 = tfd(x2, Spectrogram(nfft=200, noverlap=200 - 1, window=hamming));

# p5 = plot([traj_ana_list[1][:, 1][rg] traj_ana_list[1][:, 2][rg] traj_ana_list[1][:, 3][rg]])
# plot!(p5, [traj_ana_list[2][:, 1][rg] traj_ana_list[2][:, 2][rg] traj_ana_list[2][:, 3][rg]])
# rg2 = rg .- 100
# ft1 = reverse(y1.power[4, :][rg2]) .* 0.5e4
# ft2 = reverse(y2.power[4, :][rg2]) .* 0.5e4
# plot!(p5, [ft1 ft2])

# ts = [0 + (i - 1) * Δt for i = 1: 8000]
# t1 = 2 * (nc ÷ 2 + 0.25) * π / ω_fs
# t2 = 2 * (nc ÷ 2 + 0.75) * π / ω_fs
# d1 = sin.(2ω_fs .* ts) .* 5e-3 .* (ts .>= t1 .&& ts .<= 1100)
# d2 = -sin.(2ω_fs .* ts) .* 5e-3 .* (ts .>= t2 .&& ts .<= 1100)
# plot!(p5, d1[rg])
# plot!(p5, d2[rg])

# d3 = cos.(3ω_fs .* ts) .* 5e-3 .* (ts .>= t1 .&& ts .<= 1100)
# d4 = -cos.(3ω_fs .* ts) .* 5e-3 .* (ts .>= t2 .&& ts .<= 1100)
# plot!(p5, d3[rg])
# plot!(p5, d4[rg])

# f = open("fig2_c_data.txt", "w+")
# for i = 1: length(rg)
#     write(f, "$(ts[rg][i]) ")
#     write(f, "$(traj_ana_list[1][:, 1][rg][i]) $(traj_ana_list[1][:, 2][rg][i]) $(traj_ana_list[1][:, 3][rg][i]) ")
#     write(f, "$(traj_ana_list[2][:, 1][rg][i]) $(traj_ana_list[2][:, 2][rg][i]) $(traj_ana_list[2][:, 3][rg][i]) ")
#     write(f, "$(ft1[i]) $(ft2[i]) ")
#     write(f, "$(d1[rg][i]) $(d2[rg][i]) $(d3[rg][i]) $(d4[rg][i])\n")
# end
# close(f)

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



#########################################
## Test the small variation of phase-angle for half-cycle ionz. events
#########################################

# T1 = 2 * 5.25 * pi / ω_fs
# T2 = 2 * 5.75 * pi / ω_fs
# mask1 = [i > Int64(T1 ÷ Δt + 1) for i = 1: t_num]
# mask2 = [i > Int64(T2 ÷ Δt + 1) for i = 1: t_num]
# tmp1 = fft(Et_data[1] .* mask1)
# tmp2 = fft(Et_data[1] .* mask2)
# hhg_delta_k = 2pi / t_num / Δt
# shg_id = Int64(floor((ω_fs * 2) ÷ hhg_delta_k) + 1)
# base_id = Int64(floor(ω_fs ÷ hhg_delta_k) + 1)

# plot([Et_data[1] .* mask1 Et_data[1] .* mask2])
# pp = plot([norm.(tmp1)[1:100]])
# scatter!(pp, [(base_id, 0) ,(shg_id, 0)])

# # scatter([angle(tmp1[shg_id]) angle(tmp2[shg_id])], [norm(tmp1[shg_id]) norm(tmp2[shg_id])], proj=:polar)
# abs(rad2deg(angle(tmp1[shg_id]) - angle(tmp2[shg_id]))) - 180

# sss = plot()
# d1 = zeros(Float64, t_num)
# d2 = zeros(Float64, t_num)
# for j = 1: t_num
#     maskj = [i > j for i = 1: t_num]
#     tmp1 = fft(Et_data[1] .* maskj .* rt.weight_cc[j])[shg_id]
#     d1[j] = angle(tmp1) 
#     d2[j] = norm(tmp1)
# end
# scatter(d1, d2, proj=:polar, markerstrokewidth = 0,
# markeralpha = 0.5,
# markersize = 1)

# plot(rt.weight_cc)