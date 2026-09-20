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
hhg_plt_list = []
shg_yields = []
fs_thz_fig_list = []


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

# Define the Laser. 
E_fs = 0.0533
E_thz = 0.0001
E_dc = 0.0001
ω_fs = 0.05693
ω_thz = ω_fs / 5
nc = 15
tau_fs = 0

# create runtime for CTMC
t_num = Int64((tau_fs + 2 * nc * pi / ω_fs) ÷ Δt) + 1
trajs_num = t_num
rt = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)

task_id = 1
# for task_id = 1: 16

# t-data for Field
tau_list = get_1c_thz_delay_list(ω_fs, tau_fs, nc, ω_thz)
tau_thz = tau_list[task_id]

Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.5, phase1=0.5pi, phase2=0)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="sin2", phase1=0.5pi)
Ex_window = dc_bias(1.0, 0, max(0, minimum(tau_list)), tmax, tmax * 2)
E_applied(t) = (Ex_thz(t) + E_dc) #* Ex_window(t)

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
# ap1, ap2, ap3, shg_yield_data = trajs_analyse(rt, Et_data, trajs_num, tmax, t_num, ts, Δt, ω_fs, tau_fs, tid_cc, traj_filter_flag, nc)
# push!(ap1s, ap1)
# push!(ap2s, ap2)
# push!(ap3s, ap3)

# HHG
p1, p2, hhg_spectrum_x, hhg_spectrum_y, base_id, hhg_k_linspace = ctmc_get_hhg_spectrum(rt, Et_data, tmax, t_num, ts, Δt, ω_fs, tau_fs)

# Record
push!(hhg_plt_list, p2)
push!(shg_yields, norm.(hhg_spectrum_x[base_id * 2 + 1]))
push!(fs_thz_fig_list, fs_thz_fig)

# Clear all
clear_ctmc_rt(rt)

# end

hhg_plt_list[1]
plot(tau_list, shg_yields)

Ex_thz_tmp, Ey_thz_tmp, Ez_thz_tmp, tmax_tmp = light_pulse(ω_thz, E_thz, 1, 0, pulse_shape="sin2", phase1=0.5pi)
At_datas_tmp, Et_datas_tmp, ts_tmp = create_tdata(tmax_tmp, 0, Δt, Ex_thz_tmp, no_light, no_light)
thz_data = [Et_datas_tmp[1], ts_tmp]

unify(data) = (data .- minimum(data)) ./ (maximum(data) - minimum(data))
p = plot(unify(tau_list), unify(shg_yields))
plot!(p, unify(thz_data[2]), unify(-thz_data[1]))


# f = open("phase_fig_3.txt", "w+")
# for i = 1: length(shg_yield_data)
#     write(f, "$(angle.(shg_yield_data[i])) $(norm.(shg_yield_data[i]))\n")
# end
# close(f)


# Research on Ionz. Rate 

# tau_list = [-400, 100, 300, 500, 1000]
# -500, 100, 300, 500, 1000
p = plot()
p2 = plot()
rcd_shg = []
wt_datas = []
et_figs = []

tau_id = 1
# for tau_id in 1: 5

ω = ω_fs
ω_thz = ω / 5
E0 = E_fs
gamma = ω / E0
# nc = 15
Δt = 0.01 / 2
# E0_thz = 0.00001 * 10
# E_c = 0.00001 * 10
E0_thz = 0.00001 * 10
E_c = 0.00001 * 10

tau_lst = [-2pi/ω_thz, nc*pi/ω - 1.5pi/ω_thz,
    nc*pi/ω - pi/ω_thz, nc*pi/ω - 0.5pi/ω_thz, nc*2pi/ω]
tau = tau_lst[tau_id]
println("tau = $tau")

Et_thz(t) = (E0_thz) * sin(ω_thz*(t-tau)) * (t-tau > 0 && t-tau <(2*pi/ω_thz))
Et_fs(t) = E0 * sin(ω*t/2/nc)^2 * cos(ω*t + pi) * (t < 2*nc*pi/ω)
Et(t) = Et_fs(t) + (Et_thz(t) + E_c) * flap_top_windows_f(t, 0, 2*nc*pi/ω, 1/8)
t_linspace = 0: Δt: 2*nc*pi/ω
e_fig = plot(t_linspace, [Et_thz.(t_linspace) * 200 Et_fs.(t_linspace)])
push!(et_figs, e_fig)

Wk(gamma, Ip, omega) = exp(-(2*Ip/omega) * ((1 + 0.5/gamma^2) * asinh(gamma) - sqrt(1 + gamma^2) / 2 / gamma))
Wk2(F, Ip, omega) = Wk(omega * sqrt(2*Ip) / F, Ip, omega)
ADK_f(F) = 4 / F * exp(-2 / (3 * F))
# W(t) = ADK_f(abs(E(t)) + 1e-10)
W(t) = Wk2(abs(Et(t)) + 1e-10, 0.5, ω)
Wt_data = W.(t_linspace)
Wt_int_data = get_integral(Wt_data, Δt)
Wt_data_saturn = @. Wt_data * exp(-Wt_int_data)
push!(wt_datas, Wt_data_saturn)

# for FFT
N = length(t_linspace)
Δω = 2pi / N / Δt
ω_linspace = [(i==1) ? (Δω) : (Δω * (i - 1)) for i = 1: N]
hhg_window_f(t, tmax) = sin(t / tmax * pi) ^ 2
shg_place = Int64(floor(ω * 2 / Δω))
println("shg_place = $shg_place")


# # Method 1 (My Analyse)
# H_data = fft(W.(t_linspace) .* hhg_window_f.(t_linspace, last(t_linspace)))
# G_data = fft(E.(t_linspace) .* hhg_window_f.(t_linspace, last(t_linspace)))
# plot(ω_linspace[1:100], norm.(H_data)[1:100])
# plot(ω_linspace[1:100], norm.(G_data)[1:100])

# H_super = H_data ./ (im * ω_linspace)
# res = conv(H_super, G_data)[1:N] .+ sqrt(2pi) .* Ec .* H_super
# plot!(p, ω_linspace[1:200], norm.(res)[1:200], yscale=:log10)


# method 2 (Directly)
Eyield = Et.(t_linspace) .* get_integral(Wt_data_saturn, Δt)
Gyield = fft(Eyield .* hhg_window_f.(t_linspace, last(t_linspace)))
plot!(p2, ω_linspace[1:200], norm.(Gyield)[1:200], yscale=:log10)
#plot(ω_linspace[1:100], [norm.(res)[1:100] * 1e-5 norm.(Gyield)[1:100]], yscale=:log10)


pp = plot(hhg_k_linspace[1:250] / ω_fs, normalize(abs.(hhg_spectrum_x[1:250]) .^ 2), yscale=:log10, label="CTMC", xlabel="harmonic order", ylabel="|G(ω)|^2")
plot!(pp, ω_linspace[1:250] / ω_fs, normalize(norm.(Gyield)[1:250] .^ 2), yscale=:log10, label="PC")