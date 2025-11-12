import Pkg
Pkg.activate(".")

using Revise
using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using Printf
using FFTW
println("Number of Threads: $(Threads.nthreads())")

# Define Basic Parameters
ratio = 1
Nx = 500 * ratio
delta_x = 0.02 / ratio
delta_t = 0.01 / ratio
delta_t_itp = 0.01 / ratio
Lx = Nx * delta_x
Xi = 0.5 * Lx * 0.8
a0 = 1.0
po_func(x) = -1.0 * (x^2 + a0) ^ (-0.5) #* flap_top_windows_f(x, -Xi, Xi, 1/4) * exp(-0.1 * (x^2 + a0) ^ (0.5))
omega_s = 1.0
# po_func(x) = 0.5 * omega_s^2 * x^2
# imb_func(x) = -100im * ((abs(x) - Xi) / (Lx / 2 - Xi)) ^ 8 * (abs(x) > Xi) * 0

# Create Physics World & Runtime
pw1d = create_physics_world_1d(Nx, delta_x, delta_t, po_func, imb_func, delta_t_im=delta_t_itp)
rt1d = create_tdse_rt_1d(pw1d)

# Get Initial Wave
x_linspace = get_linspace(pw1d.xgrid)
seed_wave = gauss_package_1d(x_linspace, 1.0, 1.0, 0.0)
init_wave = itp_fd1d(seed_wave, rt1d, min_error = 1e-80)
en0 = get_energy_1d(init_wave, rt1d)

# Mainloop for Sweeping
at_data_figs = []
et_data_figs = []
part_of_hhg_data = []
shg_yield_record = []
smooth_records = []

tau_id = 1
# for tau_id = 1: 5

# Define Laser.
ω1 = 0.057 * 1           # 800 nm (375 THz)
ω2 = ω1 / 20         # 16 μm (12.5 THz)
E0 = 0.057 * 0.1
E0_thz = 0.0002 * 0
E0_c = 0.0002 * 0
nc = 8
Tp = 2 * nc * pi / ω1

# Define the tau (Delay)
induce_time = 100
tau_fs = induce_time
tau_lst = [tau_fs - 2pi/ω2, tau_fs + nc*pi/ω1 - 1.5pi/ω2,
    tau_fs + nc*pi/ω1 - pi/ω2, tau_fs + nc*pi/ω1 - 0.5pi/ω2, tau_fs + nc*2pi/ω1]
tau_thz = tau_lst[tau_id]

# Define the waveform
Et(t) = E0 * sin(ω1 * (t - tau_fs) / (2 * nc)) ^ 2 * cos(ω1 * (t - tau_fs)) * (t - tau_fs < Tp && t - tau_fs > 0)
E_thz(t) = E0_thz * sin(ω2 * (t - tau_thz)) * (t - tau_thz < 2pi / ω2 && t - tau_thz > 0)
E_thz_window(t) = flap_top_windows_f(t, 0, induce_time * 2, 1/2, right_flag = false)

appendix_time = tau_fs
T_total = tau_fs + Tp + appendix_time
steps = Int64(T_total ÷ delta_t) + 1
t_linspace = create_linspace(steps, delta_t)

Et_data_fs = Et.(t_linspace)
Et_data_thz = (E0_c .+ E_thz.(t_linspace)) .* flap_top_windows_f.(t_linspace, tau_fs, T_total, 1/2)

Et_data = Et_data_fs + Et_data_thz
At_data = -get_integral(Et_data, delta_t)

# store the figures.
fig1 = plot(At_data)
fig2 = plot([Et_data_fs (Et_data_thz) * 1])
push!(at_data_figs, fig1)
push!(et_data_figs, fig2)

# TDSE 1d
crt_wave = deepcopy(init_wave)
Xi_data, hhg_integral, energy_list, uncer_list = tdse_laser_fd1d_mainloop_penta(crt_wave, rt1d, pw1d, At_data, steps, Xi, record_steps=1000)

# Energy spectrum
Ev_list = 0.0: 0.01: 20.0
Plist_total, Plist = windows_operator_method_1d(crt_wave, 0.005, 3, Ev_list, rt1d, pw1d)
plot(Plist, yscale=:log10)

# dcrt_wave = get_derivative_two_order(crt_wave, delta_x)
# ddcrt_wave = get_derivative_two_order(dcrt_wave, delta_x)
# u1 = dot(crt_wave, x_linspace .^ 2 .* crt_wave) - dot(crt_wave, x_linspace .* crt_wave) ^ 2
# u2 = dot(crt_wave, -ddcrt_wave) - dot(crt_wave, -im * dcrt_wave) ^ 2
# u1 * u2

# # t-surf
# k_delta = 0.002
# kmin = -3.0
# kmax = 3.0
# k_linspace = kmin: k_delta: kmax
# Pk = tsurf_1d(pw1d, k_linspace, t_linspace, At_data, Xi, Xi_data)
# plot(Pk, yscale=:log10)


# HHG
hhg_delta_k = 2pi / steps / delta_t
hhg_k_linspace = [hhg_delta_k * (i - 1) for i = 1: steps]

hhg_t = -hhg_integral #- Et_data
hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(t_linspace, 0 + 0, T_total - 0)
hhg_spectrum = fft((hhg_t .* hhg_windows_data))

max_hhg_id = Int64(floor(30 * ω1 / hhg_delta_k))
shg_id = Int64(floor(2 * ω1 / hhg_delta_k))
# plot(hhg_k_linspace[1: max_hhg_id] ./ ω1, norm.(hhg_spectrum)[1: max_hhg_id], yscale=:log10)

push!(part_of_hhg_data, norm.(hhg_spectrum)[1: max_hhg_id])
push!(shg_yield_record, norm.(hhg_spectrum)[shg_id])

println("shg_id = ", shg_id)
println("hhg_delta_k = ", hhg_delta_k)
println("tau_fs = ", tau_fs)

# end

# second_hhg = [part_of_hhg_data[i][24] for i = 1: length(part_of_hhg_data)]
# plot(second_hhg .^ 0.5,
#     guidefont=Plots.font(14, "Times"),
#     tickfont=Plots.font(14, "Times"),
#     legendfont=Plots.font(14, "Times"),
#     margin = 5 * Plots.mm,
#     xlabel="Time delay τ(a.u.)",
#     ylabel = "2nd Harmonic Yield(a.u.)",
#     label = "THz",
#     linewidth = 2.0
# )

plot(norm.(hhg_spectrum)[1: 1000], yscale=:log10)

pp = plot(hhg_k_linspace[1:max_hhg_id], normalize(norm.(hhg_spectrum[1:max_hhg_id])), yscale=:log10, ylimits=(1e-15, 1e3))
plot!(pp, [1, 1], [1e-15, 1e3], label="1")

alpha_data = get_integral(Et_data .* exp.(im * omega_s * 1 * t_linspace), delta_t) .* 1
res = conj.(alpha_data) .* exp.(im * omega_s * 1 * t_linspace) .+ alpha_data .* exp.(-im * omega_s * 1 * t_linspace)
plot(real.(res))
plot!(pp, hhg_k_linspace[1:max_hhg_id], (sqrt(2)) * normalize(norm.(fft((res .* hhg_windows_data))))[1:max_hhg_id], yscale=:log10, ylimits=(1e-15, 1e3))

# R = abs(en0)
# plot!(pp, [R, R], [1e-15, 1e3])

# plot(norm.(fft(Et_data .* hhg_windows_data))[1:max_hhg_id], yscale=:log10)



# # Time-frequency Analysis
# hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
# t_num = length(hhg_t)
# w_num = 100
# ts = 1: t_num
# plot(norm.(hhg_t))

# ft_ana = zeros(ComplexF64, w_num, t_num)
# delta = t_num / 50
# j = 1
# for i = 1: (delta / 10): t_num - delta
#     ft_ana[:, j] = fft((real(hhg_t) .* hhg_windows_f.(ts, i, i + delta)))[1: w_num]
#     j += 1
# end

# heatmap(log10.(norm.(ft_ana)[1: w_num, (1 + 0): (j - 0)] .^ 2), clim=(-4.5, 0))
# heatmap((angle.(ft_ana)[1: w_num, (1): (j)]))


# f = open("tmptmp4.txt", "w+")
# for i = 1: length(hhg_t)
#     write(f, "$(real(hhg_t[i]))\n")
# end
# close(f)