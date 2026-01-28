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
ratio = 2
Nx = 3600 * ratio
delta_x = 0.2 / ratio
delta_t = 0.05 / ratio
delta_t_itp = 0.1
Lx = Nx * delta_x
Xi = 200
po_func(x) = -(x^2 + 1) ^ (-0.5) * flap_top_windows_f(x, -Xi, Xi, 1/4)
imb_func(x) = -100im * ((abs(x) - Xi) / (Lx / 2 - Xi)) ^ 8 * (abs(x) > Xi)

# Create Physics World & Runtime
pw1d = create_physics_world_1d(Nx, delta_x, delta_t, po_func, imb_func, delta_t_im=delta_t_itp)
rt1d = create_tdse_rt_1d(pw1d)

# Get Initial Wave
x_linspace = get_linspace(pw1d.xgrid)
seed_wave = gauss_package_1d(x_linspace, 1.0, 1.0, 0.0)
init_wave = itp_fd1d(seed_wave, rt1d, min_error = 1e-10)

pp = []
ap = []
xp = []
wv = []
mwv = []
tau_list = -500: 50: 800
for tau_var in tau_list

# Define Laser.
induce_time = 200
tau = tau_var
# tau_fs = induce_time + (tau < 0.0) * abs(tau)
# tau_thz = induce_time + (tau >= 0.0) * abs(tau)
tau_fs = induce_time + 500
tau_thz = induce_time + tau + 500
omega = 0.057   # 800 nm
omega_thz = 0.2omega
E0 = 0.057
rate = 1 / 5
E0_thz = 0.0075 * rate
E0_c = 0.005 * rate
nc = 6
Tp = 2 * nc * pi / omega
Et(t) = E0 * sin(omega * (t - tau_fs) / (2 * nc)) ^ 2 * cos(omega * (t - tau_fs)) * (t - tau_fs < Tp && t - tau_fs > 0)
E_thz(t) = E0_thz * sin(omega_thz * (t - tau_thz) / 2) ^ 2 * sin(omega_thz * (t - tau_thz)) * (t - tau_thz < 2pi / omega_thz && t - tau_thz > 0)

T_total = tau_fs + Tp #max(tau_fs + Tp, tau_thz + 2pi/omega_thz)
steps = Int64(T_total ÷ delta_t)
t_linspace = create_linspace(steps, delta_t)

# Et_data_fs = Et.(t_linspace)
# Et_data_thz = E_thz.(t_linspace)
# # Et_data_c = E0_c * flap_top_windows_f.(t_linspace, 0, 1.5 * induce_time, 1/2, right_flag = false)
# # Et_data_c = E0_c * flap_top_windows_f.(t_linspace, tau_thz - induce_time, tau_thz + 2pi/omega_thz + induce_time, 1/4)
# Et_data = Et_data_fs + Et_data_thz + Et_data_c

mid_timing = tau_fs + Tp / 2
mid_thz = tau_thz + pi / omega_thz
mid_Et = E_thz(mid_timing) + E0_c
println("mid_Et = $(mid_Et), tau = $(tau)")
Et_data = Et.(t_linspace) .+ mid_Et * flap_top_windows_f.(t_linspace, mid_timing - 100, mid_timing + 100, 1/2)
At_data = -get_integral(Et_data, delta_t)

xx = plot(At_data)
# dd = plot([Et_data_fs (Et_data_thz + Et_data_c) * 50])
dd = plot(Et_data)
push!(ap, dd)
push!(xp, xx)

# TDSE 1d
crt_wave = deepcopy(init_wave)
Xi_data, hhg_integral, energy_list = tdse_laser_fd1d_mainloop_penta(crt_wave, rt1d, pw1d, At_data, steps, Xi)

push!(wv, crt_wave)

rg = Nx ÷ 2 - 200 * ratio: Nx ÷ 2 + 200 * ratio + 2
# crt_wave_L = gauge_transform_V2L(crt_wave, At_data, delta_t, x_linspace)
# mid_wv_fig = plot([real.(crt_wave)[rg] real.(crt_wave_L)[rg]])
push!(mwv, (crt_wave)[rg])

# # t-surf
# k_delta = 0.002
# kmin = -3.0
# kmax = 3.0
# k_linspace = kmin: k_delta: kmax
# Pk = tsurf_1d(pw1d, k_linspace, t_linspace, At_data, Xi, Xi_data)
# plot(Pk, yscale=:log10)


# HHG
hhg_start_id = Int64(tau_fs ÷ delta_t)
hhg_end_id = Int64((tau_fs + Tp) ÷ delta_t)
hhg_len = Int64(Tp ÷ delta_t)
hhg_delta_k = 2pi / hhg_len / delta_t
hhg_k_linspace = [hhg_delta_k * i for i = 1: hhg_len]

hhg_t = (hhg_integral - Et_data)[hhg_start_id: hhg_end_id]
hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(t_linspace[hhg_start_id: hhg_end_id], tau_fs, tau_fs + Tp)
hhg_spectrum = fft(hhg_t .* hhg_windows_data)

max_hhg_id = Int64(floor(40 * omega / hhg_delta_k))
plot(hhg_k_linspace[1: max_hhg_id] ./ omega, norm.(hhg_spectrum)[1: max_hhg_id], yscale=:log10)
push!(pp, norm.(hhg_spectrum)[1: max_hhg_id])

end

second_hhg = [pp[i][13] for i = 1: length(pp)]
plot(tau_list, second_hhg .^ 0.5,
    guidefont=Plots.font(14, "Times"),
    tickfont=Plots.font(14, "Times"),
    legendfont=Plots.font(14, "Times"),
    margin = 5 * Plots.mm,
    xlabel="Time delay τ(a.u.)",
    ylabel = "2nd Harmonic Yield(a.u.)",
    label = "THz",
    linewidth = 2.0
)

plot(pp, yscale=:log10)


# mid_len = 6
# rg = Nx ÷ 2 - 99 - 1: Nx ÷ 2 + 99 + 2
# mask = [!(i in Nx ÷ 2 - mid_len: Nx ÷ 2 + mid_len + 2) for i in 1: Nx]

# tmp = get_derivative_two_order(pw1d.po_data, pw1d.delta_x)
# plot([tmp[rg] tmp2[rg]])

# dot(wv[2], wv[2] .* tmp) * delta_x
# sum(tmp .* norm.(wv[2]) .^ 2) * delta_x
# numerical_integral(tmp2 .* norm.(wv[2]) .^ 2, delta_x)

# plot([norm.(mwv[1]) norm.(mwv[2])], yscale=:log10)


# plot(energy_list)
# l2 = energy_list
# plot([l1[216 - 100: 216] l2[116 - 100: 116]])
# plot([l1[1:50] l2[1:50]])


# # Energy
# gamma = 0.001
# Ev_list = -0.7: 2gamma: 1.0
# _, Plist = windows_operator_method_1d(crt_wave_L, gamma, 3, Ev_list, rt1d, pw1d)
# ave_energy = sum(Ev_list .* Plist) / sum(Plist)
# plot(Ev_list, normalize(Plist), yscale=:log10, legend=false)