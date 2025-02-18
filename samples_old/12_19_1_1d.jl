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

# 2024-10-23-1


# Define Basic Parameters
ratio = 1
Nx = 5000 * ratio
delta_x = 0.2 / ratio
delta_t = 0.05 / ratio
delta_t_itp = 0.1
Lx = Nx * delta_x
Xi = Lx / 2 * 0.8
po_func(x) = -0.8 * (x^2 + 0.1) ^ (-0.5) * flap_top_windows_f(x, -Xi, Xi, 1/4)
imb_func(x) = -100im * ((abs(x) - Xi) / (Lx / 2 - Xi)) ^ 8 * (abs(x) > Xi)


# Create Physics World & Runtime
pw1d = create_physics_world_1d(Nx, delta_x, delta_t, po_func, imb_func, delta_t_im=delta_t_itp)
rt1d = create_tdse_rt_1d(pw1d)


# Get Initial Wave
x_linspace = get_linspace(pw1d.xgrid)
seed_wave = gauss_package_1d(x_linspace, 1.0, 1.0, 0.0)
init_wave = itp_fd1d(seed_wave, rt1d, min_error = 1e-10)
get_energy_1d(init_wave, rt1d)


# Mainloop for Sweeping
at_data_figs = []
et_data_figs = []
part_of_hhg_data = []
shg_yield_record = []
smooth_records = []

tau_id = 1

# Define Laser.
ω1 = 0.057        # 800 nm (375 THz)
ω2 = 0.05 * ω1    # 16 μm (18.75 THz)
E0 = 0.057 / 2
E0_thz = 0.00002
E0_c = 0.00002 * 5
nc = 15
Tp = 2 * nc * pi / ω1


# Define the tau (Delay)
induce_time = 0
tau_fs = induce_time + 500
tau_lst = [tau_fs - 2pi/ω2, tau_fs + nc*pi/ω1 - 1.5pi/ω2,
    tau_fs + nc*pi/ω1 - pi/ω2, tau_fs + nc*pi/ω1 - 0.5pi/ω2, tau_fs + nc*2pi/ω1]
tau_thz = induce_time + tau_lst[tau_id]


# Define the waveform
Et(t) = E0 * sin(ω1 * (t - tau_fs) / (2 * nc)) ^ 2 * cos(ω1 * (t - tau_fs)) * (t - tau_fs < Tp && t - tau_fs > 0)
E_thz(t) = E0_thz * sin(ω2 * (t - tau_thz)) * (t - tau_thz < 2pi / ω2 && t - tau_thz > 0)
E_thz_window(t) = flap_top_windows_f(t, 0, tau_fs * 2, 1/2, right_flag = false)


T_total = tau_fs + Tp
steps = Int64(T_total ÷ delta_t)
t_linspace = create_linspace(steps, delta_t)


Et_data_fs = Et.(t_linspace)
Et_data_thz = (E0_c .+ E_thz.(t_linspace)) .* E_thz_window.(t_linspace)
Et_data = Et_data_fs + Et_data_thz
At_data = -get_integral(Et_data, delta_t)


# store the figures.
fig1 = plot(At_data)
fig2 = plot([Et_data_fs (Et_data_thz) * 500])
push!(at_data_figs, fig1)
push!(et_data_figs, fig2)


# TDSE 1d
crt_wave = deepcopy(init_wave)
Xi_data, hhg_integral, energy_list = tdse_laser_fd1d_mainloop_penta(crt_wave, rt1d, pw1d, At_data, steps, Xi)


# # t-surf
# k_delta = 0.002
# kmin = -3.0
# kmax = 3.0
# k_linspace = kmin: k_delta: kmax
# Pk = tsurf_1d(pw1d, k_linspace, t_linspace, At_data, Xi, Xi_data)
# plot(Pk, yscale=:log10)


# HHG
hhg_delta_k = 2pi / steps / delta_t
hhg_k_linspace = [hhg_delta_k * i for i = 1: steps]
max_hhg_id = Int64(floor(10 * ω1 / hhg_delta_k))
shg_id = Int64(ceil(2 * ω1 / hhg_delta_k)) + 1
base_id = Int64(ceil(ω1 / hhg_delta_k)) + 1

hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(t_linspace, tau_fs + 0, tau_fs + Tp - 0)

hhg_t = -hhg_integral - Et_data
hhg_spectrum = fft(hhg_t .* hhg_windows_data)

# plot(hhg_k_linspace[1: max_hhg_id] ./ ω1, norm.(hhg_spectrum)[1: max_hhg_id], yscale=:log10)

push!(part_of_hhg_data, norm.(hhg_spectrum)[1: max_hhg_id])
push!(shg_yield_record, norm.(hhg_spectrum)[shg_id])

println("shg_id = ", shg_id)
println("hhg_delta_k = ", hhg_delta_k)
println("tau_fs = ", tau_fs)

# end

# second_hhg = [part_of_hhg_data[i][50] for i = 1: length(part_of_hhg_data)]
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

p = plot(norm.(hhg_spectrum)[1: max_hhg_id], yscale=:log10, ylimit=(1e-10, 1e3))
plot!(p, [base_id, base_id], [1e-5, 1e2])