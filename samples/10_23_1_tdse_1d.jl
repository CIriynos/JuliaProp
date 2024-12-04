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
# 1. 测试1D-TDSE，研究E增大是否会出现相干探测
# 2. 研究格点对结果的影响
# 测试结果: ratio=1,2,3时，1与2差异明显，2与3几乎一致
# 结论：最好采用ratio=2的格点，此时几乎收敛（即使电场强度×2也收敛）
# 3. 研究THz ω的影响

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
get_energy_1d(init_wave, rt1d)


# Mainloop for Sweeping
at_data_figs = []
et_data_figs = []
part_of_hhg_data = []
shg_yield_record = []
smooth_records = []
# tau_list = [-500, 100, 285, 470, 1000]
# tau_list = [-1200, -650, -270, 100, 800]
tau_list = [0]

for tau_var in tau_list

induce_time = 550
thz_rate = 1
fs_rate = 0
tau = tau_var
# tau_fs = induce_time + (tau < 0.0) * abs(tau)
# tau_thz = induce_time + (tau >= 0.0) * abs(tau)

# Define Laser.
tau_fs = induce_time + 500  # fixed
tau_thz = induce_time + tau + 500
omega = 0.057           # 800 nm (375 THz)
omega_thz = 0.05omega    # 16 μm (18.75 THz)
E0 = 0.057 * fs_rate
E0_thz = 0.0075 * thz_rate
E0_c = 0.005 * thz_rate * 0
nc = 15
Tp = 2 * nc * pi / omega
Et(t) = E0 * sin(omega * (t - tau_fs) / (2 * nc)) ^ 2 * cos(omega * (t - tau_fs)) * (t - tau_fs < Tp && t - tau_fs > 0)
E_thz(t) = E0_thz * sin(omega_thz * (t - tau_thz) / 2) ^ 2 * sin(omega_thz * (t - tau_thz)) * (t - tau_thz < 2pi / omega_thz && t - tau_thz > 0)

T_total = tau_fs + Tp #max(tau_fs + Tp, tau_thz + 2pi/omega_thz)
steps = Int64(T_total ÷ delta_t)
t_linspace = create_linspace(steps, delta_t)

Et_data_fs = Et.(t_linspace)
Et_data_thz = E_thz.(t_linspace)
Et_data_c = E0_c * flap_top_windows_f.(t_linspace, 0, 1.5 * induce_time, 1/2, right_flag = false)
# Et_data_c = E0_c * flap_top_windows_f.(t_linspace, tau_thz - induce_time, tau_thz + 2pi/omega_thz + induce_time, 1/4)

# modify here.
# mid_timing = tau_fs + Tp / 2
# mid_thz = tau_thz + pi / omega_thz
# mid_Et = E_thz(mid_timing) + E0_c
# println("mid_Et = $(mid_Et), tau = $(tau)")
# mid_len = 300
# Et_data = Et.(t_linspace) .+ mid_Et * flap_top_windows_f.(t_linspace, mid_timing - mid_len, mid_timing + mid_len, 1/2)

Et_data = Et_data_fs + Et_data_thz + Et_data_c
At_data = -get_integral(Et_data, delta_t)

# store the figures.
fig1 = plot(At_data)
fig2 = plot([Et_data_fs (Et_data_thz + Et_data_c) * 10])
push!(at_data_figs, fig1)
push!(et_data_figs, fig2)

# TDSE 1d
crt_wave = deepcopy(init_wave)
Xi_data, hhg_integral, energy_list, smooth_record = tdse_laser_fd1d_mainloop_penta(crt_wave, rt1d, pw1d, At_data, steps, Xi)
push!(smooth_records, smooth_record)

# # t-surf
# k_delta = 0.002
# kmin = -3.0
# kmax = 3.0
# k_linspace = kmin: k_delta: kmax
# Pk = tsurf_1d(pw1d, k_linspace, t_linspace, At_data, Xi, Xi_data)
# plot(Pk, yscale=:log10)


# HHG
hhg_start_id = 1 #Int64(tau_fs ÷ delta_t)
hhg_end_id = Int64((tau_fs + Tp) ÷ delta_t)
hhg_len = hhg_end_id - hhg_start_id + 1
hhg_delta_k = 2pi / hhg_len / delta_t
hhg_k_linspace = [hhg_delta_k * i for i = 1: hhg_len]

hhg_t = (hhg_integral - Et_data)[hhg_start_id: hhg_end_id]
hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(t_linspace[hhg_start_id: hhg_end_id], tau_fs, tau_fs + Tp)
hhg_spectrum = fft(hhg_t .* hhg_windows_data)

max_hhg_id = Int64(floor(5 * omega / hhg_delta_k))
shg_id = Int64(floor(2 * omega / hhg_delta_k))
# plot(hhg_k_linspace[1: max_hhg_id] ./ omega, norm.(hhg_spectrum)[1: max_hhg_id], yscale=:log10)

push!(part_of_hhg_data, norm.(hhg_spectrum)[1: max_hhg_id])
push!(shg_yield_record, norm.(hhg_spectrum)[shg_id])
println("shg_id = ", shg_id)
println("hhg_len = ", hhg_len)
println("hhg_delta_k = ", hhg_delta_k)

end

second_hhg = [part_of_hhg_data[i][50] for i = 1: length(part_of_hhg_data)]
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

plot(part_of_hhg_data, yscale=:log10)