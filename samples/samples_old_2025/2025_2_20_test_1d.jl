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
a0 = 1.0
po_func(x) = -1.0 * (x^2 + a0) ^ (-0.5) * flap_top_windows_f(x, -Xi, Xi, 1/4) * exp(-0.1 * (x^2 + a0) ^ (0.5))
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
ω1 = 0.057           # 800 nm (375 THz)
ω2 = ω1 / 20         # 16 μm (12.5 THz)
E0 = 0.057 / 1
E0_thz = 0.00005 * 0
E0_c = 0.00005 * 0
nc = 15
Tp = 2 * nc * pi / ω1

# Define the tau (Delay)
induce_time = 0
tau_fs = induce_time + 0
tau_lst = [tau_fs - 2pi/ω2, tau_fs + nc*pi/ω1 - 1.5pi/ω2,
    tau_fs + nc*pi/ω1 - pi/ω2, tau_fs + nc*pi/ω1 - 0.5pi/ω2, tau_fs + nc*2pi/ω1]
tau_thz = tau_lst[tau_id]

# Define the waveform
Et(t) = E0 * sin(ω1 * (t - tau_fs) / (2 * nc)) ^ 2 * cos(ω1 * (t - tau_fs)) * (t - tau_fs < Tp && t - tau_fs > 0)
Et_thz(t) = E0_thz * sin(ω2 * (t - tau_thz)) * (t - tau_thz < 2pi / ω2 && t - tau_thz > 0)
Et_thz_window(t) = flap_top_windows_f(t, 0, induce_time * 2, 1/2, right_flag = false)

T_total = tau_fs + Tp
steps = Int64(T_total ÷ delta_t)
t_linspace = create_linspace(steps, delta_t)

Et_data_fs = Et.(t_linspace)
Et_data_thz = (E0_c .+ Et_thz.(t_linspace)) #.* Et_thz_window.(t_linspace)

Et_data = Et_data_fs + Et_data_thz
At_data = -get_integral(Et_data, delta_t)

# store the figures.
fig1 = plot(At_data)
fig2 = plot([Et_data_fs (Et_data_thz) * 100])
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