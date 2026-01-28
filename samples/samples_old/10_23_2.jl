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
# 测试smoothness，寻找Atda大小与网格粗细的关系

# Mainloop for Sweeping
at_data_figs = []
et_data_figs = []
part_of_hhg_data = []
smooth_records = []
tau_list = [285]

for ratio = [0.2, 0.5, 1.0, 1.5, 2.0, 4.0]

# Define Basic Parameters
Nx = Int64(3600 * ratio)
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


induce_time = 0
thz_rate = 1
fs_rate = 2
tau = 285
# tau_fs = induce_time + (tau < 0.0) * abs(tau)
# tau_thz = induce_time + (tau >= 0.0) * abs(tau)

# Define Laser.
tau_fs = induce_time + 500  # fixed
tau_thz = induce_time + tau + 500
omega = 0.057           # 800 nm (375 THz)
omega_thz = 0.1omega    # 8000 nm (37.5 THz)
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
Et_data = Et_data_fs + Et_data_thz + Et_data_c
At_data = -get_integral(Et_data, delta_t)

# store the figures.
fig1 = plot(At_data)
fig2 = plot([Et_data_fs (Et_data_thz + Et_data_c) * 50])
push!(at_data_figs, fig1)
push!(et_data_figs, fig2)

# TDSE 1d
crt_wave = deepcopy(init_wave)
Xi_data, hhg_integral, energy_list, smooth_record = tdse_laser_fd1d_mainloop_penta(crt_wave, rt1d, pw1d, At_data, steps, Xi)
push!(smooth_records, smooth_record)

end

plot([maximum(smooth_records[i]) for i in 1: length(smooth_records)])