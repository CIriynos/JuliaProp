import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# 11_7_2.jl  研究椭偏光程序是否正确？用两个方向(x, y)跑线偏振样例就好！

# 现在，托偏光程序绝对是OK的了！！！！

# __TASK_ID = parse(Int64, ARGS[1])
__TASK_ID = 2

# Basic Parameters
Nr = 5000
Δr = 0.2
l_num = 100
Δt = 0.05
Z = 1.0
Ri_tsurf = 800.0
po_func_r = coulomb_potiential_zero_fixed_COS(600.0, 800.0)
rmax = Nr * Δr  # rmax = 1000.0
absorb_func = absorb_boundary_r(rmax, Ri_tsurf, pow_value=8.0, max_value=100.0)
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);


# Create Physical World and Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);


# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-9);
crt_shwave = deepcopy(init_wave_list[1]);
get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5


# Define Laser Here.
omega = 0.057             # λ = 800 nm
Ip = 0.5
E0 = 0.1                  # E0 = 0.1 (3.51 × 10^14 W⋅cm-2)
nc = 6
steps = Int64((2 * nc * pi / omega) ÷ Δt)
actual_duration = steps * Δt
t_linspace = create_linspace(steps, Δt)
gamma = omega * sqrt(2 * Ip) / E0

Ax(t) = (E0 / omega) * (sin(omega * t / 2.0 / nc) ^ 2) * sin(omega * (t - nc*pi/omega)) * (t < (2 * nc * pi / omega))

if __TASK_ID == 1           # THz on X-axis
    At_data_x = Ax.(t_linspace)
    At_data_y = zeros(Float64, steps)
    At_data_z = zeros(Float64, steps)
elseif __TASK_ID == 2       # THz on Y-axis
    At_data_x = zeros(Float64, steps)
    At_data_y = Ax.(t_linspace)
    At_data_z = zeros(Float64, steps)
end
plot([At_data_x At_data_y])

Et_data_x = -get_derivative_two_order(At_data_x, Δt)
Et_data_y = -get_derivative_two_order(At_data_y, Δt)


# Define k Space
k_delta = 0.01
kmin = 0.01
kmax = 2.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))


# Define recording range for HHG
Tp = 2 * nc * pi / omega

# # Propagation
# hhg_integral_t, phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# # Store Data
# example_name = "11_7_2_$(__TASK_ID)"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
#     write(file, "hhg_integral_t", hhg_integral_t)
# end


# Retrieve Data.
example_name = "11_7_2_$(__TASK_ID)"
hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")


# HHG
Tp = 2 * nc * pi / omega
id_range = 1: steps
# println(id_range)
hhg_xy_t = -hhg_integral_t .- (Et_data_x .+ im .* Et_data_y)
hhg_len = steps

hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(t_linspace, 0, Tp)

hhg_spectrum_x = fft(real.(hhg_xy_t[id_range]) .* hhg_windows_data[id_range])
hhg_spectrum_y = fft(imag.(hhg_xy_t[id_range]) .* hhg_windows_data[id_range])
hhg_spectrum_x_norm = norm.(hhg_spectrum_x) .^ 2
hhg_spectrum_y_norm = norm.(hhg_spectrum_y) .^ 2
hhg_spectrum = hhg_spectrum_x_norm + hhg_spectrum_y_norm

spectrum_range = 1:200
plot([norm.(hhg_spectrum_x)[spectrum_range] norm.(hhg_spectrum_y)[spectrum_range]], yscale=:log10)
# plot([real.(hhg_integral_t) imag.(hhg_xy_t)])