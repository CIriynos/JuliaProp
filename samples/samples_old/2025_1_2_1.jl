import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# 12_22_2.jl  --> 论文阶段！！
# eps=0.5  E_ratio=1  扫16个点
# 这里，我们把THz频率降低，看看shift影响如何？


# Plotting / Task-based Sweeping
tau_id = 1


# Basic Parameters
Nr = 5000
Δr = 0.2
l_num = 60
Δt = 0.05
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = rmax * 0.8
po_func_r = coulomb_potiential_zero_fixed_windows(Ri_tsurf)
absorb_func = absorb_boundary_r(rmax, Ri_tsurf)


# Create Physical World & Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);


# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-8);
crt_shwave = deepcopy(init_wave_list[1]);
en = get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5
println("Energy of ground state: ", en)


# Define the Laser. 
E_fs = 0.0533
E_thz = 0.0002
E_const = 0.00002
ω_fs = 0.05693
ω_thz = ω_fs / 25
nc = 6
tau_fs = 500
tau_thz = get_1c_thz_delay_list_selected(ω_fs, tau_fs, nc, ω_thz)[1]

Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.5)
Ex_thz, Ey_thz, Ez_thz = light_pulse(ω_fs, E_fs, nc, tau_thz)
E_dc = dc_bias(0, tau_fs, tmax, tmax * 2)

# Create Time Linspace & Data
ts = create_linespace(Int64(tmax ÷ Δt) + 1, Δt)
Et_data_x = Ex_fs.(ts) .+ Ex_thz.(ts) .+ E_dc.(ts)
Et_data_y = Ey_fs.(ts) .+ Ey_thz.(ts)
Et_data_z = no_light.(ts)
At_data_x = -get_integral(Et_data_x, Δt)
At_data_y = -get_integral(Et_data_y, Δt)
At_data_z = -get_integral(Et_data_z, Δt)

plot_pump_probe_light_field(Ex_fs, Ey_fs, Ex_thz, E_dc)


# Define k Space for t-surf
k_delta = 0.01
kmin = 0.01
kmax = 1.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))


# # Propagation
# crt_shwave = deepcopy(init_wave_list[1])
# hhg_integral_t_1, hhg_integral_t_2, hhg_integral_t_3 = tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);
# # hhg_integral_t_1, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw, rt, At_data_x, steps, Ri_tsurf);

# # Store Data
# example_name = "12_22_2_$(tau_id)"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "hhg_integral_t_1", hhg_integral_t_1)
#     write(file, "hhg_integral_t_2", hhg_integral_t_2)
#     write(file, "hhg_integral_t_3", hhg_integral_t_3)
# end


# # Retrieve Data.
# example_name = "12_20_1_$(tau_id)"
# hhg_integral_t_1 = retrieve_mat(example_name, "hhg_integral_t_1")

# # HHG
# Tp = 2 * pi * nc1 / ω1
# hhg_xy_t = -hhg_integral_t_1 #.- (Et_data_x .+ im .* Et_data_y)
# hhg_delta_k = 2pi / steps / Δt
# hhg_k_linspace = [hhg_delta_k * i for i = 1: steps]
# shg_id = Int64(floor(2 * ω1 / hhg_delta_k))

# hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
# hhg_windows_data = hhg_windows_f.(t_linspace, tau_fs, Tp + tau_fs)

# hhg_spectrum_x = fft(real.(hhg_xy_t) .* hhg_windows_data)
# hhg_spectrum_y = fft(imag.(hhg_xy_t) .* hhg_windows_data)
# # hhg_spectrum_ln = norm.(fft(real.(hhg_integral_t_ln) .* hhg_windows_data)) .^ 2
# hhg_spectrum = norm.(hhg_spectrum_x) .^ 2 + norm.(hhg_spectrum_y) .^ 2

# spectrum_range = 1: Int64((15 * ω1) ÷ hhg_delta_k)
# plot(hhg_k_linspace[spectrum_range] ./ ω1, hhg_spectrum[spectrum_range], yscale=:log10)

# # plot(hhg_k_linspace[spectrum_range] ./ ω1,
# #     [hhg_spectrum[spectrum_range] hhg_spectrum_ln[spectrum_range]], yscale=:log10)

# plot(hhg_k_linspace[spectrum_range] ./ ω1,
#     [norm.(hhg_spectrum_x[spectrum_range]) norm.(hhg_spectrum_y[spectrum_range])],
#     yscale=:log10, ylimit=(1e-7, 1e3))

