import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# 12_4_1_eps0.jl 
# 之前的xy偏振求HHG的计算方法有误，采用新的方法，我们用线偏振光测试之


# Plotting / Task-based Sweeping
__TASK_ID = 2
# __TASK_ID = parse(Int64, ARGS[1])

tau_id = __TASK_ID


# Basic Parameters
Nr = 5000
Δr = 0.2
l_num = 60
Δt = 0.05
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = 800.0
po_func_r_shit(r) = -1.33 * (r^2 + 0.5) ^ (-0.5) * flap_top_windows_f(r, 0, Ri_tsurf, 1/4, left_flag=false)
absorb_func = absorb_boundary_r(rmax, Ri_tsurf, pow_value=8.0, max_value=100.0)


# Create Physical World and Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r_shit, Z, absorb_func)
rt = create_tdse_rt_sh(pw);


# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-8);
crt_shwave = deepcopy(init_wave_list[1]);
en = get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5
println("Energy of ground state: ", en)


# Define the Laser. 
THZ_X = 1
THZ_Y = 2
thz_direction = THZ_X

induce_time = 0
# tau = tau_var
eps1 = 0.8
E_peak1 = 0.0533
E_peak2 = 0.0001           # 200 kV/cm
E_constant_x = 0.0001      # 200 kV/cm

ω1 = 0.05693    # 800 nm (375 THz)
ω2 = ω1 / 25    # 20 μm (15.0 THz)
E1x = E_peak1 * (1.0 / sqrt(eps1 ^ 2 + 1.0))
E2x = E_peak2 * (thz_direction == THZ_X)
E1y = E_peak1 * (eps1 / sqrt(eps1 ^ 2 + 1.0))
E2y = E_peak2 * (thz_direction == THZ_Y)
nc1 = 15.0
nc2 = 1.0
phase_1x = 0.5pi
phase_2x = 0.0
phase_1y = 0.0
phase_2y = 0.0    
# tau_fs = (tau < 0.0) * abs(tau) + induce_time
# tau_thz = (tau >= 0.0) * abs(tau) + induce_time
# tau_lst = [500 - 2pi/ω2, 500 + nc1*pi/ω1 - 1.5pi/ω2,
#     500 + nc1*pi/ω1 - pi/ω2, 500 + nc1*pi/ω1 - 0.5pi/ω2, 500 + nc1*2pi/ω1]

tau_fs = induce_time + 0
tau_lst = [tau_fs - 2pi/ω2; range(tau_fs - 2pi/ω2 + tau_fs, tau_fs + nc1*2pi/ω1 - tau_fs, 14); tau_fs + nc1*2pi/ω1]
tau_thz = induce_time + tau_lst[tau_id]

Ex_fs(t) = (E1x) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1x) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1))
Ex_thz(t) = (E2x) * sin(ω2 * (t - tau_thz) / 2 / nc2)^0 * sin(ω2 * (t - tau_thz) + phase_2x) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

Ey_fs(t) = (E1y) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1y) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1))
Ey_thz(t) = (E2y) * sin(ω2 * (t - tau_thz) / 2 / nc2)^0 * sin(ω2 * (t - tau_thz) + phase_2y) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

E_thz_window(t) = flap_top_windows_f(t, 0, tau_fs * 2, 1/2, right_flag = false)

Ex(t) = Ex_fs(t) + (Ex_thz(t) + E_constant_x) * E_thz_window(t)
Ey(t) = Ey_fs(t) + Ey_thz(t) * E_thz_window(t)

steps_1 = Int64((tau_fs + 2 * nc1 * pi / ω1) ÷ Δt + 1)
steps_2 = Int64((tau_thz + 2 * nc2 * pi / ω2) ÷ Δt + 1)
steps_laser = steps_1
steps = steps_laser * 1
t_linspace = create_linspace(steps, Δt)

Et_data_x = Ex.(t_linspace)
Et_data_y = Ey.(t_linspace)
Et_data_z = zeros(Float64, steps)

At_data_x = -get_integral(Et_data_x, Δt)
At_data_y = -get_integral(Et_data_y, Δt)
At_data_z = -get_integral(Et_data_z, Δt)

atp = plot([At_data_x At_data_y])
etp = plot(t_linspace, [Ex_fs.(t_linspace) Ey_fs.(t_linspace) ((Ex_thz.(t_linspace) .+ E_constant_x)) .* 500],
    labels = ["E_fs_x" "E_fs_y" "E_thz"],
    xlabel="Time t",
    ylabel="E (a.u.)",
    title="fs Laser & THz Field Visualization (τ3)",
    guidefont=Plots.font(14, "Times"),
    tickfont=Plots.font(14, "Times"),
    titlefont=Plots.font(18, "Times"),
    legendfont=Plots.font(10, "Times"),
    margin = 5 * Plots.mm)

# Define k Space
k_delta = 0.01
kmin = 0.01
kmax = 1.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))

# # Propagation
# crt_shwave = deepcopy(init_wave_list[1])
# hhg_integral_t_1, hhg_integral_t_2, hhg_integral_t_3 = tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# # Store Data
# example_name = "12_4_1_eps0_elli_3"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "hhg_integral_t_1", hhg_integral_t_1)
#     write(file, "hhg_integral_t_2", hhg_integral_t_2)
#     write(file, "hhg_integral_t_3", hhg_integral_t_3)
# end

# # Propagation
# crt_shwave_ln = deepcopy(init_wave_list[1])
# hhg_integral_t_ln, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave_ln, pw, rt, At_data_x, steps, Ri_tsurf);

# # Store Data
# example_name = "12_4_1_eps0_ln_2"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave_ln", hcat(crt_shwave_ln...))
#     write(file, "hhg_integral_t_ln", hhg_integral_t_ln)
# end

# # Propagation
# crt_shwave = deepcopy(init_wave_list[1])
# hhg_integral_t_1, hhg_integral_t_2, hhg_integral_t_3 = tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# # Store Data
# example_name = "12_4_1_eps0.8_elli_4"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "hhg_integral_t_1", hhg_integral_t_1)
#     write(file, "hhg_integral_t_2", hhg_integral_t_2)
#     write(file, "hhg_integral_t_3", hhg_integral_t_3)
# end


# Retrieve Data.
example_name = "12_4_1_eps0_elli_3"
hhg_integral_t_1 = retrieve_mat(example_name, "hhg_integral_t_1")
hhg_integral_t_2 = retrieve_mat(example_name, "hhg_integral_t_2")
hhg_integral_t_3 = retrieve_mat(example_name, "hhg_integral_t_3")
hhg_integral_t_ln = retrieve_mat("12_4_1_eps0_ln_2", "hhg_integral_t_ln")
# crt_shwave_elli = retrieve_obj("12_4_1_eps0.8_elli_4", "crt_shwave")

plot([real.(hhg_integral_t_1) imag.(hhg_integral_t_1)])
plot([real.(hhg_integral_t_2) imag.(hhg_integral_t_2)])

# HHG
Tp = 2 * pi * nc1 / ω1
hhg_xy_t = -hhg_integral_t_1 .- (Et_data_x .+ im .* Et_data_y)
hhg_delta_k = 2pi / steps / Δt
hhg_k_linspace = [hhg_delta_k * i for i = 1: steps]
shg_id = Int64(floor(2 * ω1 / hhg_delta_k))

hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(t_linspace, tau_fs + 0, Tp + tau_fs - 0)

hhg_spectrum_x = fft(real.(hhg_xy_t) .* hhg_windows_data)
hhg_spectrum_y = fft(imag.(hhg_xy_t) .* hhg_windows_data)
hhg_spectrum_ln = norm.(fft(real.(hhg_integral_t_ln) .* hhg_windows_data)) .^ 2
hhg_spectrum = norm.(hhg_spectrum_x) .^ 2 + norm.(hhg_spectrum_y) .^ 2

spectrum_range = 1:500
plot(hhg_k_linspace[spectrum_range] ./ ω1, hhg_spectrum[spectrum_range], yscale=:log10)

plot(hhg_k_linspace[spectrum_range] ./ ω1,
    [hhg_spectrum[spectrum_range] hhg_spectrum_ln[spectrum_range]], yscale=:log10)

plot(hhg_k_linspace[spectrum_range] ./ ω1,
    [norm.(hhg_spectrum_x[spectrum_range]) norm.(hhg_spectrum_y[spectrum_range])], yscale=:log10)





# # plot crt_shwave
# x0 = 1.0
# xs = -500.0: 5: 500.0
# ys = -500.0: 5: 500.0
# zs = -500.0: 5: 500.0
# len = length(ys)
# res_mat = zeros(ComplexF64, len, len)
# theta_phi_map = zeros(ComplexF64, len, len)
# ylm_buffer = []
# rs = get_linspace(pw.shgrid.rgrid)

# for (i, y) in enumerate(ys)
#     for (j, z) in enumerate(zs)
#         println("i=$i, j=$j")
#         r, theta, phi = xyz_to_sphere(y, x0, z)
#         push!(ylm_buffer, computeYlm(theta, phi, pw.l_num - 1))
#     end
# end

# Threads.@threads for i = 1: len
#     for j = 1: len
#         y = ys[i]
#         z = zs[j]
#         # println("i=$i, j=$j")
#         for id = 1: pw.l_num ^ 2
#             l = rt.lmap[id]
#             m = rt.mmap[id]

#             r, theta, phi = xyz_to_sphere(y, x0, z)
#             r_id = max(1, Int64((r - rs[1]) ÷ Δr))
#             res_mat[i, j] += (crt_shwave[id][r_id] / rs[r_id]) * ylm_buffer[(i - 1) * length(ys) + j][(l, m)]
#         end
#     end
# end

# heatmap(log10.(norm.(res_mat)))