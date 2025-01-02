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
EMPTY_TASK_ID = 17
# __TASK_ID = 1
__TASK_ID = parse(Int64, ARGS[1])

tau_id = __TASK_ID


# Basic Parameters
grid_ratio = 1
Nr = 5000 * grid_ratio
Δr = 0.2 / grid_ratio
l_num = 60
Δt = 0.05 / grid_ratio
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = 800.0
po_func_r(r) = -1.0 * (r^2) ^ (-0.5) * flap_top_windows_f(r, 0, Ri_tsurf, 1/4, left_flag=false)
absorb_func = absorb_boundary_r(rmax, Ri_tsurf, pow_value=8.0, max_value=100.0)


# Create Physical World and Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
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
eps1 = 0.5
E_peak1 = 0.0533
E_peak2 = 0.00002           # 100 kV/cm
E_constant_x = 0.00002      # 100 kV/cm
if tau_id == EMPTY_TASK_ID
    E_peak2 = 0.0
    E_constant_x = 0.0
end

ω1 = 0.05693    # 800 nm (375 THz)
ω2 = ω1 / 100    # 80 μm (3.75 THz)
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

tau_fs = induce_time + 500
tau_lst = [tau_fs - 2pi/ω2; range(tau_fs - 2pi/ω2 + tau_fs, tau_fs + nc1*2pi/ω1 - tau_fs, 14); tau_fs + nc1*2pi/ω1]
# tau_lst = [tau_fs - 2pi/ω2, tau_fs + nc1*pi/ω1 - 1.5pi/ω2,
#     tau_fs + nc1*pi/ω1 - pi/ω2, tau_fs + nc1*pi/ω1 - 0.5pi/ω2, tau_fs + nc1*2pi/ω1, 0.0]
tau_thz = induce_time + tau_lst[tau_id]
mid_point = tau_fs + nc1*pi/ω1 - pi/ω2

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
steps = steps_laser + 2000
t_linspace = create_linspace(steps, Δt)


Et_data_x = Ex.(t_linspace)
Et_data_y = Ey.(t_linspace)
Et_data_z = zeros(Float64, steps)
At_data_x = -get_integral(Et_data_x, Δt)
At_data_y = -get_integral(Et_data_y, Δt)
At_data_z = -get_integral(Et_data_z, Δt)

atp = plot([At_data_x At_data_y])
etp = plot(t_linspace, [Ex_fs.(t_linspace) Ey_fs.(t_linspace) ((Ex_thz.(t_linspace) .+ E_constant_x)) .* E_thz_window.(t_linspace) .* 500],
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


# Propagation
crt_shwave = deepcopy(init_wave_list[1])
hhg_integral_t_1, hhg_integral_t_2, hhg_integral_t_3 = tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);
# hhg_integral_t_1, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw, rt, At_data_x, steps, Ri_tsurf);

# Store Data
example_name = "12_22_2_$(tau_id)"
h5open("./data/$example_name.h5", "w") do file
    write(file, "crt_shwave", hcat(crt_shwave...))
    write(file, "hhg_integral_t_1", hhg_integral_t_1)
    write(file, "hhg_integral_t_2", hhg_integral_t_2)
    write(file, "hhg_integral_t_3", hhg_integral_t_3)
end


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

