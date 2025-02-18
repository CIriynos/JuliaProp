import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# 11_8_2.jl  改THz和恒场大小

tau_list = [-7000, -2850, -1930, -1050]        # omega / 50

# task-based sweeping
# __TASK_ID = 1
__TASK_ID = parse(Int64, ARGS[1])
tau_var = tau_list[__TASK_ID]


# Basic Parameters
grid_ratio = 1
Nr = 5000 * grid_ratio
Δr = 0.2 / grid_ratio
l_num = 60
Δt = 0.05 / grid_ratio
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

induce_time = 1
tau = tau_var
eps1 = 0.3
E_peak1 = 0.0533
E_peak2 = 0.000177 / 4
E_constant_x = 0.00005

ω1 = 0.05693    # 800 nm (375 THz)
ω2 = ω1 / 50    # 40 μm (7.5 THz)
E1x = E_peak1 * (1.0 / sqrt(eps1 ^ 2 + 1.0))
E2x = E_peak2 * (thz_direction == THZ_X)
E1y = E_peak1 * (eps1 / sqrt(eps1 ^ 2 + 1.0))
E2y = E_peak2 * (thz_direction == THZ_Y)
nc1 = 15.0
nc2 = 1.0
phase_1x = 0.5pi
phase_2x = 0.0
phase_1y = 0.0
phase_2y = 0.0    # ???
# tau_fs = (tau < 0.0) * abs(tau) + induce_time
# tau_thz = (tau >= 0.0) * abs(tau) + induce_time
tau_fs = induce_time + 500
tau_thz = induce_time + tau + 500

# Ex(t) = (E1x) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1x) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1)) + 
#         (E2x) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2x) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2)) + 
#         (E_constant_x) *  flap_top_windows_f(t, 0, 2 * induce_time, 1/2, right_flag = false)

# Ey(t) = (E1y) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1y) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1)) + 
#         (E2y) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2y) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

Ex_fs(t) = (E1x) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1x) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1))
Ex_thz(t) = (E2x) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2x) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

Ey_fs(t) = (E1y) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1y) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1))
Ey_thz(t) = (E2y) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2y) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

E_thz_window(t) = flap_top_windows_f(t, 0, 500 * 2, 1/2, right_flag = false)

Ex(t) = Ex_fs(t) + (Ex_thz(t) + E_constant_x) * E_thz_window(t)
Ey(t) = Ey_fs(t) + Ey_thz(t) * E_thz_window(t)

steps_1 = Int64((2 * nc1 * pi / ω1) ÷ Δt + 1) + Int64(tau_fs ÷ Δt + 1)
steps_2 = Int64((2 * nc2 * pi / ω2) ÷ Δt + 1) + Int64(tau_thz ÷ Δt + 1)
# fs: 1655.5  thz: 1103.65
# -1500 -> 2000
residual_steps = 2000
steps = steps_1 + residual_steps
t_linspace = create_linspace(steps, Δt)

Et_data_x = Ex.(t_linspace)
Et_data_y = Ey.(t_linspace)
Et_data_z = zeros(Float64, steps)

At_data_x = -get_integral(Et_data_x, Δt)
At_data_y = -get_integral(Et_data_y, Δt)
At_data_z = -get_integral(Et_data_z, Δt)

atp = plot([At_data_x At_data_y])
etp = plot([Ex_fs.(t_linspace) Ey_fs.(t_linspace) ((Ex_thz.(t_linspace) .+ E_constant_x)) .* 200],
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

# Define recording range for HHG
Tp = 2 * nc1 * pi / ω1
start_rcd_step = (Int64(tau_fs ÷ Δt))
end_rcd_step = (Int64((tau_fs + Tp) ÷ Δt) + 1)


# Propagation
hhg_integral_t, phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf, start_rcd_step=start_rcd_step, end_rcd_step=end_rcd_step);

# Store Data
example_name = "11_8_2_$(tau)"
h5open("./data/$example_name.h5", "w") do file
    write(file, "crt_shwave", hcat(crt_shwave...))
    write(file, "phi_record", hcat(phi_record...))
    write(file, "dphi_record", hcat(dphi_record...))
    write(file, "hhg_integral_t", hhg_integral_t)
end


# # Retrieve Data.
# example_name = "11_8_2_$(tau)"
# hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")


# # HHG
# Tp = 2 * nc1 * pi / ω1
# id_range = (Int64(tau_fs ÷ Δt)): (Int64((tau_fs + Tp) ÷ Δt) + 1)
# # println(id_range)
# hhg_xy_t = -hhg_integral_t .- (Et_data_x .+ im .* Et_data_y)
# hhg_len = steps

# hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
# hhg_windows_data = hhg_windows_f.(t_linspace, tau_fs, Tp + tau_fs)

# hhg_spectrum_x = fft(real.(hhg_xy_t[id_range]) .* hhg_windows_data[id_range])
# hhg_spectrum_y = fft(imag.(hhg_xy_t[id_range]) .* hhg_windows_data[id_range])
# hhg_spectrum_x_norm = norm.(hhg_spectrum_x) .^ 2
# hhg_spectrum_y_norm = norm.(hhg_spectrum_y) .^ 2
# hhg_spectrum = hhg_spectrum_x_norm + hhg_spectrum_y_norm


# spectrum_range = 1:200
# pl4 = plot([norm.(hhg_spectrum_x)[spectrum_range]], yscale=:log10)
# hhg_data_4 = norm.(hhg_spectrum_x)
# shg_yield4 = norm.(hhg_spectrum_x)[31]

# plot([real.(hhg_integral_t) imag.(hhg_integral_t)])

# plot([tau_list; 0], [shg_yield1, shg_yield2, shg_yield3, shg_yield4, shg_yield1],
#     xlabel="Delay Time τ",
#     ylabel="SHG Yield (arb.u.)",
#     title="SHG Variation at 9.38 THz (0.01ω)",
#     guidefont=Plots.font(14, "Times"),
#     tickfont=Plots.font(14, "Times"),
#     titlefont=Plots.font(18, "Times"),
#     legendfont=Plots.font(10),
#     margin = 5 * Plots.mm)

# plot([hhg_data_1[spectrum_range] hhg_data_2[spectrum_range] hhg_data_3[spectrum_range] hhg_data_4[spectrum_range]],
#     yscale=:log10,
#     xlabel="Delay Time τ",
#     ylabel="Yield (arb.u.)",
#     title="HHG Spectrum in Direction of x-axis",
#     labels=["τ1 Farest Value" "τ2 Trough Value" "τ3 Mid Value" "τ4 Peak Value"],
#     guidefont=Plots.font(14, "Times"),
#     tickfont=Plots.font(14, "Times"),
#     titlefont=Plots.font(18, "Times"),
#     legendfont=Plots.font(10, "Times"),
#     margin = 5 * Plots.mm)