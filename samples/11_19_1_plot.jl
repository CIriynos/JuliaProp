import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW


# Plotting / Task-based Sweeping
shg_yields = []
tau_lst = []
hhg_data = []
etps = []
atps = []
hhg_t = []
shg_id::Int64 = 0
mid_point::Float64 = 0

for tau_id = 1: 16

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


# Define the Laser. 
THZ_X = 1
THZ_Y = 2
thz_direction = THZ_X

induce_time = 0
# tau = tau_var
eps1 = 0.1
E_peak1 = 0.0533
E_peak2 = 0.000005           # 25.7 kV/cm
E_constant_x = 0.000005      # 25.7 kV/cm
E_thz = 0.00017

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

tau_fs = induce_time + 500
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
push!(etps, etp)

# Define k Space
k_delta = 0.01
kmin = 0.01
kmax = 1.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))



# Retrieve Data.
example_name = "11_19_1_$(tau_id)"
hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")


# HHG
Tp = 2 * nc1 * pi / ω1
id_range = 1: steps
# println(id_range)
hhg_xy_t = -hhg_integral_t .- (Et_data_x .+ im .* Et_data_y)
hhg_len = steps
hhg_delta_k = 2pi / hhg_len / Δt
hhg_k_linspace = [hhg_delta_k * i for i = 1: hhg_len]
shg_id = Int64(floor(2 * ω1 / hhg_delta_k))


hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
# hhg_windows_data = hhg_windows_f.(t_linspace, tau_fs, tau_fs + Tp)
hhg_windows_data = hhg_windows_f.(t_linspace, tau_fs, steps * Δt)

hhg_spectrum_x = fft(real.(hhg_xy_t[id_range]) .* hhg_windows_data[id_range])
hhg_spectrum_y = fft(imag.(hhg_xy_t[id_range]) .* hhg_windows_data[id_range])
hhg_spectrum_x_norm = norm.(hhg_spectrum_x) .^ 1
hhg_spectrum_y_norm = norm.(hhg_spectrum_y) .^ 1
hhg_spectrum = sqrt.(hhg_spectrum_x_norm .^ 2 .+ hhg_spectrum_y_norm .^ 2)


push!(shg_yields, hhg_spectrum[shg_id])
push!(hhg_data, hhg_spectrum_y_norm)
push!(hhg_t, hhg_integral_t)
println("shg_id = $shg_id")

end

p1 = plot(tau_lst, shg_yields,
    xlabel="Delay Time τ",
    ylabel="SHG Yield (arb.u.)",
    title="SHG Variation at 15.0 THz (0.04ω)",
    label="samples",
    guidefont=Plots.font(14, "Times"),
    tickfont=Plots.font(14, "Times"),
    titlefont=Plots.font(18, "Times"),
    legendfont=Plots.font(10),
    linewidth=2,
    margin = 5 * Plots.mm)
plot!(p1, [mid_point, mid_point],
    [minimum(shg_yields) * 0.6, maximum(shg_yields) * 1.1],
    ls=:dashdot, label="midpoint", linewidth=1.5)
plot!(p1, [-600, -600],
    [minimum(shg_yields) * 0.6, maximum(shg_yields) * 1.1],
    ls=:dashdot, label="actual", linewidth=1.5)
mid_point_shift = mid_point - (-600)
p1

spectrum_range = 1:250
plot([hhg_data[1][spectrum_range] hhg_data[4][spectrum_range] hhg_data[8][spectrum_range] hhg_data[12][spectrum_range]],
    yscale=:log10,
    xlabel="N Times of ωfs",
    ylabel="Yield (arb.u.)",
    title="HHG Spectrum in Direction of z-axis",
    labels=["τ1 Farest Value" "τ2 Trough Value" "τ3 Mid Value" "τ4 Peak Value"],
    guidefont=Plots.font(14, "Times"),
    tickfont=Plots.font(14, "Times"),
    titlefont=Plots.font(18, "Times"),
    legendfont=Plots.font(10, "Times"),
    margin = 5 * Plots.mm,
    ylimit=(1e-3, 1e3))

# plot(real.(hhg_t[13]))