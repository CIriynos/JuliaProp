import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# 11_1_2_coherent.jl
# 1. 为什么omega/50也不行，不能重复出双峰式！！！？？
# 试试eps=0，并且将E_THz减小一半试试？
# + add
# 再试试用线偏振光程序？

ps = []
tps = []
Ats = []
Ets = []
fk = []

# Define Sweeping Args
tau_list = [-7000, -2850, -1950, -1000]
iter_list = tau_list

# # 1. Task-based sweeping
# __TASK_ID = parse(Int64, ARGS[1])
# sub_iter_list = iter_list[__TASK_ID]
# println("task id = $__TASK_ID")
# println(sub_iter_list)

# 2. Just for look-up ...
sub_iter_list = iter_list

# in-program sweeping
for tau_var in sub_iter_list

# Basic Parameters
grid_ratio = 2
Nr = 10000 * grid_ratio
Δr = 0.2 / grid_ratio
l_num = 50
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
eps1 = 0.0
E_ratio = 1
E_peak1 = 0.0533 * E_ratio
E_peak2 = 0.001 * 10
E_constant_x = 0

ω1 = 0.05693    # 800 nm (375 THz)
ω2 = ω1 / 50    # 16 μm (18.75 THz)
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

Ex_c(t) = (E_constant_x) * flap_top_windows_f(t, 0, 2 * induce_time, 1/2, right_flag = false)
E_thz_window(t) = flap_top_windows_f(t, 0, 500 * 2, 1/2, right_flag = false)

Ex(t) = Ex_fs(t) + Ex_thz(t) * E_thz_window(t)
Ey(t) = Ey_fs(t) + Ey_thz(t) * E_thz_window(t)

steps_1 = Int64((2 * nc1 * pi / ω1) ÷ Δt + 1) + Int64(tau_fs ÷ Δt + 1)
steps_2 = Int64((2 * nc2 * pi / ω2) ÷ Δt + 1) + Int64(tau_thz ÷ Δt + 1)
# fs: 1655.5  thz: 1103.65
# -1500 -> 2000
residual_steps = 2000
steps = steps_1 + residual_steps
t_linspace = create_linspace(steps, Δt)

# # modify here.
# Tp = 2 * nc1 * pi / ω1
# mid_area_len = 1000
# mid_timing = tau_fs + Tp / 2
# mid_Et_x = Ex_thz(mid_timing) + E_constant_x
# mid_Et_y = Ey_thz(mid_timing)
# Et_data_x = Ex_fs.(t_linspace) .+ mid_Et_x * flap_top_windows_f.(t_linspace, mid_timing - mid_area_len, mid_timing + mid_area_len, 1/4)
# Et_data_y = Ey_fs.(t_linspace) .+ mid_Et_y * flap_top_windows_f.(t_linspace, mid_timing - mid_area_len, mid_timing + mid_area_len, 1/4)
# Et_data_z = zeros(Float64, steps)
# push!(fk, mid_Et_x + im * mid_Et_y)

Et_data_x = zeros(Float64, steps)
Et_data_y = zeros(Float64, steps)
Et_data_z = Ex.(t_linspace)

At_data_x = -get_integral(Et_data_x, Δt)
At_data_y = -get_integral(Et_data_y, Δt)
At_data_z = -get_integral(Et_data_z, Δt)

atp = plot([At_data_z])
etp = plot([Ex_fs.(t_linspace) (Ex_thz.(t_linspace) .* E_thz_window.(t_linspace)) .* 200])
push!(Ats, atp)
push!(Ets, etp)

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

# # Start Propagation
# hhg_integral_t, energy_list = tdseln_sh_mainloop_length_gauge_hhg(crt_shwave, pw, rt, Et_data_z, steps, m = 0, start_rcd_step=start_rcd_step, end_rcd_step=end_rcd_step)

# # Store Data
# example_name = "11_4_SHG_research_$(tau)"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "hhg_integral_t", hhg_integral_t)
# end

# Retrieve Data.
example_name = "11_4_SHG_research_$(tau)"
hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")


# HHG Part
Tp = 2 * nc1 * pi / ω1
id_range = (Int64(tau_fs ÷ Δt)): (Int64((tau_fs + Tp) ÷ Δt) + 1)
hhg_z_t = -hhg_integral_t .- Et_data_z
hhg_len = steps

hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(t_linspace, tau_fs, Tp + tau_fs)

hhg_spectrum_z = fft(real.(hhg_z_t[id_range]) .* hhg_windows_data[id_range])
hhg_spectrum = norm.(hhg_spectrum_z) .^ 2

push!(ps, hhg_spectrum)
push!(tps, hhg_z_t[id_range])

end


# Display Part
pt = plot()
pp = []
rg = 1: 100
for (i, p) in enumerate(ps)
hhg_len = length(p)
hhg_delta_k = 2pi / hhg_len / 0.05
hhg_k_linspace = [hhg_delta_k * i for i = 1: hhg_len]

plot!(pt, rg, log10.(norm.(p))[rg], label="Delay τ=$(tau_list[i])")
push!(pp, p[32])
end
pt

plot([real.(tps[1]) real.(tps[2])])
scatter([-7000, -2850, -1950, -1000, 2000], [pp[1], pp[2], pp[3], pp[4], pp[1]])
