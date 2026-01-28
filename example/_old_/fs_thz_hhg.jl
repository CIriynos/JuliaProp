import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

ps = []

Ats = []
Ets = []

# tau_list = -500: 100: 800
# tau_list = [-500, 900, 100, 300, 500]
tau_list = [-500, 100, 300, 500, 900]
for tau_var in tau_list 

# Basic Parameters
grid_ratio = 3
Nr = 5000 * grid_ratio
Δr = 0.2 / grid_ratio
l_num = 50
Δt = 0.05 / grid_ratio
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = 800.0
po_func_r = coulomb_potiential_zero_fixed_COS(600.0, 800.0)
absorb_func = absorb_boundary_r(rmax, Ri_tsurf, pow_value=8.0, max_value=100.0)

# Create Physical World and Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-9);
crt_shwave = deepcopy(init_wave_list[1]);
get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5


# Define the Laser. 
THZ_X = 1
THZ_Y = 2
thz_direction = THZ_X

induce_time = 400
tau = tau_var
eps1 = 0.3
E_peak1 = 0.0533
E_peak2 = 0.000177
E_constant_x = 0.0001

ω1 = 0.05693
ω2 = ω1 / 10
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
tau_fs = (tau < 0.0) * abs(tau) + induce_time
tau_thz = (tau >= 0.0) * abs(tau) + induce_time

# Ex(t) = (E1x) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1x) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1)) + 
#         (E2x) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2x) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2)) + 
#         (E_constant_x) *  flap_top_windows_f(t, 0, 2 * induce_time, 1/2, right_flag = false)

# Ey(t) = (E1y) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1y) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1)) + 
#         (E2y) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2y) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

Ex_fs(t) = (E1x) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1x) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1))
Ex_thz(t) = (E2x) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2x) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

Ey_fs(t) = (E1y) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1y) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1))
Ey_thz(t) = (E2y) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2y) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

Ex_c(t) = (E_constant_x) *  flap_top_windows_f(t, 0, 2 * induce_time, 1/2, right_flag = false)

Ex(t) = Ex_fs(t) + Ex_thz(t) + Ex_c(t)
Ey(t) = Ey_fs(t) + Ey_thz(t)

steps_1 = Int64((2 * nc1 * pi / ω1) ÷ Δt + 1) + Int64(tau_fs ÷ Δt + 1)
steps_2 = Int64((2 * nc2 * pi / ω2) ÷ Δt + 1) + Int64(tau_thz ÷ Δt + 1)
# fs: 1655.5  thz: 1103.65
# -1500 -> 2000
residual_steps = 2000
steps = max(steps_1, steps_2) + residual_steps
t_linspace = create_linspace(steps, Δt)

Et_data_x = Ex.(t_linspace)
Et_data_y = Ey.(t_linspace)
Et_data_z = zeros(Float64, steps)

At_data_x = -get_integral(Et_data_x, Δt)
At_data_y = -get_integral(Et_data_y, Δt)
At_data_z = -get_integral(Et_data_z, Δt)
atp = plot([At_data_x At_data_y])
etp = plot([Ex_fs.(t_linspace) (Ex_thz.(t_linspace) .+ Ex_c.(t_linspace)) .* 100])
push!(Ats, atp)
push!(Ets, etp)

# Define k Space
k_delta = 0.01
kmin = 0.01
kmax = 1.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))


# # Propagation
# hhg_integral_t, phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# δa_lm = isurf_rest_part(crt_shwave, k_linspace, last(t_linspace), Ri_tsurf, pw, rt)
# # tsurf_plot_energy_spectrum(δa_lm, k_linspace, pw)
# # tsurf_plot_xy_momentum_spectrum(δa_lm, k_space, pw, k_min=0.01)

# # Store Data
# example_name = "fs_thz_hhg_tau$(tau)_2"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
#     write(file, "δa_lm", hcat(δa_lm...))
#     write(file, "hhg_integral_t", hhg_integral_t)
# end


# Retrieve Data.
example_name = "fs_thz_hhg_tau$(tau)_2"
# crt_shwave = retrieve_obj(example_name, "crt_shwave")
# phi_record = retrieve_obj(example_name, "phi_record")
# dphi_record = retrieve_obj(example_name, "dphi_record")
hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")


# HHG
Tp = 2 * nc1 * pi / ω1
id_range = (Int64(tau_fs ÷ Δt)): (Int64((tau_fs + Tp) ÷ Δt) + 1)
# println(id_range)
hhg_xy_t = -hhg_integral_t .- (Et_data_x .+ im .* Et_data_y)
hhg_len = steps

hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(t_linspace, tau_fs, Tp + tau_fs)
# hhg_windows_data = ones(Float64, steps)

hhg_spectrum_x = fft(real.(hhg_xy_t[id_range]) .* hhg_windows_data[id_range])
hhg_spectrum_y = fft(imag.(hhg_xy_t[id_range]) .* hhg_windows_data[id_range])
hhg_spectrum = norm.(hhg_spectrum_x) .^ 2 + norm.(hhg_spectrum_y) .^ 2

push!(ps, hhg_spectrum)

end



pt = plot()
pp = []
rg = 1: 100
for (i, p) in enumerate(ps)
hhg_len = length(p)
hhg_delta_k = 2pi / hhg_len / 0.05
hhg_k_linspace = [hhg_delta_k * i for i = 1: hhg_len]

plot!(pt, rg, log10.(norm.(p))[rg], label="Delay τ=$(tau_list[i])")
push!(pp, p[29])
end
pt

k0 = (first(pp) - last(pp)) / (length(pp) - 1)
# y = first(pp) + k0 * (x - 1)
plot(tau_list[1: length(tau_list)], pp .- [(first(pp) .- k0 .* ((1: length(tau_list)) .- 1));])
plot(tau_list, pp)









# example_name = "fs_thz_hhg_tau-500.0"
# crt_shwave = retrieve_obj(example_name, "crt_shwave")
# phi_record = retrieve_obj(example_name, "phi_record")
# dphi_record = retrieve_obj(example_name, "dphi_record")
# hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")


# plot([real.(hhg_integral_t) imag.(hhg_integral_t)])

# plot(hhg_k_linspace[1: 100] ./ omega, log10.(norm.(hhg_spectrum_x))[1: 100])
# plot(hhg_k_linspace[1: 200] ./ omega, log10.(norm.(hhg_spectrum))[1: 200], xlabel="N times of ω=0.0569")


# # Retrieve Data.
# example_name = "fs_thz_hhg_tau$(tau)"
# crt_shwave = retrieve_obj(example_name, "crt_shwave")
# phi_record = retrieve_obj(example_name, "phi_record")
# dphi_record = retrieve_obj(example_name, "dphi_record")
# hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")




# # HHG
# hhg_xy_t = -hhg_integral_t .- (Et_data_x .+ im .* Et_data_y)
# hhg_len = length(hhg_xy_t)
# plot(norm.(hhg_xy_t))

# # two type of window function
# # hhg_window_f(t) = sin(t / (hhg_len / 2) * pi) ^ 2 * (t < hhg_len/2 && t > 0)
# # hhg_windows_data = hhg_window_f.(eachindex(hhg_xy_t) .- hhg_len/4)

# hhg_window_f(t) = sin(t / hhg_len * pi) ^ 2
# hhg_windows_data = hhg_window_f.(eachindex(hhg_xy_t))
# # plot(hhg_windows_data)

# hhg_spectrum_x = fft(real.(hhg_xy_t) .* hhg_windows_data)
# hhg_spectrum_y = fft(imag.(hhg_xy_t) .* hhg_windows_data)
# hhg_spectrum = norm.(hhg_spectrum_x) .^ 2 + norm.(hhg_spectrum_y) .^ 2


# hhg_delta_k = 2pi / hhg_len / Δt
# hhg_k_linspace = [hhg_delta_k * i for i = 1: hhg_len]

# plot(hhg_k_linspace[1: 100] ./ omega, log10.(norm.(hhg_spectrum_x))[1: 100])
# plot(hhg_k_linspace[1: 200] ./ omega, log10.(norm.(hhg_spectrum))[1: 200], xlabel="N times of ω=0.0569")




# # Get HHG Spectrum
# hhg_xy_t = -hhg_integral_t .- (Et_data_x .+ im .* Et_data_y)
# hhg_len = length(hhg_xy_t)
# hhg_window_f(t) = sin(t / (hhg_len / 2) * pi) ^ 2 * (t < hhg_len/2 && t > 0)
# hhg_windows_data = hhg_window_f.(eachindex(hhg_xy_t) .- hhg_len/4)
# # hhg_window_f(t) = sin(t / hhg_len * pi) ^ 2
# # hhg_windows_data = hhg_window_f.(eachindex(hhg_xy_t))
# # plot(hhg_windows_data)

# hhg_spectrum_x = fft(real.(hhg_xy_t) .* hhg_windows_data)
# hhg_spectrum_y = fft(imag.(hhg_xy_t) .* hhg_windows_data)
# hhg_spectrum = norm.(hhg_spectrum_x) .^ 2 + norm.(hhg_spectrum_y) .^ 2

# plot(log10.(hhg_spectrum)[1: 500])
# plot(log10.(norm.(hhg_spectrum_x))[1: 500])
# # plot(reverse(log10.(norm.(hhg_spectrum_x))[33112 - 499: 33112]))



# get average in new way.
# ave_kx_list = zeros(ComplexF64, length(k_linspace))
# ave_ky_list = zeros(ComplexF64, length(k_linspace))
# for (i, k) in enumerate(k_linspace)
#     for id1 = 1: pw.l_num ^ 2
#         l1 = pw.lmap[id1]
#         m1 = pw.mmap[id1]
#         for id2 = 1: pw.l_num ^ 2
#             l2 = pw.lmap[id2]
#             m2 = pw.mmap[id2]
#             co1 = sqrt(2pi / 3) * (get_SH_integral(l1, m1, 1, -1, l2, m2) - get_SH_integral(l1, m1, 1, 1, l2, m2))
#             co2 = im * sqrt(2pi / 3) * (get_SH_integral(l1, m1, 1, 1, l2, m2) + get_SH_integral(l1, m1, 1, -1, l2, m2))
#             ave_kx_list[i] += conj(δa_lm[id1][i]) * δa_lm[id2][i] * k * co1
#             ave_ky_list[i] += conj(δa_lm[id1][i]) * δa_lm[id2][i] * k * co2
#         end
#     end
#     println(i)
# end

# tmp = 0.0
# for (i, k) in enumerate(k_linspace)
#     for id = 1: pw.l_num ^ 2
#         tmp += norm(δa_lm[id][i]) .^ 2
#     end
# end


# ave_kx = sum(ave_kx_list) / tmp
# ave_ky = sum(ave_ky_list) / tmp
# tsurf_get_average_momentum_parallel(δa_lm, k_space, pw, k_min=0.0)

# rate1 = -0.0034015480209932275 / 0.00619663674209813
# rate2 = ave_kx / ave_ky