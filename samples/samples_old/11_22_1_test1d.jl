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

# 2024-11-22-1

# Define Basic Parameters
ratio = 1
Nx = 20000 * ratio
delta_x = 0.2 / ratio
delta_t = 0.05 / ratio
delta_t_itp = 0.1
Lx = Nx * delta_x
Xi = 1800
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


# Mainloop for Sweeping
at_data_figs = []
et_data_figs = []
part_of_hhg_data = []
shg_yield_record = []
smooth_records = []
# tau_list = [-500, 100, 285, 470, 1000]
tau_list = [-1500, -650, -270, 100, 800]
tau_var = -1500

induce_time = 0
thz_rate = 1 / 5
fs_rate = 1
tau = tau_var
# tau_fs = induce_time + (tau < 0.0) * abs(tau)
# tau_thz = induce_time + (tau >= 0.0) * abs(tau)

# Define Laser.
tau_fs = induce_time + 0  # fixed
tau_thz = induce_time + tau + 0
omega = 0.057           # 800 nm (375 THz)
omega_thz = 0.05omega    # 16 μm (18.75 THz)
E0 = 0.057 * fs_rate
E0_thz = 0.0075 * thz_rate * 0
E0_c = 0.005 * thz_rate * 0
nc = 15
Tp = 2 * nc * pi / omega
Et(t) = E0 * sin(omega * (t - tau_fs) / (2 * nc)) ^ 2 * cos(omega * (t - tau_fs)) * (t - tau_fs < Tp && t - tau_fs > 0)
E_thz(t) = E0_thz * sin(omega_thz * (t - tau_thz) / 2) ^ 2 * sin(omega_thz * (t - tau_thz)) * (t - tau_thz < 2pi / omega_thz && t - tau_thz > 0)

T_total = tau_fs + Tp #max(tau_fs + Tp, tau_thz + 2pi/omega_thz)
steps = Int64(T_total ÷ delta_t)
t_linspace = create_linspace(steps, delta_t)

Et_data_fs = Et.(t_linspace)
Et_data_thz = E_thz.(t_linspace) .* flap_top_windows_f.(t_linspace, 0, 2*500, 1/2, right_flag = false)
Et_data_c = E0_c * flap_top_windows_f.(t_linspace, 0, 1.5 * induce_time, 1/2, right_flag = false)

Et_data = Et_data_fs + Et_data_thz + Et_data_c
At_data = -get_integral(Et_data, delta_t)

# store the figures.
fig1 = plot(At_data)
fig2 = plot([Et_data_fs (Et_data_thz + Et_data_c) * 10])
push!(at_data_figs, fig1)
push!(et_data_figs, fig2)

# # TDSE 1d
# crt_wave = deepcopy(init_wave)
# Xi_data, hhg_integral, energy_list, smooth_record = tdse_laser_fd1d_mainloop_penta(crt_wave, rt1d, pw1d, At_data, steps, Xi)
# push!(smooth_records, smooth_record)
 
# Find tn satisfying A(tn) = 0
at_zeros_ids = []
n_list = 0: 1: Int64(floor((4*nc - 1) / 2))
tn_list = @. (2 * n_list + 1) * pi / (2 * omega)
tn_ids = Int64.(tn_list .÷ delta_t)
for i = 1: length(tn_ids) - 1
    push!(at_zeros_ids, tn_ids[i] + sortperm(abs.(At_data[tn_ids[i]: tn_ids[i+1]]))[1] - 1)
end
at_map = zeros(steps)   # Check it.
for i = 1: length(at_zeros_ids)
    at_map[at_zeros_ids[i]] += 1
end
plot([At_data at_map])
sum(abs.(At_data[at_zeros_ids .- 0]))
at_zeros_ids = [1; at_zeros_ids; steps]

# Special TDSE 1d (piece-wise) with WOM
gamma = 0.001
E_min = -0.7
E_max = 2.0
E_delta = 2 * gamma
Ev_list = E_min: 2*gamma: E_max

# Plist_total_cc = []
# Plist_cc = []
# crt_wave = deepcopy(init_wave)
# Plist_total, Plist = windows_operator_method_1d(crt_wave, gamma, 3, Ev_list, rt1d, pw1d)
# push!(Plist_total_cc, Plist_total)
# push!(Plist_cc, Plist)

# for i = 2: length(at_zeros_ids)
#     crt_step = at_zeros_ids[i] - at_zeros_ids[i - 1]
#     At_piece = At_data[at_zeros_ids[i - 1]: at_zeros_ids[i] - 1]
#     tdse_laser_fd1d_mainloop_penta(crt_wave, rt1d, pw1d, At_piece, crt_step, Xi)
#     println("finishing TDSE i=$(i)")

#     Plist_total, Plist = windows_operator_method_1d(crt_wave, gamma, 3, Ev_list, rt1d, pw1d)
#     push!(Plist_total_cc, Plist_total)
#     push!(Plist_cc, Plist)
#     println("finishing WOM i=$(i)")
# end

# # Store Data
# example_name = "11_22_1_test1d"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "init_wave", init_wave)
#     write(file, "crt_wave", crt_wave)
#     write(file, "Plist_total_cc", hcat(Plist_total_cc...))
#     write(file, "Plist_cc", hcat(Plist_cc...))
# end

# Retrieve Data.
example_name = "11_22_1_test1d"
Plist_total_cc = retrieve_obj(example_name, "Plist_total_cc")
Plist_cc = retrieve_obj(example_name, "Plist_cc")
crt_wave = retrieve_mat(example_name, "crt_wave")

# Analyse
plot(Ev_list, Plist_cc[31], yscale=:log10, ylims=(1e-10, 1))
id0 = Int64((-0.10 - E_min) ÷ E_delta)
id1 = Int64((-0.01 - E_min) ÷ E_delta)
id2 = Int64((0.05 - E_min) ÷ E_delta)
id3 = Int64((2.0 - E_min) ÷ E_delta)

ionz_rate_list = [sum(Plist_cc[i][id2: id3]) / sum(Plist_cc[i]) for i = 1: length(at_zeros_ids)]
excited_rate_list = [sum(Plist_cc[i][id0: id1]) / sum(Plist_cc[i]) for i = 1: length(at_zeros_ids)]
ionz_rate_diff = zeros(length(ionz_rate_list))
excited_rate_diff = zeros(length(excited_rate_list))
for i = 2: length(ionz_rate_list)
    ionz_rate_diff[i] = (ionz_rate_list[i] - ionz_rate_list[i-1])
    excited_rate_diff[i] = excited_rate_list[i] - excited_rate_list[i-1]
end
plot(at_zeros_ids, [ionz_rate_list ionz_rate_diff])


ff = plot(1: steps, abs.(At_data))
plot!(ff, at_zeros_ids, ionz_rate_diff .* 1e3)
plot!(ff, at_zeros_ids, excited_rate_list .* 1e3)
mid = steps ÷ 2
plot!(ff, [mid, mid], [-0.05, 1.05], ls=:dashdot, label="mid", linewidth=1.5)
sum(ionz_rate_diff[1:16])
sum(ionz_rate_diff[16:31])

tmptmp = sortperm(Plist_cc[31][1: id1 - 10], rev=true)
plot(Ev_list, Plist_cc[31], yscale=:log10, ylims=(1e-10, 1))
ff

# Anaylse 2
id_zero = Int64((0.0 - E_min) ÷ E_delta)
range1 = length(Ev_list) + id_zero: length(Ev_list) * 2 - 1
range2 = 1: length(Ev_list) - id_zero
plot([Plist_total_cc[31][range1] reverse(Plist_total_cc[31][range2])], yscale=:log10, ylims=(1e-10, 1))



# plt = plot()
# gf = @gif for i = 1: length(ionz_rate_list)
#     for j = 1: 5
#         plot(Ev_list, Plist_cc[i], yscale=:log10, ylims=(1e-10, 1))
#     end
# end every 1


# # WOM
# gamma = 0.001
# E_min = -0.7
# E_max = 2.0
# Ev_list = E_min: 2*gamma: E_max
# Plist_total, Plist = windows_operator_method_1d(crt_wave, k_delta / 2, 3, Ev_list, rt1d, pw1d)
# plot(Plist_total, yscale=:log10)


# # t-surf
# k_delta = 0.002
# kmin = -2.0
# kmax = 2.0
# # k_linspace = kmin: k_delta: kmax
# k_linspace = [-sqrt.(2 .* (2.0: -0.002: 0.002)); sqrt.(2 .* (0.002: 0.002: 2.0))]
# Pk = tsurf_1d(pw1d, k_linspace, t_linspace, At_data, Xi, Xi_data)
# plot(k_linspace, Pk, yscale=:log10)
# plot([[Pk[1:999]; Pk[1001:2000]], Plist_total * 1.2e2], yscale=:log10)


# # HHG
# hhg_start_id = Int64(tau_fs ÷ delta_t)
# hhg_end_id = Int64((tau_fs + Tp) ÷ delta_t)
# hhg_len = hhg_end_id - hhg_start_id + 1
# hhg_delta_k = 2pi / hhg_len / delta_t
# hhg_k_linspace = [hhg_delta_k * i for i = 1: hhg_len]

# hhg_t = (hhg_integral - Et_data)[hhg_start_id: hhg_end_id]
# hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
# hhg_windows_data = hhg_windows_f.(t_linspace[hhg_start_id: hhg_end_id], tau_fs, tau_fs + Tp)
# hhg_spectrum = fft(hhg_t .* hhg_windows_data)

# max_hhg_id = Int64(floor(5 * omega / hhg_delta_k))
# shg_id = Int64(floor(2 * omega / hhg_delta_k))
# # plot(hhg_k_linspace[1: max_hhg_id] ./ omega, norm.(hhg_spectrum)[1: max_hhg_id], yscale=:log10)

# push!(part_of_hhg_data, norm.(hhg_spectrum)[1: max_hhg_id])
# push!(shg_yield_record, norm.(hhg_spectrum)[shg_id])
# println("shg_id = ", shg_id)
# println("hhg_len = ", hhg_len)
# println("hhg_delta_k = ", hhg_delta_k)

# second_hhg = [part_of_hhg_data[i][31] for i = 1: length(part_of_hhg_data)]
# plot(tau_list, second_hhg .^ 0.5,
#     guidefont=Plots.font(14, "Times"),
#     tickfont=Plots.font(14, "Times"),
#     legendfont=Plots.font(14, "Times"),
#     margin = 5 * Plots.mm,
#     xlabel="Time delay τ(a.u.)",
#     ylabel = "2nd Harmonic Yield(a.u.)",
#     label = "THz",
#     linewidth = 2.0
# )

# plot(part_of_hhg_data, yscale=:log10)