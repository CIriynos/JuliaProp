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
# 1. 测试1D-TDSE，研究E增大是否会出现相干探测
# 2. 研究格点对结果的影响
# 测试结果: ratio=1,2,3时，1与2差异明显，2与3几乎一致
# 结论：最好采用ratio=2的格点，此时几乎收敛（即使电场强度×2也收敛）
# 3. 研究THz ω的影响

# Define Basic Parameters
ratio = 2
Nx = 3600 * ratio
delta_x = 0.2 / ratio
delta_t = 0.05 / ratio
delta_t_itp = 0.1
Lx = Nx * delta_x
Xi = 200
a0 = 1.0
po_func(x) = -1.0 * (x^2 + a0) ^ (-0.5) * flap_top_windows_f(x, -Xi, Xi, 1/4) * exp(-0.1 * (x^2 + a0) ^ (0.5))
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

tau_id = 1
# for tau_id = 1: 5

# Define Laser.
ω1 = 0.057           # 800 nm (375 THz)
ω2 = ω1 / 20         # 16 μm (12.5 THz)
E0 = 0.057
E0_thz = 0.0002 * 0
E0_c = 0.01
nc = 12
Tp = 2 * nc * pi / ω1

# Define the tau (Delay)
induce_time = 0
tau_fs = induce_time
tau_lst = [tau_fs - 2pi/ω2, tau_fs + nc*pi/ω1 - 1.5pi/ω2,
    tau_fs + nc*pi/ω1 - pi/ω2, tau_fs + nc*pi/ω1 - 0.5pi/ω2, tau_fs + nc*2pi/ω1]
tau_thz = tau_lst[tau_id]

# Define the waveform
Et(t) = E0 * sin(ω1 * (t - tau_fs) / (2 * nc)) ^ 2 * cos(ω1 * (t - tau_fs)) * (t - tau_fs < Tp && t - tau_fs > 0)
E_thz(t) = E0_thz * sin(ω2 * (t - tau_thz)) * (t - tau_thz < 2pi / ω2 && t - tau_thz > 0)
E_thz_window(t) = flap_top_windows_f(t, 0, induce_time * 2, 1/2, right_flag = false)

T_total = tau_fs + Tp
steps = Int64(T_total ÷ delta_t) + 1
t_linspace = create_linspace(steps, delta_t)

Et_data_fs = Et.(t_linspace)
Et_data_thz = (E0_c .+ E_thz.(t_linspace)) .* flap_top_windows_f.(t_linspace, tau_fs, T_total, 1/2)

Et_data = Et_data_fs + Et_data_thz
At_data = -get_integral(Et_data, delta_t)

# store the figures.
fig1 = plot(At_data)
fig2 = plot([Et_data_fs (Et_data_thz) * 1])
push!(at_data_figs, fig1)
push!(et_data_figs, fig2)

# TDSE 1d
crt_wave = deepcopy(init_wave)
Xi_data, hhg_integral, energy_list = tdse_laser_fd1d_mainloop_penta(crt_wave, rt1d, pw1d, At_data, steps, Xi)


# # t-surf
# k_delta = 0.002
# kmin = -3.0
# kmax = 3.0
# k_linspace = kmin: k_delta: kmax
# Pk = tsurf_1d(pw1d, k_linspace, t_linspace, At_data, Xi, Xi_data)
# plot(Pk, yscale=:log10)


# HHG
start_id = Int64(floor(tau_fs ÷ delta_t)) + 1
end_id = Int64(floor((tau_fs + Tp) ÷ delta_t)) + 1
hhg_delta_k = 2pi / (end_id - start_id) / delta_t
hhg_k_linspace = [hhg_delta_k * i for i = 1: (end_id - start_id)]

hhg_t = -hhg_integral #- Et_data
hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(t_linspace, tau_fs, tau_fs + Tp)
hhg_spectrum = fft((hhg_t .* hhg_windows_data)[start_id: end_id])

max_hhg_id = Int64(floor(20 * ω1 / hhg_delta_k))
shg_id = Int64(floor(2 * ω1 / hhg_delta_k))
# plot(hhg_k_linspace[1: max_hhg_id] ./ ω1, norm.(hhg_spectrum)[1: max_hhg_id], yscale=:log10)

push!(part_of_hhg_data, norm.(hhg_spectrum)[1: max_hhg_id])
push!(shg_yield_record, norm.(hhg_spectrum)[shg_id])

println("shg_id = ", shg_id)
println("hhg_delta_k = ", hhg_delta_k)
println("tau_fs = ", tau_fs)

# end

second_hhg = [part_of_hhg_data[i][24] for i = 1: length(part_of_hhg_data)]
plot(second_hhg .^ 0.5,
    guidefont=Plots.font(14, "Times"),
    tickfont=Plots.font(14, "Times"),
    legendfont=Plots.font(14, "Times"),
    margin = 5 * Plots.mm,
    xlabel="Time delay τ(a.u.)",
    ylabel = "2nd Harmonic Yield(a.u.)",
    label = "THz",
    linewidth = 2.0
)

plot(part_of_hhg_data, yscale=:log10, ylimit=(1e-7, 1e3))


ll = (hhg_t .* hhg_windows_data)[start_id: end_id]
plot(norm.(fft(ll))[1:400], yscale=:log10)

plot(real.(hhg_t))



# Time-frequency Analysis
hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
t_num = length(hhg_t)
w_num = 100
ts = 1: t_num
plot(norm.(hhg_t))

ft_ana = zeros(ComplexF64, w_num, t_num)
delta = t_num / 50
j = 1
for i = 1: (delta / 10): t_num - delta
    ft_ana[:, j] = fft((real(hhg_t) .* hhg_windows_f.(ts, i, i + delta)))[1: w_num]
    j += 1
end

heatmap(log10.(norm.(ft_ana)[1: w_num, (1 + 0): (j - 0)] .^ 2), clim=(-4.5, 0))
heatmap((angle.(ft_ana)[1: w_num, (1): (j)]))


f = open("tmptmp4.txt", "w+")
for i = 1: length(hhg_t)
    write(f, "$(real(hhg_t[i]))\n")
end
close(f)