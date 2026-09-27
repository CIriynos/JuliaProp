import Pkg
Pkg.activate(".")

using Revise
using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using Printf
using FFTW
using SparseArrays
using Arpack
println("Number of Threads: $(Threads.nthreads())")

task_id = get_task_id_from_cmd_args()

E0_list = 0.01: 0.0025: 0.12
omega_ratio_list = 0.1: 0.05: 2.0
omega_ratio = omega_ratio_list[task_id]

for k in eachindex(E0_list)

# Define Basic Parameters
ratio = 2
x_grid_expand_ratio = 6
Nx = 3600 * ratio * x_grid_expand_ratio
delta_x = 0.2 / ratio
delta_t = 0.05 / ratio
delta_t_itp = 0.1
Lx = Nx * delta_x
Xi = Lx / 2 * 0.8
a0 = 2.0
po_func(x) = -1.0 * (x^2 + a0) ^ (-0.5) * exp(-x ^ 2 / (5.0 ^ 2))
imb_func(x) = -100im * ((abs(x) - Xi) / (Lx / 2 - Xi)) ^ 8 * (abs(x) > Xi)

# Create Physics World & Runtime
pw = create_physics_world_1d(Nx, delta_x, delta_t, po_func, imb_func, delta_t_im=delta_t_itp)
rt = create_tdse_rt_1d(pw)

# Get Initial Wave
x_linspace = get_linspace(pw.xgrid)
seed_wave = gauss_package_1d(x_linspace, 1.0, 1.0, 0.0)
init_wave = itp_fd1d(seed_wave, rt, min_error = 1e-10)
get_energy_1d(init_wave, rt)

# Define Laser.
ω1 = 0.057 * omega_ratio
E0 = E0_list[k]
E0_thz = 0.0005
nc = 8
Tp = 2 * nc * pi / ω1

# Define the waveform
Et_fs(t) = E0 * sin(ω1 * t / (2 * nc)) ^ 2 * cos(ω1 * t) * (t < Tp && t > 0)
# Et_fs_2(t) = 0.2 * E0 * sin(ω1 * t / (2 * nc)) ^ 2 * cos((2 * ω1) * t) * (t < Tp && t > 0)
Et(t) = Et_fs(t)
# Et(t) = Et_fs(t) + E0_thz

T_total = Tp
steps = Int64(T_total ÷ delta_t)
t_linspace = create_linspace(steps, delta_t)

Et_data = Et.(t_linspace)
At_data = -get_integral(Et_data, delta_t)
plot(t_linspace, Et_data, xlabel="Time (a.u.)", ylabel="Electric Field (a.u.)", title="Laser Electric Field", legend=false)

# # TDSE 1d
# crt_wave = deepcopy(init_wave)
# Xi_data, hhg_integral, energy_list = tdse_laser_fd1d_mainloop_penta(crt_wave, rt, pw, At_data, steps, Xi)


# # t-surf
# k_delta = 0.002
# kmin = -3.0
# kmax = 3.0
# k_linspace = kmin: k_delta: kmax
# Pk = tsurf_1d(pw, k_linspace, t_linspace, At_data, Xi, Xi_data)
# plot(Pk, yscale=:log10)

# # HHG
# hhg_delta_k = 2pi / steps / delta_t
# hhg_k_linspace = [hhg_delta_k * i for i = 1: steps]

# hhg_t = -hhg_integral - Et_data
# hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
# hhg_windows_data = hhg_windows_f.(t_linspace, tau_fs, tau_fs + Tp)
# hhg_spectrum = fft(hhg_t .* hhg_windows_data)

# max_hhg_id = Int64(floor(50 * ω1 / hhg_delta_k))
# shg_id = Int64(floor(2 * ω1 / hhg_delta_k))
# plot(hhg_k_linspace[1: max_hhg_id] ./ ω1, norm.(hhg_spectrum)[1: max_hhg_id], yscale=:log10, ylimits=(1e-10, 1e3))

####

E, V = get_bound_states_1d(pw, [-0.47526, -0.15756, -0.02853], 1000)
bound_states = V

pos_expect = zeros(ComplexF64, steps)
pos_expect_bb = zeros(ComplexF64, steps)
pos_expect_cc = zeros(ComplexF64, steps)
pos_expect_bc = zeros(ComplexF64, steps)
ionization_rate = zeros(ComplexF64, steps)
total_norm = zeros(ComplexF64, steps)

crt_wave = deepcopy(init_wave)
for it = 1: steps
    @fastmath @. rt.tmp_penta_1 = rt.A_pos_penta - 0.5 * pw.delta_t * At_data[it] * rt.D1_penta
    @fastmath @. rt.tmp_penta_2 = rt.A_neg_penta + 0.5 * pw.delta_t * At_data[it] * rt.D1_penta

    penta_mul(rt.half_wave, rt.tmp_penta_1, crt_wave)
    pentamat_elimination(crt_wave, rt.tmp_penta_2, rt.half_wave, rt.A_buffer, rt.B_buffer)

    crt_wave_L = gauge_transform_V2L(crt_wave, At_data[1:it], pw.delta_t, x_linspace)

    crt_wave_b = sum(dot(bound_states[i], crt_wave_L) * bound_states[i] for i in 1:length(bound_states))
    crt_wave_c = crt_wave_L .- crt_wave_b

    # get pos expect
    pos_expect[it] = dot(crt_wave_L, x_linspace .* crt_wave_L)
    pos_expect_bb[it] = dot(crt_wave_b, x_linspace .* crt_wave_b)
    pos_expect_cc[it] = dot(crt_wave_c, x_linspace .* crt_wave_c)
    pos_expect_bc[it] = dot(crt_wave_b, x_linspace .* crt_wave_c) + dot(crt_wave_c, x_linspace .* crt_wave_b)
    ionization_rate[it] = norm(crt_wave_b)^2
    total_norm[it] = norm(crt_wave_L)^2

    if it % 200 == 0
        println("[FD_1d]: step $it / $steps, norm = $(norm(crt_wave_L)^2), ionization = $(norm(crt_wave_b)^2)")
    end
end

# plot(t_linspace, [real.(pos_expect) real.(pos_expect_bb) real.(pos_expect_cc) real.(pos_expect_bc)], labels=["total" "bb" "cc" "bc"])

# Store Data Manually
example_name = "2026_9_27_tdse_1d_total_$(nc)_$(E0)_$(ω1)_$(a0)"
h5open("./data/$example_name.h5", "w") do file
    write(file, "pos_expect", pos_expect)
    write(file, "pos_expect_bb", pos_expect_bb)
    write(file, "pos_expect_cc", pos_expect_cc)
    write(file, "pos_expect_bc", pos_expect_bc)
    write(file, "ionization_rate", ionization_rate)
    write(file, "total_norm", total_norm)
end

end