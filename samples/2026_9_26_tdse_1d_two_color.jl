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
using LaTeXStrings
println("Number of Threads: $(Threads.nthreads())")


# Define Basic Parameters
ratio = 2
x_grid_expand_ratio = 2
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
ω1 = 0.057 * 0.6           # 800 nm (375 THz)
E0 = 0.04
E0_thz = 0.0005
nc = 15
Tp = 2 * nc * pi / ω1

# Define the waveform
Et_fs(t) = E0 * sin(ω1 * t / (2 * nc)) ^ 2 * cos(ω1 * t) * (t < Tp && t > 0)
Et_fs_2(t) = 0.2 * E0 * sin(ω1 * t / (2 * nc)) ^ 2 * cos((2 * ω1) * t) * (t < Tp && t > 0)
Et(t) = Et_fs(t) + Et_fs_2(t)
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

# function diag5(H; nev=20)
#     N = size(H, 1)

#     A = spdiagm(
#         -2 => H[3:N, 1],
#         -1 => H[2:N, 2],
#          0 => H[:, 3],
#          1 => H[1:N-1, 4],
#          2 => H[1:N-2, 5]
#     )

#     E, V = eigs(
#         Hermitian(A);
#         nev = nev,
#         which = :SR,
#         maxiter = 10000,
#         ncv = max(40, 4 * nev),
#         tol = 1e-8
#     )

#     p = sortperm(real.(E))
#     return real.(E[p]), V[:, p]
# end

# E, V = diag5(rt.H_penta, nev=4)
# bound_states = collect(eachcol(V[:, E .< 0]))
# println("Bound states energies: ", E[E .< 0])

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

# # Store Data Manually
# example_name = "2026_9_26_tdse_1d_$(nc)_$(E0)_$(ω1)_$(a0)"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "pos_expect", pos_expect)
#     write(file, "pos_expect_bb", pos_expect_bb)
#     write(file, "pos_expect_cc", pos_expect_cc)
#     write(file, "pos_expect_bc", pos_expect_bc)
#     write(file, "ionization_rate", ionization_rate)
#     write(file, "total_norm", total_norm)
# end


# HHG
hhg_spectrum, hhg_spectrum_angle, ks = get_hg_spectrum_from_dipole(t_linspace, pos_expect, 100 * ω1)
hhg_spectrum_bb, hhg_spectrum_bb_angle, ks = get_hg_spectrum_from_dipole(t_linspace, pos_expect_bb, 100 * ω1)
hhg_spectrum_bc, hhg_spectrum_bc_angle, ks = get_hg_spectrum_from_dipole(t_linspace, pos_expect_bc, 100 * ω1)
hhg_spectrum_cc, hhg_spectrum_cc_angle, ks = get_hg_spectrum_from_dipole(t_linspace, pos_expect_cc, 100 * ω1)
hhg_delta_k = ks[2] - ks[1]

max_hhg_id = Int64(floor(6 * ω1 / hhg_delta_k))
fhg_id = Int64(floor(ω1 / hhg_delta_k)) + 2
shg_id = fhg_id * 2
thg_id = fhg_id * 3 - 2
# plot(ks[1: max_hhg_id] ./ ω1, norm.(hhg_spectrum)[1: max_hhg_id], yscale=:log10, ylimits=(1e-10, 1e5))

pp = plot(ks[1: max_hhg_id] ./ ω1, 
    [hhg_spectrum[1: max_hhg_id] hhg_spectrum_bb[1: max_hhg_id] hhg_spectrum_bc[1: max_hhg_id] hhg_spectrum_cc[1: max_hhg_id]],
    yscale=:log10, ylimits=(1e-10, 1e0), 
    # labels=[L"\langle d \rangle" L"\langle d \rangle_{bb} " L"\langle d \rangle_{bc} " L"\langle d \rangle_{cc}"],
    labels=[L"G(\omega)" L"G_{bb}(\omega) " L"G_{bc}(\omega) " L"G_{cc}(\omega)"],
    xlabel="Harmonic Order", ylabel="Yield (arb. unit)",
    titlefontsize=12, guidefontsize=12, tickfontsize=12, legendfontsize=14,
    linewidth=1.5, alpha=[1.0, 0.75, 0.75, 0.75],
    linecolor=[:black :red :green :blue], linestyle=[:solid :dash :dash :dash],
    xticks=[1, 3, 5, 7, 9], yticks=[1e-6, 1e-2, 1e2],
    legend=:none, grid=true, size=(600 / 2, 400 / 1.5), framestyle = :semi)

# p3 = plot(hhg_k_linspace[1: max_hhg_id] ./ ω1, 
#     [hhg_k_linspace[1: max_hhg_id] .^ 3 .* abs2.(hhg_spectrum)[1: max_hhg_id]  hhg_k_linspace[1: max_hhg_id] .^ 3 .* abs2.(hhg_spectrum_bb)[1: max_hhg_id]  hhg_k_linspace[1: max_hhg_id] .^ 3 .* abs2.(hhg_spectrum_bc)[1: max_hhg_id]  hhg_k_linspace[1: max_hhg_id] .^ 3 .* abs2.(hhg_spectrum_cc)[1: max_hhg_id]],
#     yscale=:log10, ylimits=(1e-8, 1e4), labels=[L"\langle d \rangle" L"\langle d \rangle_{bb} " L"\langle d \rangle_{bc} " L"\langle d \rangle_{cc}"],
#     xlabel="Harmonic Order", ylabel="Yield (arb. unit)",
#     titlefontsize=14, guidefontsize=12, tickfontsize=12, legendfontsize=12,
#     linewidth=1.5, alpha=[1.0, 0.75, 0.75, 0.75],
#     linecolor=[:black :red :blue :green], linestyle=[:solid :dash :dash :dash],
#     xticks=[1, 3, 5, 7, 9], yticks=[1e-6, 1e-2, 1e2],
#     legend=:none, grid=true, size=(600 / 2, 400 / 1.5))


