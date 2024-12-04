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


# Basic Parameters
Nr = 5000
Δr = 0.2
l_num = 100
Δt = 0.05
Z = 1.0
Ri_tsurf = 800.0
po_func_r = coulomb_potiential_zero_fixed_COS(600.0, 800.0)
rmax = Nr * Δr  # rmax = 1000.0
absorb_func = absorb_boundary_r(rmax, Ri_tsurf, pow_value=8.0, max_value=100.0)
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# Create Physical World and Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-9);
crt_shwave = deepcopy(init_wave_list[1]);
get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5



# Define Laser Here.
omega = 0.057             # λ = 800 nm
Ip = 0.5
E0 = 0.1
nc = 6
steps = Int64((2 * nc * pi / omega) ÷ Δt)
actual_duration = steps * Δt
t_linspace = create_linspace(steps, Δt)
gamma = omega * sqrt(2 * Ip) / E0

Ax(t) = (E0 / omega) * (sin(omega * t / 2.0 / nc) ^ 2) * sin(omega * (t - nc*pi/omega)) * (t < (2 * nc * pi / omega))

At_data_x = Ax.(t_linspace)
At_data_y = zeros(Float64, steps)
At_data_z = zeros(Float64, steps)
plot([At_data_x At_data_y])

Et_data_x = -get_derivative_two_order(At_data_x, Δt)
Et_data_y = -get_derivative_two_order(At_data_y, Δt)


# Define k Space
k_delta = 0.01
kmin = 0.01
kmax = 2.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))


# Start Propagation
# hhg_integral_t, phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);
# tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true)


# # Store Data
# example_name = "hhg_example_in_book"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
#     write(file, "hhg_integral_t", hhg_integral_t)
# end

# Retrieve Data.
example_name = "hhg_example_in_book"
crt_shwave = retrieve_obj(example_name, "crt_shwave")
phi_record = retrieve_obj(example_name, "phi_record")
dphi_record = retrieve_obj(example_name, "dphi_record")
hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")

hhg_spectrum = get_hhg_spectrum_xy(hhg_integral_t, Et_data_x, Et_data_y)
plot(log10.(hhg_spectrum)[1: 400])


hhg_xy_t = -hhg_integral_t .- (Et_data_x .+ im .* Et_data_y)
hhg_len = length(hhg_xy_t)

# two type of window function
# hhg_window_f(t) = sin(t / (hhg_len / 2) * pi) ^ 2 * (t < hhg_len/2 && t > 0)
# hhg_windows_data = hhg_window_f.(eachindex(hhg_xy_t) .- hhg_len/4)

hhg_window_f(t) = sin(t / hhg_len * pi) ^ 2
hhg_windows_data = hhg_window_f.(eachindex(hhg_xy_t))
# plot(hhg_windows_data)

hhg_spectrum_x = fft(real.(hhg_xy_t) .* hhg_windows_data)
hhg_spectrum_y = fft(imag.(hhg_xy_t) .* hhg_windows_data)
hhg_spectrum = norm.(hhg_spectrum_x) .^ 2 + norm.(hhg_spectrum_y) .^ 2


hhg_delta_k = 2pi / hhg_len / Δt
hhg_k_linspace = [hhg_delta_k * i for i = 1: hhg_len]

plot(hhg_k_linspace[1: 380] ./ omega, log10.(norm.(hhg_spectrum_x))[1: 380])
# plot(hhg_k_linspace[1: 380] ./ omega, log10.(norm.(hhg_spectrum))[1: 380])

# plot(log10.(norm.(hhg_spectrum_x))[hhg_len - 380: hhg_len])