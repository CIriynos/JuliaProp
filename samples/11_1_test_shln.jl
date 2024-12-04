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
grid_ratio = 1
Nr = 5000 * grid_ratio
Δr = 0.2 / grid_ratio
l_num = 50
Δt = 0.05 / grid_ratio
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = 800.0
po_func_r_shit(r) = -1.3249 * (r^2 + 0.5) ^ (-0.5) * flap_top_windows_f(r, 0, Ri_tsurf, 1/4, left_flag=false)
absorb_func = absorb_boundary_r(rmax, Ri_tsurf, pow_value=8.0, max_value=100.0)

# Create Physical World and Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r_shit, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-8);
crt_shwave = deepcopy(init_wave_list[1]);
en = get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5
println("Energy of ground state: ", en)


# Define Laser Here.
omega = 0.057             # λ = 800 nm
Ip = 0.5
E0 = 0.0534               # 10^14 W⋅cm-2
nc = 10
steps = Int64((2 * nc * pi / omega) ÷ Δt)
actual_duration = steps * Δt
t_linspace = create_linspace(steps, Δt)
gamma = omega * sqrt(2 * Ip) / E0

Ez(t) = (E0) * flap_top_windows_f(t, 0, (2 * nc * pi / omega), 1/4) * sin(omega * (t - nc*pi/omega)) * (t < (2 * nc * pi / omega))

Et_data_x = zeros(Float64, steps)
Et_data_y = zeros(Float64, steps)
Et_data_z = Ez.(t_linspace)
plot([-Et_data_z])

At_data_x = -get_integral(Et_data_x, Δt)
At_data_y = -get_integral(Et_data_y, Δt)
At_data_z = -get_integral(Et_data_z, Δt)
plot([At_data_z])


# Define k Space
k_delta = 0.01
kmin = 0.01
kmax = 2.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, theta_linspace(180), fixed_phi(0))

# Start Propagation
hhg_integral_t, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw, rt, At_data_z, steps, Ri_tsurf; m = 0);

# # t-surf
# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_PL)
# fig = tsurf_plot_xz_momentum_spectrum_vector(a_tsurff_vec, k_space; kr_flag=false, kr_min=0.05)

# Store Data
example_name = "11_1_test_shln_optimized"
h5open("./data/$example_name.h5", "w") do file
    write(file, "crt_shwave", hcat(crt_shwave...))
    write(file, "phi_record", hcat(phi_record...))
    write(file, "dphi_record", hcat(dphi_record...))
    write(file, "hhg_integral_t", hhg_integral_t)
end


# Retrieve Data.
example_name = "11_1_test_shln_optimized"
crt_shwave = retrieve_obj(example_name, "crt_shwave")
phi_record = retrieve_obj(example_name, "phi_record")
dphi_record = retrieve_obj(example_name, "dphi_record")
hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")


# HHG for Linear Polarized 
Tp = 2 * nc * pi / omega
id_range = 1: length(t_linspace)
# println(id_range)
hhg_z_t = -hhg_integral_t .- (Et_data_z)
hhg_len = steps
hhg_delta_k = 2pi / hhg_len / Δt
hhg_k_linspace = [hhg_delta_k * i for i = 1: hhg_len]

#hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_f(t, tmin, tmax) = flap_top_windows_f(t, tmin + 200, tmax - 200, 1/2)
hhg_windows_data = hhg_windows_f.(t_linspace, 0, actual_duration)
# hhg_windows_data = ones(Float64, steps)

hhg_spectrum_z = fft(real.(hhg_z_t[id_range]) .* hhg_windows_data[id_range])
hhg_spectrum = norm.(hhg_spectrum_z) .^ 2

plot(hhg_k_linspace[1:350] ./ omega, hhg_spectrum[1: 350], yscale=:log10)