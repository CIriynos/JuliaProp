import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# Basic Parameters
Nr = 5000
Δr = 0.2
l_num = 60
Δt = 0.05
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = rmax * 0.8
po_func_r = coulomb_potiential_zero_fixed_windows(Ri_tsurf)
absorb_func = absorb_boundary_r(rmax, Ri_tsurf)


field_plt_list = []
hhg_plt_list = []
hhg_data_x_list = []
shg_yields = []
tau_list = []
tau_thz_mid = 0.0

for task_id = 1: 16

# # Create Physical World & Runtime
# pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
# rt = create_tdse_rt_sh(pw);

# # Initial Wave
# init_wave_list = itp_fdsh(pw, rt, err=1e-8);
# crt_shwave = deepcopy(init_wave_list[1]);
# en = get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5
# println("Energy of ground state: ", en)

# Define the Laser. 
E_fs = 0.0533
E_thz = 0.00002
E_dc = 0.00002
ω_fs = 0.05693
ω_thz = ω_fs / 25
nc = 15
tau_fs = 200
tau_list = get_1c_thz_delay_list(ω_fs, tau_fs, nc, ω_thz)
tau_thz = tau_list[task_id]

Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.5)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="none", phase1=0.5pi)
Ex_window = dc_bias(1.0, 0, tau_fs, tmax, tmax * 2)

# Create Time Linspace & Data
steps = Int64((tmax + 100) ÷ Δt) + 1
ts = create_linspace(steps, Δt)
E_applied(t) = (Ex_thz(t) + E_dc) * Ex_window(t)
Et_data_x = Ex_fs.(ts) .+ E_applied.(ts)
Et_data_y = Ey_fs.(ts)
Et_data_z = no_light.(ts)
At_data_x = -get_integral(Et_data_x, Δt)
At_data_y = -get_integral(Et_data_y, Δt)
At_data_z = -get_integral(Et_data_z, Δt)

p1 = plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts)
push!(field_plt_list, p1)

# # Propagation
# crt_shwave = deepcopy(init_wave_list[1])
# hhg_integral_t_1, hhg_integral_t_2, hhg_integral_t_3 = tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);
# # hhg_integral_t_1, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw, rt, At_data_x, steps, Ri_tsurf);

# # Store Data Manually
# example_name = "2025_1_17_test_$(task_id)"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "hhg_integral_t_1", hhg_integral_t_1)
#     write(file, "hhg_integral_t_2", hhg_integral_t_2)
#     write(file, "hhg_integral_t_3", hhg_integral_t_3)
# end

# Retrieve Data.
example_name = "2025_1_17_test_$(task_id)"
hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t_1")

# HHG
p, hhg_data_x, hhg_data_y, base_id = get_hhg_spectrum_xy(hhg_integral_t, Et_data_x, Et_data_y, tau_fs, tmax, ω_fs, ts, Δt, max_display_rate=10)

push!(hhg_plt_list, p)
push!(hhg_data_x_list, hhg_data_x)
push!(shg_yields, norm.(hhg_data_x[base_id * 2 + 3]))

tau_thz_mid = get_exactly_coincided_delay(ω_fs, tau_fs, nc, ω_thz)

end

hhg_plt_list[12]
plot(tau_list, shg_yields)