import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# task_id = get_task_id_from_cmd_args()
task_id = 1


# Basic Parameters
Nr = 5000
Δr = 0.2 / 1
l_num = 60
Δt = 0.05 / 1
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = rmax * 0.8
a0 = 0.5
po_func(r) = -1.57 * (r^2 + a0) ^ (-0.5) * flap_top_windows_f(r, 0, Ri_tsurf, 1/4, left_flag=false) * exp(-0.1 * (r^2 + a0) ^ (0.5))
absorb_func = absorb_boundary_r(rmax, Ri_tsurf)

# Create Physical World & Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func)
rt = create_tdse_rt_sh(pw);
plot(-pw.po_data_r .+ 1e-50, yscale=:log10)

# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-8);
crt_shwave = deepcopy(init_wave_list[1]);
en = get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5
println("Energy of ground state: ", en)


# Define the Laser. 
E_fs = 0.0533
E_thz = 0.002 * 1
E_dc = 0.002 * 1
ω_fs = 0.05693
ω_thz = ω_fs / 30
nc = 12
tau_fs = 0
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz)
tau_thz = tau_list[task_id]

Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.0)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="sin2", phase1=0.5pi)
E_applied(t) = (Ex_thz(t) + E_dc) * flap_top_windows_f(t, 0, tmax, 1/2)
At_datas, Et_datas, ts, steps = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light, appendix_steps=1)
plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts)


# Propagation
crt_shwave = deepcopy(init_wave_list[1])
hhg_integral_t, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw, rt, At_datas[1], steps, Ri_tsurf);

kr_linspace = 0.01: 0.01: 1.0
k_space = create_k_space(kr_linspace, theta_linspace(180), fixed_phi(0))
a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_datas[1], At_datas[2], At_datas[3], Ri_tsurf, ts, k_space, TSURF_MODE_PL)
tsurf_plot_xz_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_min=0.01)

x_linspace, xproj_data_2 = tsurf_plot_xproj_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=false)
plot(xproj_data_2)

# f = open("xproj_data.txt", "w+")
# for i = 1: length(xproj_data)
#     write(f, "$(x_linspace[i]) $(xproj_data[i])\n")
# end
# close(f)


# # Store Data Manually
# example_name = "2025_3_15_1"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "hhg_integral_t", hhg_integral_t)
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
# end
