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
Nr = 20000
Δr = 0.2
l_num = 50
Δt = 0.05
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = rmax * 0.8
a0 = 0.5
po_func(r) = -1 / r * flap_top_windows_f(r, 0, Ri_tsurf, 1/4, left_flag=false)
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
E_thz = 0.00002 * 0
E_dc = 0.00002 * (task_id == 1 ? 1 : 0)
ω_fs = 0.05693
ω_thz = ω_fs / 25
nc = 15
tau_fs = 0
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz, samples_num=16)
tau_thz = tau_list[1]

Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.3, phase1=0.5pi, phase2=0.0)
Ey_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="sin2", phase1=0.5pi)
E_applied(t) = (Ey_thz(t) + E_dc) #* flap_top_windows_f(t, 0, tmax, 1/2)
At_datas, Et_datas, ts, steps = create_tdata(tmax, 0, Δt, t -> Ex_fs(t), t -> Ey_fs(t) + E_applied(t), no_light, appendix_steps=1)
plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts)

# define k space
k_delta = 0.002
kmin = 0.002
kmax = 1.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))

# # Propagation
# crt_shwave = deepcopy(init_wave_list[1])
# phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy_optimized(crt_shwave, pw, rt, At_datas[1], At_datas[2], steps, Ri_tsurf);

# # Store Data Manually
# example_name = "2025_10_13_$(task_id)"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
# end