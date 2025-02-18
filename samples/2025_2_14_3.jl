import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# 2025/2/14 3
# 彻底延长时间box，使得THz能够完整参与演化 (后半段可以cut掉,因为不重要)
# 采用 ω_thz = ω_fs / 25


task_id = get_task_id_from_cmd_args()
# task_id = 1


# Basic Parameters
Nr = 5000
Δr = 0.2
l_num = 50
Δt = 0.05
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = rmax * 0.8
po_func_r = coulomb_potiential_zero_fixed_windows(Ri_tsurf)
absorb_func = absorb_boundary_r(rmax, Ri_tsurf)

# Create Physical World & Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-8);
crt_shwave = deepcopy(init_wave_list[1]);
en = get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5
println("Energy of ground state: ", en)

# Define the Laser. 
E_fs = 0.0533
E_thz = 0.00002
E_dc = 0.00002
ω_fs = 0.05693
ω_thz = ω_fs / 25
nc = 12
tau_fs = 3000
tau_list = get_1c_thz_delay_list_ok(ω_fs, 3000, nc, ω_thz)
tau_thz = tau_list[task_id]

Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.5, phase1=0.5pi, phase2=0)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="none", phase1=0.5pi)
Ex_window = dc_bias(1.0, 0, max(0, minimum(tau_list)), tmax, tmax * 2)
E_applied(t) = (Ex_thz(t) + E_dc) * Ex_window(t)
At_datas, Et_datas, ts, steps = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light, appendix_steps=500)
plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts)

# Propagation
crt_shwave = deepcopy(init_wave_list[1])
hhg_integral_t_1, hhg_integral_t_2, hhg_integral_t_3 = tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw, rt, At_datas[1], At_datas[2], steps, Ri_tsurf);
# hhg_integral_t_1, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw, rt, At_data_x, steps, Ri_tsurf);


# Store Data Manually
example_name = "2025_2_14_3_$(task_id)"
h5open("./data/$example_name.h5", "w") do file
    write(file, "crt_shwave", hcat(crt_shwave...))
    write(file, "hhg_integral_t_1", hhg_integral_t_1)
    write(file, "hhg_integral_t_2", hhg_integral_t_2)
    write(file, "hhg_integral_t_3", hhg_integral_t_3)
end