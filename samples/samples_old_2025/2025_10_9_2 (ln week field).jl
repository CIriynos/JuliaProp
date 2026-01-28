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
Δr = 0.2 / 2
l_num = 50
Δt = 0.05 / 2
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = rmax * 0.8
po_func(r) = -1.0 / r
absorb_func = absorb_boundary_r(rmax, Ri_tsurf)

# Create Physical World & Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func)
rt = create_tdse_rt_sh(pw);
plot(-pw.po_data_r .+ 1e-50, yscale=:log10)

# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-20);
crt_shwave = deepcopy(init_wave_list[1]);
en = get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5
println("Energy of ground state: ", en)

# Define the Laser. 
E_fs = 0.0533 * 0.1
E_thz = 0.00002 * 0
E_dc = 0.00002 * 0
ω_fs = 0.05693
ω_thz = ω_fs / 30
nc = 12
tau_fs = 0
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz)
tau_thz = tau_list[1]

Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.0)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="sin2", phase1=0.5pi)
E_applied(t) = (Ex_thz(t) + E_dc) * flap_top_windows_f(t, 0, tmax, 1/2)
At_datas, Et_datas, ts, steps = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light, appendix_steps=1)
plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts)


# Propagation
crt_shwave = deepcopy(init_wave_list[1])
hhg_integral_t, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw, rt, At_datas[1], steps, Ri_tsurf);


# Store Data Manually
example_name = "2025_10_9_2_$(task_id)"
h5open("./data/$example_name.h5", "w") do file
    write(file, "crt_shwave", hcat(crt_shwave...))
    write(file, "hhg_integral_t", hhg_integral_t)
end