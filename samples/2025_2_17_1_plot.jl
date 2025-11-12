import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW


field_plt_list = []
hhg_plt_list = []
hhg_data_x_list = []
shg_yields = []
thz_data = []
tau_list = []
hhg_k_linspace = []
mid_Efs_record = []
tau_thz_mid = 0.0
shg_id = 0


# Basic Parameters
Nr = 5000
Δr = 0.2 / 2
l_num = 50
Δt = 0.05 / 2
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = rmax * 0.8
a0 = 0.5
po_func(r) = -1.57 * (r^2 + a0) ^ (-0.5) * flap_top_windows_f(r, 0, Ri_tsurf, 1/4, left_flag=false) * exp(-0.1 * (r^2 + a0) ^ (0.5))
absorb_func = absorb_boundary_r(rmax, Ri_tsurf)


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
E_fs = 0.0533 / 2.0
E_thz = 0.00002
E_dc = 0.00002
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
example_name = "2025_2_17_1_$(task_id)"
hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")

# HHG
p, hhg_data_x, hhg_data_y, base_id, hhg_k_linspace = get_hhg_spectrum_xy(hhg_integral_t, Et_datas[1], Et_datas[2], tau_fs, tmax, ω_fs, ts, Δt, max_display_rate=10)

# recording
shg_id = base_id * 2
tau_thz_mid = get_exactly_coincided_delay(ω_fs, tau_fs, nc, ω_thz)
push!(hhg_plt_list, p)
push!(hhg_data_x_list, hhg_data_x)
push!(shg_yields, norm.(hhg_data_x[shg_id]))
push!(field_plt_list, plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts))

Ex_thz_tmp, Ey_thz_tmp, Ez_thz_tmp, tmax_tmp = light_pulse(ω_thz, E_thz, 1, 0, pulse_shape="sin2", phase1=0.5pi)
At_datas_tmp, Et_datas_tmp, ts_tmp = create_tdata(tmax_tmp, 0, Δt, Ex_thz_tmp, no_light, no_light)
thz_data = [Et_datas_tmp[1], ts_tmp]

# At_THz_datas, Et_THz_datas, = create_tdata(tmax, 0, Δt, t -> E_applied(t), no_light, no_light, appendix_steps=1)
# push!(mid_Efs_record, Et_THz_datas[1][Int64(floor((tau_fs + nc * pi / ω_fs) ÷ Δt) + 1)])

Tp = 2 * nc * pi / ω_fs
Ex_thz_tmp, Ey_thz_tmp, Ez_thz_tmp, tmax_tmp = light_pulse(ω_thz, E_thz, 1, 0, pulse_shape="sin2", phase1=0.5pi)
At_datas_tmp, Et_datas_tmp, ts_tmp = create_tdata(tmax_tmp + Tp/2, -Tp/2, Δt, Ex_thz_tmp, no_light, no_light)
thz_data = [Et_datas_tmp[1], ts_tmp]

end

plot(hhg_k_linspace, norm.(hhg_data_x_list[4][1:length(hhg_k_linspace)]), yscale=:log10)
plot(hhg_k_linspace, norm.(hhg_data_x_list[10][1:length(hhg_k_linspace)]), yscale=:log10)
plot(tau_list, shg_yields)
plot(tau_list, mid_Efs_record)

unify(data) = (data .- minimum(data)) ./ (maximum(data) - minimum(data))
p2 = plot(unify(tau_list), unify(shg_yields))
plot!(p2, unify(thz_data[2]), unify(reverse(thz_data[1])))

hhg_plt_list[1]
p2


# unify(data) = (data .- minimum(data)) ./ (maximum(data) - minimum(data))
# p2 = plot(unify(tau_list), unify(shg_yields))
# plot!(p2, unify(thz_data[2]), unify(-thz_data[1]))


# f = open("saoyanchi_data.txt", "w+")
# for i = 1: length(tau_list)
#     write(f, "$(tau_list[i]) $(shg_yields[4])\n")
# end
# close(f)
