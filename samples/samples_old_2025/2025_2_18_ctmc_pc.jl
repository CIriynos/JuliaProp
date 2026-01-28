import Pkg
Pkg.activate(".")

using Revise
using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using Printf
using FFTW
using DSP


ap1s = []
ap2s = []
ap3s = []
tau_list = []
hhg_plt_list = []
shg_yields = []
fs_thz_fig_list = []
pmd_fig_list = []
py_ave_list = Float64[]
actual_At_list = Float64[]
pdd_list = []

# Define the function of Movement
F1(E, Z, x, y, z, rj) = -E - Z * rj / (x^2 + y^2 + z^2 + 0.0 + 1e-5) ^ (3 / 2)

# Core Parameters
Z = 1.0
Ip = 0.5

# CTMC Parameters
Δt = 0.2
filter_threshold = 2.0
p_min = -1.0
p_max = 1.0
p_delta = 0.005

# Define the Laser. 
E_fs = 0.0533
E_thz = 0.001
E_dc = 0.0
ω_fs = 0.05693
ω_thz = ω_fs / 10
nc = 12
tau_fs = 2000
nc_thz = 2.5
cep = 0.5pi

# create runtime for CTMC
trajs_num = Int64((2 * nc * pi / ω_fs) ÷ Δt) + 1
tmax = last(get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz)) + 2 * nc_thz * pi / ω_thz
t_num = Int64(tmax ÷ Δt) + 1
rt = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)

# task_id = 1
tasks_num = 16
for task_id = [1, 11]

# t-data for Field
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz, samples_num=tasks_num, nc_thz=2)
tau_thz = tau_list[task_id]

Ex_fs, Ey_fs, Ez_fs = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.3, phase1=0.5pi+cep, phase2=0+cep)
mid_point = tau_fs + nc*pi/ω_fs - pi/ω_thz
E_thz_1, E_thz_2, E_thz_3, tmax_thz = light_pulse(ω_thz, E_thz, nc_thz, tau_thz, pulse_shape="sin2", phase1=0.5pi)
E_window = dc_bias(1.0, 0, tau_fs, tmax, tmax * 2)
E_applied(t) = (E_thz_1(t)) #* E_window(t)

At_data_xyz, Et_data, ts, = create_tdata(tmax, 0, Δt, Ex_fs, t -> Ey_fs(t) + E_applied(t), no_light)
At_data_xyz_hf, Et_data_hf, ts_hf, = create_tdata(tmax, Δt/2, Δt, Ex_fs, t -> Ey_fs(t) + E_applied(t), no_light)
At_data_xyz_no_thz, Et_data_no_thz, = create_tdata(tmax, 0, Δt, Ex_fs, t -> Ey_fs(t), no_light)
At_data_xyz_hf_no_thz, Et_data_hf_no_thz, = create_tdata(tmax, Δt/2, Δt, Ex_fs, t -> Ey_fs(t), no_light)
fs_thz_fig = plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts, thz_ratio=40)

plot(sqrt.(Et_data[1] .^ 2 + Et_data[2] .^ 2 + Et_data[3] .^ 2))


pdd_tmp = Matrix{Float64}
for m = 1: 5
# m = 1
    pv_max = 0.5
    # tid_cc, pv_cc, theta_cc = generate_start_point_uniform_special(trajs_num, m, shift=Int64(tau_fs ÷ Δt + 1))
    tid_cc, pv_cc, theta_cc = generate_start_point_random(trajs_num, t_num, pv_max, Et_data, Ip, Z)
    update_start_point(rt, trajs_num, tid_cc, pv_cc, theta_cc, Et_data, Ip, Z, Δt)

    # Mainloop for CTMC
    traj_filter_flag = ctmc_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, filter_threshold)
    println((length(traj_filter_flag), sum(traj_filter_flag)))

    # Add data to PMD/HHG
    add_to_pmd(rt, trajs_num, t_num, Z, traj_filter_flag)
    add_to_hhg(rt, trajs_num, t_num, Δt, tid_cc, traj_filter_flag)

# # Trajs Analyse
# ap1, ap2, ap3, shg_yield_data = trajs_analyse(rt, Et_data, trajs_num, tmax, t_num, ts, Δt, ω_fs, tau_fs, tid_cc, traj_filter_flag, nc)
# push!(ap1s, ap1)
# push!(ap2s, ap2)
# push!(ap3s, ap3)

# Special PDD Analyse
if m == 1
    pdd_tmp = trajs_pdd_analyse(rt, tid_cc, traj_filter_flag)
else
    pdd_tmp .+= trajs_pdd_analyse(rt, tid_cc, traj_filter_flag)
end
end
push!(pdd_list, pdd_tmp)

pdd_list_tmp = []

# HHG
p1, p2, hhg_spectrum_x, hhg_spectrum_y, base_id, hhg_k_linspace = ctmc_get_hhg_spectrum(rt, Et_data, tmax, t_num, ts, Δt, ω_fs, tau_fs)

# PMD
p_axis = p_min: p_delta: p_max
p3 = heatmap(p_axis, p_axis, rt.pxy_final, color=:jet1, size=(500, 420))
px_ave, py_ave, pz_ave = get_average_p_ctmc(rt)


# Record
push!(hhg_plt_list, p2)
push!(shg_yields, norm.(hhg_spectrum_x[base_id * 2 + 1]))
push!(fs_thz_fig_list, fs_thz_fig)
push!(pmd_fig_list, p3)
push!(py_ave_list, py_ave)

At_thz_xyz, Et_thz_data, = create_tdata(tmax, 0, Δt, no_light, t -> E_applied(t), no_light)
push!(actual_At_list, At_thz_xyz[2][Int64((tau_fs + nc*pi/ω_fs) ÷ Δt) + 1])

# Clear all (Important!!!)
clear_ctmc_rt(rt)

end

# plot(tau_list, [py_ave_list -actual_At_list])

tstart = Int64(tau_fs ÷ Δt) + 1
# plot(pdd_list[2][rt.t_num, :])
heatmap(transpose(pdd_list[1][tstart: rt.t_num, :]) .^ 1)
heatmap(transpose(pdd_list[2][tstart: rt.t_num, :]) .^ 0.4, size=(800, 200))