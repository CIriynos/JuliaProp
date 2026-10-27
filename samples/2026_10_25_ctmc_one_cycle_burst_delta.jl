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
using Statistics


function W0(E, Ip, Z)
    F = abs(E) + 1e-50
    D = (4 * Z^3 / F)
    return (F * D^2 / (8 * pi * Z)) * exp(-2 * (2 * Ip) ^ (3/2) / (3 * F))
end

function W1(p⊥, E, Ip)
    F = abs(E) + 1e-50
    return sqrt(2 * Ip) / (pi * F) * exp(- (p⊥^2) * sqrt(2 * Ip) / F)
end

W(E, p⊥, Ip, Z) = W0(E, Ip, Z) * W1(p⊥, E, Ip)


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
Δt = 0.2 / 2
# Δt = 0.05
filter_threshold = 0.1
# filter_threshold = 0.5
# filter_threshold = 2.0
# filter_threshold = 5.0
p_min = -1.0
p_max = 1.0
p_delta = 0.005

# error estimation (for 2)
err_r_threshold = 1e-1
err_v_threshold = 1e-4

# Define the Laser. 
E_fs = 0.0533 * 1.0
E_thz = 0.00005 * 1
E_dc = 0.0005 * 0
ω_fs = 0.05693
ω_thz = ω_fs / 10

nc = 4
tau_fs = 0
nc_thz = 2.5
cep = 0.5pi
# cep = 0

# create runtime for CTMC
trajs_num = 2000
# trajs_num = 5000
tmax = 2 * nc * pi / ω_fs
t_num = Int64(tmax ÷ Δt) + 1
rt = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)
rt_zero = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)

task_id = 7
tasks_num = 16
# for task_id = [1]

# t-data for Field
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz, samples_num=tasks_num, nc_thz=2)
tau_thz = tau_list[task_id]

Ex_fs, Ey_fs, Ez_fs = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.0, phase1=0.5pi+cep, phase2=0+cep)
E_thz_1, E_thz_2, E_thz_3, tmax_thz = light_pulse(ω_thz, E_thz, nc_thz, tau_thz, pulse_shape="sin2", phase1=0.5pi)
E_window = dc_bias(1.0, 0, tau_fs, tmax, tmax * 2)
E_applied(t) = (E_thz_1(t)) #* E_window(t)

At_data_xyz, Et_data, ts, = create_tdata(tmax, 0, Δt, Ex_fs, t -> Ey_fs(t) + E_applied(t), no_light)
At_data_xyz_hf, Et_data_hf, ts_hf, = create_tdata(tmax, Δt/2, Δt, Ex_fs, t -> Ey_fs(t) + E_applied(t), no_light)
At_data_xyz_no_thz, Et_data_no_thz, = create_tdata(tmax, 0, Δt, Ex_fs, t -> Ey_fs(t), no_light)
At_data_xyz_hf_no_thz, Et_data_hf_no_thz, = create_tdata(tmax, Δt/2, Δt, Ex_fs, t -> Ey_fs(t), no_light)

fs_thz_fig = plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts, thz_ratio=400)
mid_point = tau_fs + nc*pi/ω_fs
center_burst_left_bound = mid_point - 0.5 * pi/ω_fs
center_burst_right_bound = mid_point + 0.5 * pi/ω_fs
vline!(fs_thz_fig, [center_burst_left_bound, center_burst_right_bound], ls=:dashdot, label="mid")


pdd_tmp = Matrix{Float64}
inbound_trajs_ti = []
inbound_trajs_pv = []
inbound_trajs_theta = []

inbound_trajs_ti_zero = []
inbound_trajs_pv_zero = []
inbound_trajs_theta_zero = []

soft_recolli_trajs_ti = []
soft_recolli_trajs_pv = []
soft_recolli_trajs_theta = []

soft_recolli_trajs_ti_zero = []
soft_recolli_trajs_pv_zero = []
soft_recolli_trajs_theta_zero = []

all_trajs_ti = []
all_trajs_pv = []
all_trajs_theta = []
all_start_points = []
all_err_r = []
all_err_v = []

delta_py_data_cc = []
delta_py_data_flag_cc = []

Et_data_one_cycle_burst = deepcopy(Et_data)
for it = 1: t_num
    t = ts[it]
    if t <= center_burst_left_bound || t >= center_burst_right_bound
    # if t <= center_burst_left_bound || t >= (mid_point + 1.5 * pi/ω_fs)
        Et_data_one_cycle_burst[1][it] = 0.0
        Et_data_one_cycle_burst[2][it] = 0.0
        Et_data_one_cycle_burst[3][it] = 0.0
    end
end

# 0: Δt: tmax
# tids_central = nearest_index(0: Δt: tmax, center_burst_left_bound): 

py_ave_varying_list = []
pv_max = 0.4
max_m = 30
for m = 1: max_m
# m = 1
    println("m = $m / $max_m")
    # pv_max = 1.0
    # tid_cc, pv_cc, theta_cc = generate_start_point_uniform_special(trajs_num, m, shift=Int64(tau_fs ÷ Δt + 1))

    # tid_cc, pv_cc, theta_cc = generate_start_point_random(trajs_num, t_num, pv_max, Et_data_one_cycle_burst, Ip, Z)
    # tid_cc, pv_cc, theta_cc = generate_start_point_random_xy_plane(trajs_num, t_num, pv_max, Et_data_one_cycle_burst, Ip, Z)
    # tid_cc, pv_cc, theta_cc = generate_start_point_random_xy_plane(trajs_num, t_num, pv_max, Et_data, Ip, Z)
    # tid_cc, pv_cc, theta_cc = generate_start_point_uniform_xy_plane(trajs_num, t_num, pv_max, Et_data_one_cycle_burst, Ip, Z)
    tid_cc, pv_cc, theta_cc = generate_start_point_uniform_xy_plane_m(trajs_num, t_num, pv_max, Et_data_one_cycle_burst, Ip, Z, max_m, m)

    update_start_point(rt, trajs_num, tid_cc, pv_cc, theta_cc, Et_data, Ip, Z, Δt)
    update_start_point(rt_zero, trajs_num, tid_cc, pv_cc, theta_cc, Et_data, Ip, Z, Δt)

    # Mainloop for CTMC
    traj_filter_flag = ctmc_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, filter_threshold)
    traj_filter_flag_zero = ctmc_mainloop(F1, rt_zero, t_num, trajs_num, tid_cc, Δt, Et_data_no_thz, Et_data_hf_no_thz, Z, filter_threshold)

    println((length(traj_filter_flag), sum(traj_filter_flag)))
    println((length(traj_filter_flag_zero), sum(traj_filter_flag_zero)))

    # Add data to PMD/HHG
    add_to_pmd(rt, trajs_num, t_num, Z, traj_filter_flag)
    add_to_hhg(rt, trajs_num, t_num, Δt, tid_cc, traj_filter_flag)
    add_to_pmd(rt_zero, trajs_num, t_num, Z, traj_filter_flag_zero)
    add_to_hhg(rt_zero, trajs_num, t_num, Δt, tid_cc, traj_filter_flag_zero)

    # in bound electrons
    asym_px_data, asym_py_data, asym_pz_data, asym_filter_flag = calculate_asymptotic_momentum(rt, trajs_num, t_num, Z)
    asym_filtered_ti = [tid_cc[i] for i = 1: trajs_num if asym_filter_flag[i] == true]
    asym_filtered_pv = [pv_cc[i] for i = 1: trajs_num if asym_filter_flag[i] == true]
    asym_filtered_theta = [theta_cc[i] for i = 1: trajs_num if asym_filter_flag[i] == true]

    asym_px_data_zero, asym_py_data_zero, asym_pz_data_zero, asym_filter_flag_zero = calculate_asymptotic_momentum(rt_zero, trajs_num, t_num, Z)
    asym_filtered_ti_zero = [tid_cc[i] for i = 1: trajs_num if asym_filter_flag_zero[i] == true]
    asym_filtered_pv_zero = [pv_cc[i] for i = 1: trajs_num if asym_filter_flag_zero[i] == true]
    asym_filtered_theta_zero = [theta_cc[i] for i = 1: trajs_num if asym_filter_flag_zero[i] == true]

    # statistics
    weighted_asym_py_data = asym_py_data .* rt.weight_cc .* .!(asym_filter_flag) .* .!(traj_filter_flag)
    asym_py_data = asym_py_data .* .!(asym_filter_flag) .* .!(traj_filter_flag)
    weighted_asym_py_std = std(weighted_asym_py_data)
    weighted_asym_py_mean = mean(weighted_asym_py_data)
    weighted_asym_py_max = maximum(abs.(weighted_asym_py_data))
    weighted_var_sigma = round(weighted_asym_py_max / weighted_asym_py_std, digits=1)
    asym_py_std = std(asym_py_data)
    asym_py_mean = mean(asym_py_data)
    asym_py_max = maximum(abs.(asym_py_data))
    var_sigma = round(asym_py_max / asym_py_std, digits=1)
    println("Max of weighted_asym_py_data = $(weighted_asym_py_max), at $(weighted_var_sigma)-σ")
    println("Max of asym_py_data = $(asym_py_max), at $(var_sigma)-σ")
    Up_momentum = sqrt(2 * E_fs ^ 2 / (4 * ω_fs^2))
    Up_momentum_10 = sqrt(2 * E_fs ^ 2 / (4 * ω_fs^2) * 10)

    # compare the contribution from each trajs to the delta p_y
    # trajs below the threshold should be omitted
    delta_py_data = zeros(Float64, trajs_num)
    delta_py_data_flag = zeros(Int64, trajs_num)  # 0: normal; 1: not inbound -> bound; 2: inbound -> not inbound; 3: always inbound
    for i = 1: trajs_num
        delta_py_data[i] = asym_py_data[i] - asym_py_data_zero[i]
        if asym_filter_flag[i] == 0 && asym_filter_flag_zero[i] == 1
            delta_py_data_flag[i] = 1
        elseif asym_filter_flag[i] == 1 && asym_filter_flag_zero[i] == 0
            delta_py_data_flag[i] = 2
        elseif asym_filter_flag[i] == 1 && asym_filter_flag_zero[i] == 1
            delta_py_data_flag[i] = 3
        elseif traj_filter_flag[i] || traj_filter_flag_zero[i]
            delta_py_data_flag[i] = 4
        end
        if delta_py_data_flag[i] != 0
            delta_py_data[i] = 0
        end
    end

    # plot abnormal curves
    most_odd_id = findall(abs.(delta_py_data) .== maximum(abs.(delta_py_data)))[1]
    delta_py_data_flag[most_odd_id]
    traj_filter_flag[most_odd_id]

    id = most_odd_id
    stp = tid_cc[id]
    asym_filter_flag[id]
    traj_filter_flag[id]
    curve_display_limit = 20
    fig_traj_curve_2d = plot(rt.traj_data[4][id][stp + 1:end], rt.traj_data[5][id][stp + 1:end], xlimits=(-curve_display_limit, curve_display_limit), ylimits=(-curve_display_limit, curve_display_limit), aspect_ratio = :equal)
    scatter!(fig_traj_curve_2d, [0], [0], markersize=2, markerstrokewidth=0)
    plot!(fig_traj_curve_2d, filter_threshold .* sin.(0:0.01:2pi), filter_threshold .* cos.(0:0.01:2pi), linestyle=:dashdot)
    plot(ts[tid_cc[id]: t_num], rt.traj_data[4][id][tid_cc[id]: t_num], ylimits=(-curve_display_limit, curve_display_limit))
    plot(ts[tid_cc[id]: t_num], rt.traj_data[5][id][tid_cc[id]: t_num], ylimits=(-curve_display_limit, curve_display_limit))
    plot(rt.traj_data[1][id][stp + 1:end], rt.traj_data[2][id][stp + 1:end], xlimits=(-2.0, 2.0), ylimits=(-2.0, 2.0), aspect_ratio=:equal)

    fig_traj_curve_2d

    append!(inbound_trajs_ti, asym_filtered_ti)
    append!(inbound_trajs_pv, asym_filtered_pv)
    append!(inbound_trajs_theta, asym_filtered_theta)
    append!(inbound_trajs_ti_zero, asym_filtered_ti_zero)
    append!(inbound_trajs_pv_zero, asym_filtered_pv_zero)
    append!(inbound_trajs_theta_zero, asym_filtered_theta_zero)

    append!(soft_recolli_trajs_ti, [tid_cc[i] for i = 1: trajs_num if traj_filter_flag[i] == true])
    append!(soft_recolli_trajs_pv, [pv_cc[i] for i = 1: trajs_num if traj_filter_flag[i] == true])
    append!(soft_recolli_trajs_theta, [theta_cc[i] for i = 1: trajs_num if traj_filter_flag[i] == true])
    append!(soft_recolli_trajs_ti_zero, [tid_cc[i] for i = 1: trajs_num if traj_filter_flag_zero[i] == true])
    append!(soft_recolli_trajs_pv_zero, [pv_cc[i] for i = 1: trajs_num if traj_filter_flag_zero[i] == true])
    append!(soft_recolli_trajs_theta_zero, [theta_cc[i] for i = 1: trajs_num if traj_filter_flag_zero[i] == true])

    append!(delta_py_data_cc, delta_py_data)
    append!(delta_py_data_flag_cc, delta_py_data_flag)

    append!(all_trajs_ti, tid_cc)
    append!(all_trajs_pv, pv_cc)
    append!(all_trajs_theta, theta_cc)
    append!(all_start_points, rt.start_point_cc[4])

    # px_ave, py_ave, pz_ave = get_average_p_ctmc(rt)    
    # println("py_ave = $py_ave")
    # push!(py_ave_varying_list, py_ave)
end


# PMD
p_axis = p_min: p_delta: p_max
p3 = heatmap(p_axis, p_axis, rt.pxy_final, color=:jet1, size=(500, 420))
p3_zero = heatmap(p_axis, p_axis, rt_zero.pxy_final, color=:jet1, size=(500, 420))

px_ave, py_ave, pz_ave = get_average_p_ctmc(rt)
px_ave_zero, py_ave_zero, pz_ave_zero = get_average_p_ctmc(rt_zero)
println("Py_average = $py_ave")
println("Py_average_zero = $py_ave_zero")

# check the delta_py by the E_thz
delta_py_data_cc_display_data = copy(delta_py_data_cc)
absmax = maximum(abs, delta_py_data_cc_display_data)
for i in eachindex(delta_py_data_cc_display_data)
    if delta_py_data_flag_cc[i] != 0
        delta_py_data_cc_display_data[i] = NaN
    end
end
valid_idx = .!isnan.(delta_py_data_cc_display_data)
fig_delta_py = scatter(ts[all_trajs_ti[valid_idx]], all_trajs_pv[valid_idx] .* cos.(all_trajs_theta[valid_idx] .- pi/2), marker_z=delta_py_data_cc_display_data[valid_idx],
    xlimits=(center_burst_left_bound, center_burst_right_bound), clims = (-absmax, absmax),
    markersize=1.2, markershape=:circle, markerstrokewidth=0, markeralpha=1.0, color=cgrad([:blue, :white, :red])) #color=cgrad([:blue, :cyan, :white, :red, :magenta]))
mid_pulse_tid_rg = minimum(all_trajs_ti): maximum(all_trajs_ti)
mid_pulse_pv_rg = -maximum(all_trajs_pv): 0.001: maximum(all_trajs_pv)
ionz_rate_data = W.(Et_data[1][mid_pulse_tid_rg]', mid_pulse_pv_rg, Ip, Z)
contour!(fig_delta_py, ts[mid_pulse_tid_rg], mid_pulse_pv_rg, (ionz_rate_data) ./ (maximum(ionz_rate_data) / absmax), alpha=0.5, color=cgrad([:grey, :white, :grey]), xlimits=(center_burst_left_bound, center_burst_right_bound))
fig_delta_py


# check the validity of the trajs
scatter(inbound_trajs_ti, inbound_trajs_pv, inbound_trajs_theta, xlabel="time", ylabel="pv", zlabel="theta", markersize=1.0, camera = (90, 0))
scatter(inbound_trajs_ti, inbound_trajs_pv, inbound_trajs_theta, xlabel="time", ylabel="pv", zlabel="theta", markersize=1.0)

pp1 = scatter(ts[inbound_trajs_ti], inbound_trajs_pv, xlabel="time", ylabel="pv", label="inbound", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))
pp2 = scatter(ts[inbound_trajs_ti], inbound_trajs_theta, xlabel="time", ylabel="theta", label="inbound", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))
pp3 = scatter(inbound_trajs_pv, inbound_trajs_theta, xlabel="pv", ylabel="theta", label="inbound", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))

scatter!(pp1, ts[soft_recolli_trajs_ti], soft_recolli_trajs_pv, xlabel="time", ylabel="pv", label="soft_recolli", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))
scatter!(pp2, ts[soft_recolli_trajs_ti], soft_recolli_trajs_theta, xlabel="time", ylabel="theta", label="soft_recolli", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))
scatter!(pp3, soft_recolli_trajs_pv, soft_recolli_trajs_theta, xlabel="pv", ylabel="theta", label="soft_recolli", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))

all_trajs_display_interval = 1
scatter!(pp1, ts[all_trajs_ti[1:all_trajs_display_interval:end]], all_trajs_pv, xlabel="time", ylabel="pv", label="all", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.2, xlimits=(center_burst_left_bound, center_burst_right_bound))
scatter!(pp2, ts[all_trajs_ti[1:all_trajs_display_interval:end]], all_trajs_theta, xlabel="time", ylabel="theta", label="all", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.2, xlimits=(center_burst_left_bound, center_burst_right_bound))
scatter!(pp3, all_trajs_pv[1:all_trajs_display_interval:end], all_trajs_theta, xlabel="pv", ylabel="theta", label="all", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.2, xlimits=(center_burst_left_bound, center_burst_right_bound))
# vline!(pp1, [center_burst_left_bound mid_point center_burst_right_bound], ls=:dashdot, label=["left bound" "mid" "right bound"])
vline!(pp1, [mid_point], ls=:dashdot, label="mid")

mid_pulse_tid_rg = minimum(all_trajs_ti): maximum(all_trajs_ti)
mid_pulse_pv_rg = minimum(all_trajs_pv): 0.001: maximum(all_trajs_pv)
pp1_weight = contour(pp1, ts[mid_pulse_tid_rg], mid_pulse_pv_rg, W.(Et_data[1][mid_pulse_tid_rg]', mid_pulse_pv_rg, Ip, Z) .* 1e1, alpha=0.5)

length(all_trajs_ti)
length(soft_recolli_trajs_ti)
length(inbound_trajs_ti)

# check the initial points
histogram(all_start_points)

# # Clear all (Important!!!)
# clear_ctmc_rt(rt)

p3
p3_zero

# some statistics
println("Proportion of the inbound samples : ", @sprintf("%.2f%%", (length(inbound_trajs_ti) / length(all_trajs_ti)) * 100))
println("Proportion of the soft_recolli samples : ", @sprintf("%.2f%%\n", (length(soft_recolli_trajs_ti) / length(all_trajs_ti)) * 100))

plot(py_ave_varying_list)
# plot(diff(py_ave_varying_list))

# fig_err_vc_cmp
# pp1_weight

p3_zero
p3

fig_delta_py