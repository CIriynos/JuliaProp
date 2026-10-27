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
filter_threshold = 0.5
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

nc = 6
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
rt_perturb = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)

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

soft_recolli_trajs_ti = []
soft_recolli_trajs_pv = []
soft_recolli_trajs_theta = []

all_trajs_ti = []
all_trajs_pv = []
all_trajs_theta = []
all_start_points = []
all_err_r = []
all_err_v = []

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

max_m = 200
for m = 1: max_m
# m = 1
    println("m = $m")
    # pv_max = 1.0
    pv_max = 0.5
    # tid_cc, pv_cc, theta_cc = generate_start_point_uniform_special(trajs_num, m, shift=Int64(tau_fs ÷ Δt + 1))

    # tid_cc, pv_cc, theta_cc = generate_start_point_random(trajs_num, t_num, pv_max, Et_data_one_cycle_burst, Ip, Z)
    # tid_cc, pv_cc, theta_cc = generate_start_point_random_xy_plane(trajs_num, t_num, pv_max, Et_data_one_cycle_burst, Ip, Z)
    # tid_cc, pv_cc, theta_cc = generate_start_point_random_xy_plane(trajs_num, t_num, pv_max, Et_data, Ip, Z)
    # tid_cc, pv_cc, theta_cc = generate_start_point_uniform_xy_plane(trajs_num, t_num, pv_max, Et_data_one_cycle_burst, Ip, Z)
    tid_cc, pv_cc, theta_cc = generate_start_point_uniform_xy_plane_m(trajs_num, t_num, pv_max, Et_data_one_cycle_burst, Ip, Z, max_m, m)

    update_start_point(rt, trajs_num, tid_cc, pv_cc, theta_cc, Et_data, Ip, Z, Δt)
    update_start_point(rt_perturb, trajs_num, tid_cc, pv_cc, theta_cc, Et_data, Ip, Z, Δt)

    # Mainloop for CTMC
    traj_filter_flag = ctmc_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, filter_threshold)
    traj_filter_flag_perturb = ctmc_perturb_mainloop(
        F1, rt_perturb, t_num, trajs_num, tid_cc, Δt,
        Et_data, Et_data_hf, Z, filter_threshold;
        E_ref_data=Et_data_no_thz,
        E_ref_hf_data=Et_data_hf_no_thz,
        order=2,
        softening=1e-5
    )
    # traj_filter_flag_perturb = traj_filter_flag
    println((length(traj_filter_flag), sum(traj_filter_flag)))
    println((length(traj_filter_flag_perturb), sum(traj_filter_flag_perturb)))

    # Add data to PMD/HHG
    add_to_pmd(rt, trajs_num, t_num, Z, traj_filter_flag)
    add_to_hhg(rt, trajs_num, t_num, Δt, tid_cc, traj_filter_flag)

    # error estimation
    err_list_r = zeros(Float64, rt.trajs_num)
    err_list_v = zeros(Float64, rt.trajs_num)
    for i = 1: rt.trajs_num
        delta_r = (rt.traj_data[4][i][end] - rt_perturb.traj_data[4][i][end]) ^ 2 +
            (rt.traj_data[5][i][end] - rt_perturb.traj_data[5][i][end]) ^ 2 +
            (rt.traj_data[6][i][end] - rt_perturb.traj_data[6][i][end]) ^ 2
        delta_v = (rt.traj_data[1][i][end] - rt_perturb.traj_data[1][i][end]) ^ 2 +
            (rt.traj_data[2][i][end] - rt_perturb.traj_data[2][i][end]) ^ 2 +
            (rt.traj_data[3][i][end] - rt_perturb.traj_data[3][i][end]) ^ 2
        err_list_r[i] = delta_r
        err_list_v[i] = delta_v
    end

    # for the perturb, one should filter those trajs with big error
    traj_filter_flag_perturb_with_error_esti = copy(traj_filter_flag_perturb)
    for i = 1: rt.trajs_num
        if err_list_v[i] > err_v_threshold
            traj_filter_flag_perturb_with_error_esti[i] = 1
        end
    end

    # then added to the pmd (get rid of the error trajs)
    add_to_pmd(rt_perturb, trajs_num, t_num, Z, traj_filter_flag_perturb_with_error_esti)
    add_to_hhg(rt_perturb, trajs_num, t_num, Δt, tid_cc, traj_filter_flag_perturb_with_error_esti)

    # in bound electrons
    asym_px_data, asym_py_data, asym_pz_data, asym_filter_flag = calculate_asymptotic_momentum(rt, trajs_num, t_num, Z)
    asym_filtered_ti = [tid_cc[i] for i = 1: trajs_num if asym_filter_flag[i] == true]
    asym_filtered_pv = [pv_cc[i] for i = 1: trajs_num if asym_filter_flag[i] == true]
    asym_filtered_theta = [theta_cc[i] for i = 1: trajs_num if asym_filter_flag[i] == true]

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

    # fig_asym_py_histo = histogram(asym_py_data, yscale=:log10)
    # vline!(fig_asym_py_histo, [-Up_momentum_10, Up_momentum_10])
    # fig_weighted_asym_py_histo = histogram(weighted_asym_py_data, yscale=:log10)

    # fig_asym_py = scatter(ts[tid_cc], asym_py_data, markersize=1.5, markerstrokewidth=0, xlimits=(center_burst_left_bound, center_burst_right_bound))
    # hline!(fig_asym_py, [asym_py_mean, asym_py_mean + asym_py_std, asym_py_mean - asym_py_std])
    # fig_weighted_asym_py = scatter(ts[tid_cc], weighted_asym_py_data, markersize=1.5, markerstrokewidth=0, xlimits=(center_burst_left_bound, center_burst_right_bound))
    # hline!(fig_weighted_asym_py, [weighted_asym_py_mean, weighted_asym_py_mean + weighted_asym_py_std, weighted_asym_py_mean - weighted_asym_py_std])

    # abnormal_id = findall(abs.(weighted_asym_py_data) .>= std(weighted_asym_py_data))
    # inbound_ids = findall(asym_filter_flag)
    # scatter!(fig_weighted_asym_py, tid_cc[abnormal_id], weighted_asym_py_data[abnormal_id], markersize=2, markerstrokewidth=1, marker=:x, markercolor="green")
    # # sum(abs.(weighted_asym_py_data) .>= std(weighted_asym_py_data)) / length(weighted_asym_py_data)

    # # plot abnormal curves
    # # most_odd_id = findall(abs.(weighted_asym_py_data) .== weighted_asym_py_max)[1]
    # most_odd_id = findall(abs.(asym_py_data) .== asym_py_max)[1]

    # id = most_odd_id
    # # id = abnormal_id[1]
    # # id = inbound_ids[1]
    # stp = tid_cc[id]
    # asym_filter_flag[id]
    # traj_filter_flag[id]
    # # curve_display_limit = 50
    # curve_display_limit = 20
    # fig_traj_curve_2d = plot(rt.traj_data[4][id][stp + 1:end], rt.traj_data[5][id][stp + 1:end], xlimits=(-curve_display_limit, curve_display_limit), ylimits=(-curve_display_limit, curve_display_limit), aspect_ratio = :equal)
    # scatter!(fig_traj_curve_2d, [0], [0], markersize=2, markerstrokewidth=0)
    # plot!(fig_traj_curve_2d, filter_threshold .* sin.(0:0.01:2pi), filter_threshold .* cos.(0:0.01:2pi), linestyle=:dashdot)
    # plot(ts[tid_cc[id]: t_num], rt.traj_data[4][id][tid_cc[id]: t_num], ylimits=(-curve_display_limit, curve_display_limit))
    # plot(ts[tid_cc[id]: t_num], rt.traj_data[5][id][tid_cc[id]: t_num], ylimits=(-curve_display_limit, curve_display_limit))
    # fig_traj_curve_2d
    # minimum(sqrt.(rt.traj_data[4][id][tid_cc[id]: t_num] .^ 2 .+ rt.traj_data[5][id][tid_cc[id]: t_num] .^ 2 .+ rt.traj_data[6][id][tid_cc[id]: t_num] .^ 2))

    # # plot all trajs
    # trace_display_limit = 50 * 2
    # trace_display_interval = 0.1 * 2
    # trace_display_range = (-trace_display_limit): trace_display_interval: trace_display_limit
    # fig_all_traj_trace_data = zeros(Float64, length(trace_display_range), length(trace_display_range))
    # for i = 1: 1: rt.trajs_num
    #     # if !traj_filter_flag[i]
    #     #     continue
    #     # end
    #     # if !asym_filter_flag[i]
    #     #     continue
    #     # end
    #     if traj_filter_flag[i] || asym_filter_flag[i]
    #         continue
    #     end
    #     for j = tid_cc[i]: rt.t_num
    #         x = rt.traj_data[4][i][j]
    #         y = rt.traj_data[5][i][j]
    #         x_id = round(Int, (x - first(trace_display_range)) / step(trace_display_range)) + 1
    #         y_id = round(Int, (y - first(trace_display_range)) / step(trace_display_range)) + 1
    #         if x_id >= 1 && x_id <= length(trace_display_range) && y_id >= 1 && y_id <= length(trace_display_range)
    #             fig_all_traj_trace_data[x_id, y_id] += rt.weight_cc[i]
    #             # fig_all_traj_trace_data[x_id, y_id] += 1
    #         end
    #     end
    # end
    # fig_all_traj_trace = heatmap(trace_display_range, trace_display_range, log10.(fig_all_traj_trace_data))

    append!(inbound_trajs_ti, asym_filtered_ti)
    append!(inbound_trajs_pv, asym_filtered_pv)
    append!(inbound_trajs_theta, asym_filtered_theta)

    append!(soft_recolli_trajs_ti, [tid_cc[i] for i = 1: trajs_num if traj_filter_flag[i] == true])
    append!(soft_recolli_trajs_pv, [pv_cc[i] for i = 1: trajs_num if traj_filter_flag[i] == true])
    append!(soft_recolli_trajs_theta, [theta_cc[i] for i = 1: trajs_num if traj_filter_flag[i] == true])

    append!(all_trajs_ti, tid_cc)
    append!(all_trajs_pv, pv_cc)
    append!(all_trajs_theta, theta_cc)

    append!(all_start_points, rt.start_point_cc[4])

    append!(all_err_r, err_list_r)
    append!(all_err_v, err_list_v)

    px_ave, py_ave, pz_ave = get_average_p_ctmc(rt)    
    println("py_ave = $py_ave")
    push!(py_ave_varying_list, py_ave)
end
push!(pdd_list, pdd_tmp)

pdd_list_tmp = []

# HHG
p1, p2, hhg_spectrum_x, hhg_spectrum_y, base_id, hhg_k_linspace = ctmc_get_hhg_spectrum(rt, Et_data, tmax, t_num, ts, Δt, ω_fs, tau_fs)

# PMD
p_axis = p_min: p_delta: p_max
p3 = heatmap(p_axis, p_axis, rt.pxy_final, color=:jet1, size=(500, 420))
p3_perturb = heatmap(p_axis, p_axis, rt_perturb.pxy_final, color=:jet1, size=(500, 420))

px_ave, py_ave, pz_ave = get_average_p_ctmc(rt)
px_ave_perturb, py_ave_perturb, pz_ave_perturb = get_average_p_ctmc(rt_perturb)
println("Py_average = $py_ave")
println("Py_average_perturb = $py_ave_perturb")

# Record
push!(hhg_plt_list, p2)
push!(shg_yields, norm.(hhg_spectrum_x[base_id * 2 + 1]))
push!(fs_thz_fig_list, fs_thz_fig)
push!(pmd_fig_list, p3)
push!(py_ave_list, py_ave)

# At_thz_xyz, Et_thz_data, = create_tdata(tmax, 0, Δt, no_light, t -> E_applied(t), no_light)
# push!(actual_At_list, At_thz_xyz[2][Int64((tau_fs + nc*pi/ω_fs) ÷ Δt) + 1])


# check the validity of the trajs
scatter(inbound_trajs_ti, inbound_trajs_pv, inbound_trajs_theta, xlabel="time", ylabel="pv", zlabel="theta", markersize=1.0, camera = (90, 0))
scatter(inbound_trajs_ti, inbound_trajs_pv, inbound_trajs_theta, xlabel="time", ylabel="pv", zlabel="theta", markersize=1.0)

pp1 = scatter(ts[inbound_trajs_ti], inbound_trajs_pv, xlabel="time", ylabel="pv", label="inbound", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))
pp2 = scatter(ts[inbound_trajs_ti], inbound_trajs_theta, xlabel="time", ylabel="theta", label="inbound", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))
pp3 = scatter(inbound_trajs_pv, inbound_trajs_theta, xlabel="pv", ylabel="theta", label="inbound", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))

scatter!(pp1, ts[soft_recolli_trajs_ti], soft_recolli_trajs_pv, xlabel="time", ylabel="pv", label="soft_recolli", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))
scatter!(pp2, ts[soft_recolli_trajs_ti], soft_recolli_trajs_theta, xlabel="time", ylabel="theta", label="soft_recolli", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))
scatter!(pp3, soft_recolli_trajs_pv, soft_recolli_trajs_theta, xlabel="pv", ylabel="theta", label="soft_recolli", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.8, xlimits=(center_burst_left_bound, center_burst_right_bound))

# all_trajs_display_interval = 1
# scatter!(pp1, ts[all_trajs_ti[1:all_trajs_display_interval:end]], all_trajs_pv, xlabel="time", ylabel="pv", label="all", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.2, xlimits=(center_burst_left_bound, center_burst_right_bound))
# scatter!(pp2, ts[all_trajs_ti[1:all_trajs_display_interval:end]], all_trajs_theta, xlabel="time", ylabel="theta", label="all", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.2, xlimits=(center_burst_left_bound, center_burst_right_bound))
# scatter!(pp3, all_trajs_pv[1:all_trajs_display_interval:end], all_trajs_theta, xlabel="pv", ylabel="theta", label="all", markersize=1.0, markershape=:circle, markerstrokewidth=0, markeralpha=0.2, xlimits=(center_burst_left_bound, center_burst_right_bound))
# # vline!(pp1, [center_burst_left_bound mid_point center_burst_right_bound], ls=:dashdot, label=["left bound" "mid" "right bound"])
vline!(pp1, [mid_point], ls=:dashdot, label="mid")

mid_pulse_tid_rg = minimum(all_trajs_ti): maximum(all_trajs_ti)
mid_pulse_pv_rg = minimum(all_trajs_pv): 0.001: maximum(all_trajs_pv)
# pp1_twin = plot!(twinx(), ts[mid_pulse_tid_rg], normalize(W0.(Et_data[1][mid_pulse_tid_rg], Ip, Z)), label="ionz. rate")

pp1_weight = contour(pp1, ts[mid_pulse_tid_rg], mid_pulse_pv_rg, W.(Et_data[1][mid_pulse_tid_rg]', mid_pulse_pv_rg, Ip, Z) .* 1e1, alpha=0.5)


length(all_trajs_ti)
length(soft_recolli_trajs_ti)
length(inbound_trajs_ti)

# check the initial points
histogram(all_start_points)

# # Clear all (Important!!!)
# clear_ctmc_rt(rt)

p3
p3_perturb


# err estimation
# scatter(all_trajs_ti, all_trajs_pv, min.(all_err_r, 20), markersize=1, markerstrokewidth=0)
fig_err_r_less_threshold = scatter(all_trajs_ti[all_err_r .> err_r_threshold], all_trajs_pv[all_err_r .> err_r_threshold], markersize=1, markerstrokewidth=0, label="err_r > $(err_r_threshold)", legend=:topright)
scatter!(fig_err_r_less_threshold, all_trajs_ti[all_err_r .< err_r_threshold], all_trajs_pv[all_err_r .< err_r_threshold], markersize=1, markerstrokewidth=0, label="err_r < $(err_r_threshold)", legend=:topright)
scatter!(fig_err_r_less_threshold, pp1)

# scatter(all_trajs_ti, all_trajs_pv, min.(all_err_v, 20), markersize=1, markerstrokewidth=0)
fig_err_v_less_threshold = scatter(all_trajs_ti[all_err_v .> err_v_threshold], all_trajs_pv[all_err_v .> err_v_threshold], markersize=1, markerstrokewidth=0, label="err_v > $(err_v_threshold)", legend=:topright)
scatter!(fig_err_v_less_threshold, all_trajs_ti[all_err_v .< err_v_threshold], all_trajs_pv[all_err_v .< err_v_threshold], markersize=1, markerstrokewidth=0, markeralpha=0.3, label="err_v < $(err_v_threshold)", legend=:topright)
fig_err_vc_cmp = scatter(fig_err_v_less_threshold, pp1)

# some statistics
println("Proportion of the invalid samples (err_r > $err_r_threshold): ", @sprintf("%.2f%%", (sum(all_err_r .> err_r_threshold) / length(all_trajs_ti)) * 100))
println("Proportion of the invalid samples (err_v > $err_v_threshold): ", @sprintf("%.2f%%", (sum(all_err_v .> err_v_threshold) / length(all_trajs_ti)) * 100))
println("Proportion of the inbound samples : ", @sprintf("%.2f%%", (length(inbound_trajs_ti) / length(all_trajs_ti)) * 100))
println("Proportion of the soft_recolli samples : ", @sprintf("%.2f%%\n", (length(soft_recolli_trajs_ti) / length(all_trajs_ti)) * 100))

plot(py_ave_varying_list)
# plot(diff(py_ave_varying_list))

# fig_err_vc_cmp
# pp1_weight

p3_perturb
p3