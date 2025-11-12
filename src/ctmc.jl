######################################

# index 1 ~ 6 --> px, py, pz, x, y, z
const PX_ID = 1; const PY_ID = 2; const PZ_ID = 3;
const X_ID = 4;  const Y_ID = 5;  const Z_ID = 6;
p2r(id) = id + 3
r2p(id) = id - 3


function RK4_procedure(
    F1::Function, i::Int64,
    traj_data, start_point_cc::Vector{Vector{Float64}},
    tid::Int64, t_num::Int64, Δt::Float64,
    E_data::Vector{Vector{Float64}}, E_hf_data::Vector{Vector{Float64}}, Z::Float64,
    filter_threshold::Float64
)
    r = [0.0, 0.0, 0.0]
    ks = [zeros(Float64, 4) for _ = 1: 6]
    filter_flag = false
    # copy start point to the traj_data
    for j = 1: 6
        traj_data[j][i][tid] = start_point_cc[j][i]
    end
    # start RK4
    for k = tid + 1: t_num
        r[1] = traj_data[X_ID][i][k-1]
        r[2] = traj_data[Y_ID][i][k-1]
        r[3] = traj_data[Z_ID][i][k-1]

        if sqrt(r[1]^2 + r[2]^2 + r[3]^2) < filter_threshold
            filter_flag = true
        end

        # dp/dt = -E(t) - Zr/r^3
        @fastmath @inbounds for j = PX_ID: PZ_ID
            ks[j][1] = F1(E_data[j][k-1], Z, r[1], r[2], r[3], r[j])
        end
        @fastmath @inbounds for j = PX_ID: PZ_ID
            ks[j][2] = F1(E_hf_data[j][k-1], Z,
                r[1] + 0.5Δt * ks[1][1], r[2] + 0.5Δt * ks[2][1],
                r[3] + 0.5Δt * ks[3][1], r[j] + 0.5Δt * ks[j][1])
        end
        @fastmath @inbounds for j = PX_ID: PZ_ID
            ks[j][3] = F1(E_hf_data[j][k-1], Z,
                r[1] + 0.5Δt * ks[1][2], r[2] + 0.5Δt * ks[2][2],
                r[3] + 0.5Δt * ks[3][2], r[j] + 0.5Δt * ks[j][2])
        end
        @fastmath @inbounds for j = PX_ID: PZ_ID
            ks[j][4] = F1(E_data[j][k], Z,
                r[1] + Δt * ks[1][3], r[2] + Δt * ks[2][3],
                r[3] + Δt * ks[3][3], r[j] + Δt * ks[j][3])
        end

        # dr/dt = p
        @fastmath @inbounds for j = X_ID: Z_ID
            ks[j][1] = traj_data[r2p(j)][i][k-1]
        end
        @fastmath @inbounds for j = X_ID: Z_ID
            ks[j][2] = traj_data[r2p(j)][i][k-1] + 0.5Δt * ks[j][1]
        end
        @fastmath @inbounds for j = X_ID: Z_ID
            ks[j][3] = traj_data[r2p(j)][i][k-1] + 0.5Δt * ks[j][2]
        end
        @fastmath @inbounds for j = X_ID: Z_ID
            ks[j][4] = traj_data[r2p(j)][i][k-1] + Δt * ks[j][3]
        end

        # Get them together.
        @fastmath @inbounds for j = 1: 6
            traj_data[j][i][k] = traj_data[j][i][k-1] + (Δt / 6) * (ks[j][1] + 2*ks[j][2] + 2*ks[j][3] + ks[j][4])
        end
    end

    return filter_flag
end

function ppp(plt, ori, vec)
    scatter!(plt, [ori[1]], [ori[2]], [ori[3]], lims=(-5, 5))
    quiver!(plt, [ori[1]], [ori[2]], [ori[3]], 
        quiver=([vec[1]], [vec[2]], [vec[3]]), lims=(-5, 5))
end

function get_exit_point(E, Ip)
    tmp = norm(E)^2 + 1e-50
    return (-Ip*E[1] / tmp, -Ip*E[2] / tmp, -Ip*E[3] / tmp)
end

function get_p_vec(E, pv, theta)
    c = pv * sqrt(E[1]^2 + E[2]^2) / norm(E)
    b = -c * E[2] * E[3] / (E[1]^2 + E[2]^2)
    a = -(b * E[2] + c * E[3]) / E[1]
    r2 = (a, b, c)
    
    n = (E[2]*c - E[3]*b, E[3]*a - E[1]*c, E[1]*b - E[2]*a)
    n = n ./ norm(n) .* sqrt(a^2 + b^2 + c^2)
    res = r2 .* cos(theta) .+ n .* sin(theta)
    return res
end

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

function combine_trajs_data(ID, combined_data, traj_data, traj_filter_flag, weight_cc, tid_cc, trajs_num, t_num, delta_t)
    ai::Float64 = 0.0
    for i = 1: trajs_num
        if traj_filter_flag[i] == false
            for j = 1: t_num
                if j - 2 >= 1 && j + 2 <= t_num
                    ai = four_order_difference(traj_data[ID][i], j, delta_t)
                elseif j == 2 || j == t_num - 1
                    ai = two_order_difference(traj_data[ID][i], j, delta_t)
                elseif j == 1
                    ai = (traj_data[ID][i][2] - traj_data[ID][i][1]) / delta_t
                elseif j == t_num
                    ai = (traj_data[ID][i][t_num] - traj_data[ID][i][t_num-1]) / delta_t
                end
                combined_data[j] += ai * weight_cc[i]
            end
        end
    end
end

function traj_get_acc(P_ID, traj, tid, t_num, delta_t)
    acc_data = zeros(Float64, t_num)
    for j = 1: t_num
        if j - 2 >= 1 && j + 2 <= t_num
            acc_data[j] = four_order_difference(traj, j, delta_t)
        elseif j == 2 || j == t_num - 1
            acc_data[j] = two_order_difference(traj, j, delta_t)
        elseif j == 1
            acc_data[j] = (traj[2] - traj[1]) / delta_t
        elseif j == t_num
            acc_data[j] = (traj[t_num] - traj[t_num-1]) / delta_t
        end
    end
    return acc_data
end


mutable struct ctmc_rt
    # basic
    trajs_num::Int64
    t_num::Int64
    p_min::Float64      # for PMD
    p_max::Float64
    p_delta::Float64

    # RK4 Data Allocation Reusable (It spends a lot of time)
    traj_data::Vector{Vector{Vector{Float64}}}

    # Prepare dataset for PMD / HHG / Analyse
    px_final::Vector{Float64}
    py_final::Vector{Float64}
    pz_final::Vector{Float64}
    pxy_final::Matrix{Float64}
    pxz_final::Matrix{Float64}
    px_data::Vector{Float64}
    py_data::Vector{Float64}
    pz_data::Vector{Float64}
    ax_data::Vector{Float64}
    ay_data::Vector{Float64}
    az_data::Vector{Float64}
    pmd_p_total::Vector{Float64} # 3 elements
    pmd_total_w::Float64

    # Start points
    start_point_cc::Vector{Vector{Float64}}
    weight_cc::Vector{Float64}
end


function create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)

    p_axis = p_min: p_delta: p_max

    # RK4 Data Allocation Reusable (It spends a lot of time)
    traj_data = [[zeros(Float64, t_num) for i = 1: trajs_num] for j = 1: 6]

    # Prepare dataset for PMD / HHG / Analyse
    px_final = zeros(Float64, length(p_axis))
    py_final = zeros(Float64, length(p_axis))
    pz_final = zeros(Float64, length(p_axis))
    pxy_final = zeros(Float64, (length(p_axis), length(p_axis)))
    pxz_final = zeros(Float64, (length(p_axis), length(p_axis)))
    px_data = zeros(Float64, t_num)
    py_data = zeros(Float64, t_num)
    pz_data = zeros(Float64, t_num)
    ax_data = zeros(Float64, t_num)
    ay_data = zeros(Float64, t_num)
    az_data = zeros(Float64, t_num)
    weight_cc = zeros(Float64, trajs_num)
    start_point_cc = [zeros(Float64, trajs_num) for i = 1: 6]

    rt = ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta,
        traj_data, px_final, py_final, pz_final, pxy_final, pxz_final,
        px_data, py_data, pz_data, ax_data, ay_data, az_data, [0.0, 0.0, 0.0], 0,
        start_point_cc, weight_cc)
    return rt
end

function clear_ctmc_rt(rt)
    for j = 1: 6
        Threads.@threads for k = 1: rt.trajs_num
            rt.traj_data[j][k] .= 0.0
        end
        rt.start_point_cc[j] .= 0.0
    end
    rt.px_final .= 0.0
    rt.py_final .= 0.0
    rt.pz_final .= 0.0
    rt.pxy_final .= 0.0
    rt.pxz_final .= 0.0
    rt.px_data .= 0.0
    rt.py_data .= 0.0
    rt.pz_data .= 0.0
    rt.ax_data .= 0.0
    rt.ay_data .= 0.0
    rt.az_data .= 0.0
    rt.weight_cc .= 0.0
    rt.pmd_p_total .= 0.0
    rt.pmd_total_w = 0
end


function generate_start_point_random(trajs_num, t_num, pv_max, E_data, Ip, Z; min_threshold = 1e-8)
    # Random Start-Point Generator
    tid_cc = zeros(Int64, trajs_num)
    p = 1
    while p <= trajs_num
        tid = Int64(rand(1: t_num))
        e_norm = norm([E_data[1][tid], E_data[2][tid], E_data[3][tid]])
        w = W.(e_norm, 0, Ip, Z)
        if w > min_threshold
            tid_cc[p] = tid
            p += 1
        end
    end
    # tid_cc = rand(1: t_num, trajs_num)
    pv_cc = rand(trajs_num) * pv_max
    theta_cc = rand(trajs_num) * 2pi
    return tid_cc, pv_cc, theta_cc
end

function generate_start_point_uniform_special(trajs_num, m; shift::Int64 = 0)
    theta_m_list = [0; range(0, 2*pi, 6); range(0, 2*pi, 12); range(0, 2*pi, 18)]
    pv_m_list = [1e-5; ones(6) .* 0.2; ones(12) .* 0.4; ones(18) .* 0.8]
    tid_cc = 1 + shift: trajs_num + shift
    pv_cc = ones(trajs_num) .* pv_m_list[m]
    theta_cc = ones(trajs_num) .* theta_m_list[m]
    return tid_cc, pv_cc, theta_cc
end

function update_start_point(rt, trajs_num, tid_cc, pv_cc, theta_cc, E_data, Ip, Z, Δt)
    
    # calculate weight_cc
    E_norm_data = [norm([E_data[1][i], E_data[2][i], E_data[3][i]]) for i = 1: length(E_data[1])]
    
    # wt_data = W.(E_norm_data[tid_cc], pv_cc, Ip, Z)
    # wt_int_data = get_integral(wt_data, Δt)
    # println(last(wt_intupdate_start_point
    # @. rt.weight_cc = 1.0 * wt_data * exp(-wt_int_data * C)

    @. rt.weight_cc = W.(E_norm_data[tid_cc], pv_cc, Ip, Z)

    for i = 1: trajs_num
        E_ti = (E_data[1][tid_cc[i]], E_data[2][tid_cc[i]], E_data[3][tid_cc[i]])
        exit_point = get_exit_point(E_ti, Ip)
        rt.start_point_cc[X_ID][i] = exit_point[1]
        rt.start_point_cc[Y_ID][i] = exit_point[2]
        rt.start_point_cc[Z_ID][i] = exit_point[3]

        pv_vec = get_p_vec(E_ti, pv_cc[i], theta_cc[i])
        rt.start_point_cc[PX_ID][i] = pv_vec[1]
        rt.start_point_cc[PY_ID][i] = pv_vec[2]
        rt.start_point_cc[PZ_ID][i] = pv_vec[3]
    end
end

function ctmc_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt, E_data, E_hf_data, Z, filter_threshold)
    traj_filter_flag = zeros(Bool, trajs_num)
    Threads.@threads for i = 1: trajs_num
        flag = RK4_procedure(F1, i, rt.traj_data, rt.start_point_cc, tid_cc[i], t_num, Δt, E_data, E_hf_data, Z, filter_threshold)
        traj_filter_flag[i] = flag
    end
    # println("RK4 ended.")
    return traj_filter_flag
end

function calculate_asymptotic_momentum(rt, trajs_num, t_num, Z)
    asym_px_data = zeros(Float64, trajs_num)
    asym_py_data = zeros(Float64, trajs_num)
    asym_pz_data = zeros(Float64, trajs_num)
    asym_filter_flag = zeros(Bool, trajs_num)

    for i = 1: trajs_num
        xf = rt.traj_data[X_ID][i][t_num]
        yf = rt.traj_data[Y_ID][i][t_num]
        zf = rt.traj_data[Z_ID][i][t_num]
        pxf = rt.traj_data[PX_ID][i][t_num]
        pyf = rt.traj_data[PY_ID][i][t_num]
        pzf = rt.traj_data[PZ_ID][i][t_num]
        rf = sqrt(xf^2 + yf^2 + zf^2)
        energy_inf = (pxf^2 + pyf^2 + pzf^2) / 2 - Z / rf

        if energy_inf < 0
            asym_filter_flag[i] = true
            continue
        end

        p_inf = sqrt(2 * energy_inf)
        l = cross([xf, yf, zf], [pxf, pyf, pzf])
        a = cross([pxf, pyf, pzf], l) .- (Z / rf) .* (xf, yf, zf)
        p_inf_vec = (p_inf .* cross(l, a) .- a) .* (p_inf / (1 + p_inf^2 * norm(l)^2))
        asym_px_data[i] = p_inf_vec[1]
        asym_py_data[i] = p_inf_vec[2]
        asym_pz_data[i] = p_inf_vec[3]
    end

    return asym_px_data, asym_py_data, asym_pz_data, asym_filter_flag
end


function get_average_p_ctmc(rt)
    return rt.pmd_p_total ./ rt.pmd_total_w
end


function add_to_pmd(rt, trajs_num, t_num, Z, traj_filter_flag)
    asym_px_data, asym_py_data, asym_pz_data, asym_filter_flag = calculate_asymptotic_momentum(rt, trajs_num, t_num, Z)
    p_axis = rt.p_min: rt.p_delta: rt.p_max

    for i = 1: trajs_num
        if traj_filter_flag[i] == true
            continue
        end
        idx = Int64((asym_px_data[i] - rt.p_min) ÷ rt.p_delta)
        idy = Int64((asym_py_data[i] - rt.p_min) ÷ rt.p_delta)
        idz = Int64((asym_pz_data[i] - rt.p_min) ÷ rt.p_delta)
        if  idx < 1 || idx > length(p_axis) ||
            idy < 1 || idy > length(p_axis) ||
            idz < 1 || idz > length(p_axis)
            continue
        end
        if asym_filter_flag[i] == true
            continue
        end
        rt.px_final[idx] += rt.weight_cc[i]
        rt.py_final[idy] += rt.weight_cc[i]
        rt.pxy_final[idx, idy] += rt.weight_cc[i]
        rt.pxz_final[idx, idz] += rt.weight_cc[i]

        rt.pmd_p_total[1] += asym_px_data[i] * rt.weight_cc[i]
        rt.pmd_p_total[2] += asym_py_data[i] * rt.weight_cc[i]
        rt.pmd_p_total[3] += asym_pz_data[i] * rt.weight_cc[i]
        rt.pmd_total_w += rt.weight_cc[i]
    end
end

function add_to_hhg(rt, trajs_num, t_num, Δt, tid_cc, traj_filter_flag)
    combine_trajs_data(PX_ID, rt.ax_data, rt.traj_data, traj_filter_flag, rt.weight_cc, tid_cc, trajs_num, t_num, Δt)
    combine_trajs_data(PY_ID, rt.ay_data, rt.traj_data, traj_filter_flag, rt.weight_cc, tid_cc, trajs_num, t_num, Δt)
    combine_trajs_data(PZ_ID, rt.az_data, rt.traj_data, traj_filter_flag, rt.weight_cc, tid_cc, trajs_num, t_num, Δt)
end

function ctmc_get_hhg_spectrum(rt, E_data, tmax, t_num, ts, Δt, ω, tau; max_display_rate = 10)
    # HHG
    hhg_delta_k = 2pi / t_num / Δt
    hhg_k_linspace = [hhg_delta_k * (i - 1) for i = 1: t_num]
    spectrum_range = 1: Int64(floor(ω * max_display_rate / hhg_delta_k) + 1)
    hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
    hhg_windows_data = hhg_windows_f.(ts, tau, tmax)
    shg_id = Int64(floor((ω * 2) ÷ hhg_delta_k) + 1)
    base_id = Int64(floor(ω ÷ hhg_delta_k) + 1)

    hhg_spectrum_x = fft(rt.ax_data .* hhg_windows_data)
    hhg_spectrum_y = fft(rt.ay_data .* hhg_windows_data)

    p1 = plot(hhg_k_linspace[spectrum_range] / ω,
        [norm.(hhg_spectrum_x)[spectrum_range]],
        yscale=:log10,
        ylimit=(1e-7, 1e3))
    plot!(p1, [1, 1], [1e-10, 1e10])
    plot!(p1, [2, 2], [1e-10, 1e10])
    plot!(p1, [3, 3], [1e-10, 1e10])
    p1

    p2 = plot(hhg_k_linspace[spectrum_range] / ω, 
        [norm.(hhg_spectrum_x)[spectrum_range] .* 2e2],
        yscale=:log10,
        xlabel="N Times of ωfs",
        ylabel="Yield (arb.u.)",
        title="HHG Spectrum in Direction of z-axis",
        labels=["τ1" "τ2" "τ3" "τ4" "τ5"],
        guidefont=Plots.font(14, "Times"),
        tickfont=Plots.font(14, "Times"),
        titlefont=Plots.font(18, "Times"),
        legendfont=Plots.font(10, "Times"),
        margin = 5 * Plots.mm,
        ylimit=(1e-7, 1e3),
        xticks=0:1:max_display_rate)
    
    # id1 = Int64((2 * (nc1 ÷ 2 + 0) * π / ω) ÷ Δt)
    # id2 = Int64((2 * (nc1 ÷ 2 + 0.5) * π / ω) ÷ Δt)
    # e_fft_1 = fft(E_data[1] .* [i >= id1 for i = 1: t_num])
    # e_fft_2 = fft(E_data[1] .* [i >= id2 for i = 1: t_num])
    # plot([E_data[1] .* [i >= id1 for i = 1: t_num] E_data[1] .* [i >= id2 for i = 1: t_num]] )
    # tmpp = plot([norm.(e_fft_1)[1:100] norm.(e_fft_2)[1:100]], yscale=:log10)
    # plot!(tmpp, [base_id, base_id], [1e-2,1e2])
    # rad2deg(angle.(e_fft_1)[shg_id])
    # rad2deg(angle.(e_fft_2)[shg_id])
    # (id2 - id1) * Δt
    # pi / ω

    # fft(E_data[1])
    # barp = plot(norm.(fft(E_data[1]))[1:100], yscale=:log10)
    # plot!(barp, [base_id, base_id], [1e-2,1e2])
    
    return p1, p2, hhg_spectrum_x, hhg_spectrum_y, base_id, hhg_k_linspace
end


function trajs_analyse(rt, E_data, trajs_num, tmax, t_num, ts, Δt, ω, tau, tid_cc, traj_filter_flag, nc1; max_display_rate = 10)
    # HHG
    hhg_delta_k = 2pi / t_num / Δt
    hhg_k_linspace = [hhg_delta_k * (i - 1) for i = 1: t_num]
    spectrum_range = 1: Int64(floor(ω * max_display_rate / hhg_delta_k) + 1)
    hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t > tmin && t < tmax)
    hhg_windows_data = hhg_windows_f.(ts, 0, tmax)
    shg_id = Int64(floor((ω * 2) / hhg_delta_k)) + 1
    base_id = Int64(floor(ω / hhg_delta_k)) + 1
    # shg_id = (base_id - 1) * 2 + 1
    println("shg_id = $shg_id, base_id = $base_id")

    ax_data_single = zeros(Float64, t_num)
    shg_yield_data = zeros(ComplexF64, t_num)
    for i = 1: trajs_num
        if traj_filter_flag[i] == true
            continue
        end
        ax_data_single = traj_get_acc(PX_ID, rt.traj_data[PX_ID][i], tid_cc[i], t_num, Δt)
        # shg_yield_data[i] = fft(E_data[1] .* (rt.traj_data[PX_ID][i] .!= 0.0))[shg_id] #.* rt.weight_cc[i]
        shg_yield_data[i] = fft(ax_data_single)[shg_id] .* rt.weight_cc[i] * exp(-1im * deg2rad(-3.0))
    end

    norm_fig = plot([norm.(shg_yield_data) .* 1e3], labels=["|F(2ω)|(t)" "E_x(t)"])
    phase_fig = scatter(angle.(shg_yield_data), norm.(shg_yield_data),
        proj = :polar,
        markerstrokewidth = 0,
        markeralpha = 0.5,
        markersize = 1)

    # filtered_ids = [id for id = 1: trajs_num if traj_filter_flag[id] == true]
    # scatter!(norm_fig, filtered_ids, zeros(length(filtered_ids)))

    id1 = Int64((tau + (nc1 - 1 + 0.5) * π / ω) ÷ Δt + 1)
    id2 = Int64((tau + (nc1 - 1 + 1.5) * π / ω) ÷ Δt + 1)
    # scatter!(norm_fig, [id1 id2], [0, 0], marker='x')
    traj1 = (rt.traj_data[X_ID][id1] .!= 0.0) .* (rt.traj_data[X_ID][id1] .- rt.traj_data[X_ID][id1][tid_cc[id1]]) .* 0.5e-3
    traj2 = (rt.traj_data[X_ID][id2] .!= 0.0) .* (rt.traj_data[X_ID][id2] .- rt.traj_data[X_ID][id2][tid_cc[id2]]) .* 0.5e-3
    # plot!(norm_fig, traj1)
    # plot!(norm_fig, traj2)

    ax_data_single_1 = traj_get_acc(PX_ID, rt.traj_data[PX_ID][id1], tid_cc[id1], t_num, Δt)
    ax_data_single_2 = traj_get_acc(PX_ID, rt.traj_data[PX_ID][id2], tid_cc[id2], t_num, Δt)
    tmp1 = fft(ax_data_single_1 .* hhg_windows_data)
    tmp2 = fft(ax_data_single_2 .* hhg_windows_data)
    p3 = plot(hhg_k_linspace[spectrum_range], [norm.(tmp1 .+ tmp2)[spectrum_range] norm.(tmp1)[spectrum_range] norm.(tmp2)[spectrum_range]], yscale=:log10)
    plot!(p3, [shg_id - 1, shg_id - 1] .* hhg_delta_k, [1e-3, 1e1])
    plot!(p3, [base_id - 1, base_id - 1] .* hhg_delta_k, [1e-3, 1e1])


    p4 = plot()
    ff(x) = sign(x) * log(1.0 + abs(x))
    for i = 3: nc1 - 1 - 3
        id1 = Int64((tau + 2 * (i + 0.25) * π / ω) ÷ Δt + 1)
        id2 = Int64((tau + 2 * (i + 0.75) * π / ω) ÷ Δt + 1)

        o1 = rt.traj_data[X_ID][id1][tid_cc[id1]]
        o2 = rt.traj_data[Y_ID][id1][tid_cc[id1]]
        if sqrt(o1.^2 + o2.^2) <= 1e2
            plot!(p4, (rt.traj_data[X_ID][id1]),
                (rt.traj_data[Y_ID][id1]),
                xlimit=(-0.5e3, 0.5e3), ylimit=(-0.5e3, 0.5e3))
        end

        o1 = rt.traj_data[X_ID][id2][tid_cc[id2]]
        o2 = rt.traj_data[Y_ID][id2][tid_cc[id2]]
        if sqrt(o1.^2 + o2.^2) <= 1e2
            plot!(p4, (rt.traj_data[X_ID][id2]),
                (rt.traj_data[Y_ID][id2]),
                xlimit=(-0.5e3, 0.5e3), ylimit=(-0.5e3, 0.5e3))
        end
    end

    exit_point_norm = zeros(Float64, trajs_num)
    for i = 1: trajs_num
        if traj_filter_flag[i] == true
            continue
        end
        o1 = rt.traj_data[X_ID][i][tid_cc[i]]
        o2 = rt.traj_data[Y_ID][i][tid_cc[i]]
        exit_point_norm[i] = sqrt(o1 ^ 2 + o2 ^ 2)
    end
    p5 = plot(exit_point_norm)

    id1 = Int64((tau + (nc1 - 1 + 0.5) * π / ω) ÷ Δt + 2)
    id2 = Int64((tau + (nc1 - 1 + 1.5) * π / ω) ÷ Δt + 1)
    interval = 4
    rgg1 = id1 - interval * 2: interval: id1 + interval * 2
    rgg2 = id2 - interval * 2: interval: id2 + interval * 2
    p6 = scatter(angle.(shg_yield_data[[rgg1; rgg2]]),
        norm.(shg_yield_data[[rgg1; rgg2]]),
        proj = :polar, ylimit=(1.5e-4, 1.9e-4),
        yticks=[1.6e-4, 1.8e-4],
        markerstrokewidth = 0,
        markeralpha = 0.8,
        markersize = 5,
        markershape = :circle)

    
    # T_piece = 50
    # ft_ana_1 = zeros(ComplexF64, t_num)
    # ft_ana_2 = zeros(ComplexF64, t_num)
    # for i = 1: t_num - Int64(T_piece ÷ Δt + 1) - 1
    #     t_start = tau + (i - 1) * Δt
    #     tmp1 = fft(ax_data_single_1 .* hhg_windows_f.(ts, t_start, t_start + T_piece))[shg_id * 2]
    #     tmp2 = fft(ax_data_single_2 .* hhg_windows_f.(ts, t_start, t_start + T_piece))[shg_id * 2]
    #     ft_ana_1[i] = tmp1
    #     ft_ana_2[i] = tmp2
    # end

    traj_data = [E_data[1] traj1 traj2 ax_data_single_1 ax_data_single_2]
    return norm_fig, phase_fig, p6, shg_yield_data[[rgg1; rgg2]], traj_data
end


function trajs_pdd_analyse(rt, tid_cc, traj_filter_flag, tau_fs, nc, ω_fs, Δt)

    # dmin = -200.0
    # dmax = 200.0
    dmin = -0.6
    dmax = 0.6
    ddelta = (dmax - dmin) / 500
    displace_x_range = dmin: ddelta: dmax
    pdd_matrix = zeros(Float64, rt.t_num, length(displace_x_range))

    id1 = Int64((tau_fs + (nc - 1 + 0.5) * π / ω_fs) ÷ Δt + 1)
    id2 = Int64((tau_fs + (nc - 1 + 1.5) * π / ω_fs) ÷ Δt + 1)
    
    for i = 1: rt.trajs_num
        if traj_filter_flag[i] == true
            continue
        end
        for j = tid_cc[i]: rt.t_num
            displacement = rt.traj_data[PY_ID][i][j] - rt.traj_data[PY_ID][i][tid_cc[i]]
            displace_id = Int64((displacement - dmin) ÷ ddelta)
            if  displace_id < 1 || displace_id > length(displace_x_range)
                continue
            end
            pdd_matrix[j, displace_id] += rt.weight_cc[i]
            if tid_cc[i] >= id1 && tid_cc[i] <= id2
                pdd_matrix[j, displace_id] += 1e3
            end
        end
    end
    return pdd_matrix
end