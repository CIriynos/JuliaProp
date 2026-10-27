######################################

# index 1 ~ 6 --> px, py, pz, x, y, z
const PX_ID = 1; const PY_ID = 2; const PZ_ID = 3;
const X_ID = 4;  const Y_ID = 5;  const Z_ID = 6;
p2r(id) = id + 3
r2p(id) = id - 3


# function RK4_procedure(
#     F1::Function, i::Int64,
#     traj_data, start_point_cc::Vector{Vector{Float64}},
#     tid::Int64, t_num::Int64, Δt::Float64,
#     E_data::Vector{Vector{Float64}}, E_hf_data::Vector{Vector{Float64}}, Z::Float64,
#     filter_threshold::Float64
# )
#     r = [0.0, 0.0, 0.0]
#     ks = [zeros(Float64, 4) for _ = 1: 6]
#     filter_flag = false
#     # copy start point to the traj_data
#     for j = 1: 6
#         traj_data[j][i][tid] = start_point_cc[j][i]
#     end
#     # start RK4
#     for k = tid + 1: t_num
#         r[1] = traj_data[X_ID][i][k-1]
#         r[2] = traj_data[Y_ID][i][k-1]
#         r[3] = traj_data[Z_ID][i][k-1]

#         if sqrt(r[1]^2 + r[2]^2 + r[3]^2) < filter_threshold
#             filter_flag = true
#         end

#         # dp/dt = -E(t) - Zr/r^3
#         @fastmath @inbounds for j = PX_ID: PZ_ID
#             ks[j][1] = F1(E_data[j][k-1], Z, r[1], r[2], r[3], r[j])
#         end
#         @fastmath @inbounds for j = PX_ID: PZ_ID
#             ks[j][2] = F1(E_hf_data[j][k-1], Z,
#                 r[1] + 0.5Δt * ks[1][1], r[2] + 0.5Δt * ks[2][1],
#                 r[3] + 0.5Δt * ks[3][1], r[j] + 0.5Δt * ks[j][1])
#         end
#         @fastmath @inbounds for j = PX_ID: PZ_ID
#             ks[j][3] = F1(E_hf_data[j][k-1], Z,
#                 r[1] + 0.5Δt * ks[1][2], r[2] + 0.5Δt * ks[2][2],
#                 r[3] + 0.5Δt * ks[3][2], r[j] + 0.5Δt * ks[j][2])
#         end
#         @fastmath @inbounds for j = PX_ID: PZ_ID
#             ks[j][4] = F1(E_data[j][k], Z,
#                 r[1] + Δt * ks[1][3], r[2] + Δt * ks[2][3],
#                 r[3] + Δt * ks[3][3], r[j] + Δt * ks[j][3])
#         end

#         # dr/dt = p
#         @fastmath @inbounds for j = X_ID: Z_ID
#             ks[j][1] = traj_data[r2p(j)][i][k-1]
#         end
#         @fastmath @inbounds for j = X_ID: Z_ID
#             ks[j][2] = traj_data[r2p(j)][i][k-1] + 0.5Δt * ks[j][1]
#         end
#         @fastmath @inbounds for j = X_ID: Z_ID
#             ks[j][3] = traj_data[r2p(j)][i][k-1] + 0.5Δt * ks[j][2]
#         end
#         @fastmath @inbounds for j = X_ID: Z_ID
#             ks[j][4] = traj_data[r2p(j)][i][k-1] + Δt * ks[j][3]
#         end

#         # Get them together.
#         @fastmath @inbounds for j = 1: 6
#             traj_data[j][i][k] = traj_data[j][i][k-1] + (Δt / 6) * (ks[j][1] + 2*ks[j][2] + 2*ks[j][3] + ks[j][4])
#         end
#     end

#     return filter_flag
# end

function RK4_procedure(
    F1::Function, i::Int64,
    traj_data, start_point_cc::Vector{Vector{Float64}},
    tid::Int64, t_num::Int64, Δt::Float64,
    E_data::Vector{Vector{Float64}}, E_hf_data::Vector{Vector{Float64}}, Z::Float64,
    filter_threshold::Float64
)
    r = zeros(Float64, 3)
    r_stage = similar(r)
    ks = [zeros(Float64, 4) for _ = 1:6]
    filter_flag = false

    for j = 1:6
        traj_data[j][i][tid] = start_point_cc[j][i]
    end

    for k = (tid + 1):t_num
        for j = PX_ID:PZ_ID
            r[j] = traj_data[p2r(j)][i][k-1]
        end

        # 保留原有筛选逻辑：只记录标志，不中止传播。
        if sqrt(r[1]^2 + r[2]^2 + r[3]^2) < filter_threshold
            filter_flag = true
        end

        # 第一阶段：动量导数为加速度，位置导数为动量。
        for j = PX_ID:PZ_ID
            ks[j][1] = F1(E_data[j][k-1], Z, r[1], r[2], r[3], r[j])
            ks[p2r(j)][1] = traj_data[j][i][k-1]
        end

        # 第二、三阶段使用半步场，第四阶段使用下一整步场。
        for s = 2:4
            h_stage = s == 4 ? Δt : 0.5 * Δt
            field = s == 4 ? E_data : E_hf_data
            field_index = s == 4 ? k : k - 1

            # 中间位置由上一阶段的位置导数构造。
            for j = PX_ID:PZ_ID
                r_stage[j] = r[j] + h_stage * ks[p2r(j)][s-1]
            end

            for j = PX_ID:PZ_ID
                ks[j][s] = F1(field[j][field_index], Z,
                    r_stage[1], r_stage[2], r_stage[3], r_stage[j])

                # 中间动量由上一阶段的动量导数构造。
                ks[p2r(j)][s] =
                    traj_data[j][i][k-1] + h_stage * ks[j][s-1]
            end
        end

        for j = 1:6
            traj_data[j][i][k] = traj_data[j][i][k-1] + (Δt / 6) *
                (ks[j][1] + 2 * ks[j][2] + 2 * ks[j][3] + ks[j][4])
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

# function get_p_vec(E, pv, theta)
#     c = pv * sqrt(E[1]^2 + E[2]^2) / norm(E)
#     b = -c * E[2] * E[3] / (E[1]^2 + E[2]^2)
#     a = -(b * E[2] + c * E[3]) / E[1]
#     r2 = (a, b, c)
    
#     n = (E[2]*c - E[3]*b, E[3]*a - E[1]*c, E[1]*b - E[2]*a)
#     n = n ./ norm(n) .* sqrt(a^2 + b^2 + c^2)
#     res = r2 .* cos(theta) .+ n .* sin(theta)
#     return res
# end


function get_p_vec(E, pv, theta)
    E1, E2, E3 = E
    normE = norm(E)
    normE == 0 && return (zero(pv), zero(pv), zero(pv))  # 退化向量

    cθ = cos.(theta)   # 支持 theta 为标量或数组
    sθ = sin.(theta)

    r = sqrt(E1*E1 + E2*E2)

    # 构造垂直于 E 的标准正交基 (u0, v0)
    if r == 0
        # E 平行于 z 轴：任选 xy 平面的一组基
        u1, u2, u3 = 1.0, 0.0, 0.0
        v1, v2, v3 = 0.0, sign(E3), 0.0   # = e × u0
    else
        invr = 1 / r
        invN = 1 / normE
        u1 = -E1 * E3 * invr * invN
        u2 = -E2 * E3 * invr * invN
        u3 =  r * invN
        v1 =  E2 * invr
        v2 = -E1 * invr
        v3 =  0.0
    end

    return (pv * (u1*cθ + v1*sθ),
            pv * (u2*cθ + v2*sθ),
            pv * (u3*cθ + v3*sθ))
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


# function generate_start_point_random(trajs_num, t_num, pv_max, E_data, Ip, Z; min_threshold = 1e-8)
#     # Random Start-Point Generator
#     tid_cc = zeros(Int64, trajs_num)
#     p = 1
#     while p <= trajs_num
#         tid = Int64(rand(1: t_num))
#         e_norm = norm([E_data[1][tid], E_data[2][tid], E_data[3][tid]])
#         w = W.(e_norm, 0, Ip, Z)
#         if w > min_threshold
#             tid_cc[p] = tid
#             p += 1
#         end
#     end
#     # tid_cc = rand(1: t_num, trajs_num)
#     pv_cc = rand(trajs_num) * pv_max
#     theta_cc = rand(trajs_num) * 2pi
#     return tid_cc, pv_cc, theta_cc
# end

# 这个版本才是真正的随机抽样，上一个版本会导致抽样不均匀(更偏向小pv)
function generate_start_point_random(trajs_num, t_num, pv_max, E_data, Ip, Z; min_threshold = 1e-8)

    trajs_num >= 0 ||
        throw(ArgumentError("trajs_num must be nonnegative"))
    t_num >= 1 ||
        throw(ArgumentError("t_num must be positive"))
    isfinite(pv_max) && pv_max > 0 ||
        throw(ArgumentError("pv_max must be finite and positive"))

    tid_cc = zeros(Int64, trajs_num)
    pv_cc = zeros(Float64, trajs_num)
    theta_cc = zeros(Float64, trajs_num)
    trajs_num == 0 && return tid_cc, pv_cc, theta_cc

    # 保留原来的时间筛选条件。
    valid_tids = Int64[]
    for tid = 1:t_num
        E_norm = norm((
            E_data[1][tid], E_data[2][tid], E_data[3][tid]
        ))
        if W(E_norm, 0, Ip, Z) > min_threshold
            push!(valid_tids, tid)
        end
    end

    isempty(valid_tids) &&
        throw(ArgumentError("no ionization time satisfies min_threshold"))

    for i = 1: trajs_num
        # 在合格时刻中均匀、有放回抽样。
        tid_cc[i] = rand(valid_tids)

        # p1、p2 是局部横向正交基下的动量分量。
        while true
            p1 = pv_max * (2 * rand() - 1)
            p2 = pv_max * (2 * rand() - 1)
            pv = hypot(p1, p2)

            if pv <= pv_max
                pv_cc[i] = pv
                theta_cc[i] = mod(atan(p2, p1), 2pi)
                break
            end
        end
    end

    return tid_cc, pv_cc, theta_cc
end


function generate_start_point_random_xy_plane(trajs_num, t_num, pv_max, E_data, Ip, Z; min_threshold = 1e-8)
    trajs_num >= 0 ||
        throw(ArgumentError("trajs_num must be nonnegative"))
    t_num >= 1 ||
        throw(ArgumentError("t_num must be positive"))
    isfinite(pv_max) && pv_max > 0 ||
        throw(ArgumentError("pv_max must be finite and positive"))

    tid_cc = zeros(Int64, trajs_num)
    pv_cc = zeros(Float64, trajs_num)
    theta_cc = zeros(Float64, trajs_num)
    trajs_num == 0 && return tid_cc, pv_cc, theta_cc

    # 保留原来的时间筛选条件。
    valid_tids = Int64[]
    for tid = 1:t_num
        E_norm = norm((
            E_data[1][tid], E_data[2][tid], E_data[3][tid]
        ))
        if W(E_norm, 0, Ip, Z) > min_threshold
            push!(valid_tids, tid)
        end
    end

    isempty(valid_tids) &&
        throw(ArgumentError("no ionization time satisfies min_threshold"))

    for i = 1: trajs_num
        # 在合格时刻中均匀、有放回抽样。
        tid_cc[i] = rand(valid_tids)

        # p1、p2 是局部横向正交基下的动量分量。
        while true
            p1 = 0
            p2 = pv_max * (2 * rand() - 1)
            pv = hypot(p1, p2)

            if pv <= pv_max
                pv_cc[i] = pv
                theta_cc[i] = mod(atan(p2, p1), 2pi)
                break
            end
        end
    end

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

function generate_start_point_uniform_xy_plane(trajs_num, t_num, pv_max, E_data, Ip, Z; min_threshold = 1e-8)
    tid_cc = ones(Int64, trajs_num)
    pv_cc = zeros(Float64, trajs_num)
    theta_cc = zeros(Float64, trajs_num)

    # 保留原来的时间筛选条件。
    valid_tids = Int64[]
    for tid = 1: t_num
        E_norm = norm((E_data[1][tid], E_data[2][tid], E_data[3][tid]))
        if W(E_norm, 0, Ip, Z) > min_threshold
            push!(valid_tids, tid)
        end
    end

    pv_sample_num = floor(Int64, trajs_num / length(valid_tids))
    if pv_sample_num % 2 == 0
        pv_sample_num -= 1
    end
    pv_delta = 2 * pv_max / pv_sample_num
    pv_sample_list = ((-pv_sample_num ÷ 2): 1 : (pv_sample_num ÷ 2)) .* pv_delta
    shorten_len = length(valid_tids) * pv_sample_num

    println("pv_sample_num = $pv_sample_num")

    tid_cc[1: shorten_len] = repeat(valid_tids; inner = length(pv_sample_list))   # 每个 A 元素重复 length(B) 次
    pv_cc[1: shorten_len] = repeat(pv_sample_list; outer = length(valid_tids))   # 整个 B 重复 length(A) 次
    theta_cc[1: shorten_len] = [acos(sign(pv + 1e-5)) + pi/2 for pv in pv_cc[1: shorten_len]]
    pv_cc[1: shorten_len] = abs.(pv_cc[1: shorten_len])

    return tid_cc, pv_cc, theta_cc
end

function generate_start_point_uniform_xy_plane_m(
    trajs_num, t_num, pv_max, E_data, Ip, Z, max_m, m;
    min_threshold = 1e-8
)
    valid_tids = Int64[]
    for tid in 1:t_num
        E_norm = norm((E_data[1][tid], E_data[2][tid], E_data[3][tid]))
        if W(E_norm, 0, Ip, Z) > min_threshold
            push!(valid_tids, tid)
        end
    end

    L = length(valid_tids)
    L == 0 && error("No valid time steps found.")

    # 保留原来的初始化
    tid_cc   = ones(Int64, trajs_num) .* t_num 
    pv_cc    = zeros(Float64, trajs_num)
    theta_cc = zeros(Float64, trajs_num)

    B = div(trajs_num, L)
    B < 1 && error("trajs_num < length(valid_tids)：有效样本数必然超过 trajs_num，无解。")

    N_raw = floor(Int, trajs_num * max_m / L)
    N_cap = max_m * B
    N_max_odd = N_cap - (iseven(N_cap) ? 1 : 0)
    N = min(N_raw, N_max_odd)
    iseven(N) && (N -= 1)
    N = max(1, N)

    H = div(N - 1, 2)
    delta = H > 0 ? pv_max / H : 0.0
    pv_list = [k * delta for k in -H:H]     # 含 0，正负对称，长度 N

    start = div((m - 1) * N, max_m) + 1
    stop  = div(m * N, max_m)
    pv_block = (stop >= start) ? pv_list[start:stop] : Float64[]
    n_pv = length(pv_block)

    shorten_len = L * n_pv
    @assert shorten_len <= trajs_num

    if n_pv > 0
        tid_cc[1:shorten_len] = repeat(valid_tids; inner = n_pv)
        pv_raw = repeat(pv_block; outer = L)
        theta_cc[1:shorten_len] = [acos(sign(pv + 1e-5)) + pi/2 for pv in pv_raw]
        pv_cc[1:shorten_len]    = abs.(pv_raw)
    end

    # 长度仍为 trajs_num；额外返回有效长度
    return tid_cc, pv_cc, theta_cc, shorten_len
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
        E_ti = (E_data[1][tid_cc[i]], E_data[2][tid_cc[i]], E_data[3][tid_cc[i]]) .+ 1e-50
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

# function calculate_asymptotic_momentum(rt, trajs_num, t_num, Z)
#     asym_px_data = zeros(Float64, trajs_num)
#     asym_py_data = zeros(Float64, trajs_num)
#     asym_pz_data = zeros(Float64, trajs_num)
#     asym_filter_flag = zeros(Bool, trajs_num)

#     for i = 1: trajs_num
#         xf = rt.traj_data[X_ID][i][t_num]
#         yf = rt.traj_data[Y_ID][i][t_num]
#         zf = rt.traj_data[Z_ID][i][t_num]
#         pxf = rt.traj_data[PX_ID][i][t_num]
#         pyf = rt.traj_data[PY_ID][i][t_num]
#         pzf = rt.traj_data[PZ_ID][i][t_num]
#         rf = sqrt(xf^2 + yf^2 + zf^2)
#         energy_inf = (pxf^2 + pyf^2 + pzf^2) / 2 - Z / rf

#         if energy_inf < 0
#             asym_filter_flag[i] = true
#             continue
#         end

#         p_inf = sqrt(2 * energy_inf)
#         l = cross([xf, yf, zf], [pxf, pyf, pzf])
#         a = cross([pxf, pyf, pzf], l) .- (Z / rf) .* (xf, yf, zf)
#         p_inf_vec = (p_inf .* cross(l, a) .- a) .* (p_inf / (1 + p_inf^2 * norm(l)^2))
#         asym_px_data[i] = p_inf_vec[1]
#         asym_py_data[i] = p_inf_vec[2]
#         asym_pz_data[i] = p_inf_vec[3]
#     end

#     return asym_px_data, asym_py_data, asym_pz_data, asym_filter_flag
# end

# AI修改后的，可以适用于不同的Z，原先的版本只能用于Z = 1
function calculate_asymptotic_momentum(rt, trajs_num, t_num, Z)
    isfinite(Z) && Z >= 0 ||
        throw(ArgumentError("Z must be finite and nonnegative"))

    asym_px_data = zeros(Float64, trajs_num)
    asym_py_data = zeros(Float64, trajs_num)
    asym_pz_data = zeros(Float64, trajs_num)
    asym_filter_flag = zeros(Bool, trajs_num)

    for i = 1:trajs_num
        r = [rt.traj_data[X_ID][i][t_num],
             rt.traj_data[Y_ID][i][t_num],
             rt.traj_data[Z_ID][i][t_num]]

        p = [rt.traj_data[PX_ID][i][t_num],
             rt.traj_data[PY_ID][i][t_num],
             rt.traj_data[PZ_ID][i][t_num]]

        if !all(isfinite, r) || !all(isfinite, p)
            asym_filter_flag[i] = true
            continue
        end

        if Z == 0
            # 无库仑力时，外场结束后的动量保持不变。
            p_inf_vec = p
        else
            rf = norm(r)
            if !isfinite(rf) || rf == 0
                asym_filter_flag[i] = true
                continue
            end

            energy_inf = dot(p, p) / 2 - Z / rf
            if !isfinite(energy_inf) || energy_inf < 0
                asym_filter_flag[i] = true
                continue
            end

            l = cross(r, p)

            # 严格径向入射会撞上点库仑奇点，
            # 不能直接映射为正常的出射态。
            if all(iszero, l) && dot(r, p) < 0
                asym_filter_flag[i] = true
                continue
            end

            a = cross(p, l) .- (Z / rf) .* r
            p_inf = sqrt(2 * energy_inf)

            denominator = Z^2 + p_inf^2 * dot(l, l)
            p_inf_vec = (p_inf / denominator) .*
                (p_inf .* cross(l, a) .- Z .* a)

            if !all(isfinite, p_inf_vec)
                asym_filter_flag[i] = true
                continue
            end
        end

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
        # if traj_filter_flag[i] == true
        #     continue
        # end
        # # idx = Int64((asym_px_data[i] - rt.p_min) ÷ rt.p_delta)
        # # idy = Int64((asym_py_data[i] - rt.p_min) ÷ rt.p_delta)
        # # idz = Int64((asym_pz_data[i] - rt.p_min) ÷ rt.p_delta)
        # if  idx < 1 || idx > length(p_axis) ||
        #     idy < 1 || idy > length(p_axis) ||
        #     idz < 1 || idz > length(p_axis)
        #     continue
        # end
        # if asym_filter_flag[i] == true
        #     continue
        # end
        
        # 实际上，并不需要筛掉traj_filter_flag的那一部分，而只需要筛掉根本出不来的轨线。(?存疑)
        # 我认为traj_filter_flag也得筛，尤其是非常靠近零点的那些轨线，因为根本算不准，会乱飞
        if asym_filter_flag[i] || traj_filter_flag[i]
            continue
        end
        
        px = asym_px_data[i]
        py = asym_py_data[i]
        pz = asym_pz_data[i]
        
        # NaN、Inf 不能转换成有效索引。
        if !(isfinite(px) && isfinite(py) && isfinite(pz))
            continue
        end
        
        # 以实际生成的轴端点为准。
        p_lo = first(p_axis)
        p_hi = last(p_axis)
        if !(p_lo <= px <= p_hi && p_lo <= py <= p_hi && p_lo <= pz <= p_hi)
            continue
        end
        
        idx = round(Int, (px - p_lo) / rt.p_delta) + 1
        idy = round(Int, (py - p_lo) / rt.p_delta) + 1
        idz = round(Int, (pz - p_lo) / rt.p_delta) + 1

        rt.px_final[idx] += rt.weight_cc[i]
        rt.py_final[idy] += rt.weight_cc[i]
        rt.pz_final[idz] += rt.weight_cc[i]
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




# ============================================================
# 新增：固定 Δt 的 CTMC 一阶 / 二阶轨线展开
#
# 沿用 ctmc_rt，不新增数据结构，不修改任何原有函数。
# 新增函数：
#   _ctmc_expansion_rhs!      内部辅助函数：展开方程右端
#   RK4_perturb_procedure     单条轨线，接口对应 RK4_procedure
#   ctmc_perturb_mainloop     并行主循环，接口对应 ctmc_mainloop
#
# 模型必须与传入的 F1 一致：
#   F1(E, Z, x, y, z, rj) = -E - Z*rj/(x^2+y^2+z^2+softening)^(3/2)
# softening 是分母中直接相加的常数，不是该常数的平方。
# 默认 1e-5 对应当前调用脚本；纯库仑模型请设 softening=0.0。
# 本实现使用该模型的解析导数，不适用于其他形式的 F1。
#
# E_data / E_hf_data：总场，沿用原调用中的参数。
# E_ref_data / E_ref_hf_data：参考场，分别对应整步与半步。
# 微扰场直接由两者相减得到，不构造场函数、不插值、不调整 Δt。
#
# 展开约定：r = r0 + u + w，p = p0 + du + dw。
# w 已包含二阶 Taylor 系数 1/2，不需要再乘 1/2。
# 三组内部状态均按 [px, py, pz, x, y, z] 排列：
#   y = [p0; r0; du; u; dw; w]（一阶时仅前 12 项）。
# 所有阶数共用 start_point_cc 中的初值，修正项初值均为零。
# 初值和权重保持固定，不对隧穿出口、初始动量或电离权重作展开。
# ============================================================

function _ctmc_expansion_rhs!(dy, y, F1, E_data, E_ref_data, k, Z, softening)
    x, yy, z = y[X_ID], y[Y_ID], y[Z_ID]
    s = x*x + yy*yy + z*z + softening
    s > 0 || throw(DomainError(s, "reference trajectory reached the Coulomb singularity"))
    invR3 = s^(-1.5)
    invR5 = invR3 / s
    ru = x*y[10] + yy*y[11] + z*y[12]

    for j = PX_ID:PZ_ID
        rj = y[p2r(j)]
        uj = y[6+p2r(j)]
        δE = E_data[j][k] - E_ref_data[j][k]
        # 参考轨线：仍通过调用者提供的 F1 计算力。
        dy[j] = F1(E_ref_data[j][k], Z, x, yy, z, rj)
        dy[p2r(j)] = y[j]
        # 一阶：du' = D F_C(r0) u - δE。
        dy[6+j] = Z*(-uj*invR3 + 3*ru*rj*invR5) - δE
        dy[6+p2r(j)] = y[6+j]
    end
    length(y) == 12 && return nothing

    rw = x*y[16] + yy*y[17] + z*y[18]
    uu = y[10]^2 + y[11]^2 + y[12]^2
    qfac = 1.5*uu*invR5 - 7.5*ru^2*invR5/s
    for j = PX_ID:PZ_ID
        rj = y[p2r(j)]
        uj = y[6+p2r(j)]
        wj = y[12+p2r(j)]
        # 二阶：dw' = D F_C(r0) w + (1/2) D² F_C(r0)[u,u]。
        dy[12+j] = Z*(-wj*invR3 + 3*rw*rj*invR5 +
                       3*ru*uj*invR5 + qfac*rj)
        dy[12+p2r(j)] = y[12+j]
    end
    return nothing
end

"""
    RK4_perturb_procedure(F1, i, traj_data, start_point_cc, tid, t_num, Δt,
        E_data, E_hf_data, E_ref_data, E_ref_hf_data, Z, filter_threshold;
        order=2, softening=1e-5)

新增：用固定步长 RK4 推进第 i 条展开轨线。
order=1 记录 [p0+du; r0+u]；order=2 记录 [p0+du+dw; r0+u+w]。
记录位置为 traj_data[j][i][k]，与原函数完全相同。
返回 Bool：记录的近似轨线是否在任一网格点进入 filter_threshold。
仍继续推进被标记的轨线；起点以前的数据置零，避免复用 rt 时残留旧轨线。
通常通过 ctmc_perturb_mainloop 调用，由主循环统一检查参数和数组长度。
"""
function RK4_perturb_procedure(
    F1, i, traj_data, start_point_cc, tid, t_num, Δt,
    E_data, E_hf_data, E_ref_data, E_ref_hf_data, Z, filter_threshold;
    order=2, softening=1e-5
)
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2"))
    y = zeros(Float64, 6*(order+1))
    ks = [similar(y) for _ = 1:4]
    tmp = similar(y)
    for j = 1:6
        fill!(traj_data[j][i], 0.0)
        y[j] = start_point_cc[j][i]
        traj_data[j][i][tid] = y[j]
    end
    all(isfinite, y) || throw(ArgumentError("initial state must be finite"))
    filter_flag = sum(abs2, @view y[X_ID:Z_ID]) < filter_threshold^2

    for k = tid+1:t_num
        # 整步 k-1，半步 k-1，半步 k-1，整步 k。
        # 每一阶段同时推进参考轨线及修正项的位置和动量。
        _ctmc_expansion_rhs!(ks[1], y, F1, E_data, E_ref_data, k-1, Z, softening)
        @. tmp = y + 0.5Δt*ks[1]
        _ctmc_expansion_rhs!(ks[2], tmp, F1, E_hf_data, E_ref_hf_data, k-1, Z, softening)
        @. tmp = y + 0.5Δt*ks[2]
        _ctmc_expansion_rhs!(ks[3], tmp, F1, E_hf_data, E_ref_hf_data, k-1, Z, softening)
        @. tmp = y + Δt*ks[3]
        _ctmc_expansion_rhs!(ks[4], tmp, F1, E_data, E_ref_data, k, Z, softening)
        @. y = y + (Δt/6)*(ks[1]+2*ks[2]+2*ks[3]+ks[4])
        all(isfinite, y) || error("nonfinite expansion state: trajectory $i, time index $k")

        for j = 1:6
            traj_data[j][i][k] = y[j] + y[6+j] + (order == 2 ? y[12+j] : 0.0)
        end
        r2 = traj_data[X_ID][i][k]^2 + traj_data[Y_ID][i][k]^2 + traj_data[Z_ID][i][k]^2
        filter_flag |= r2 < filter_threshold^2
    end
    return filter_flag
end

"""
    ctmc_perturb_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt,
        E_data, E_hf_data, Z, filter_threshold;
        E_ref_data, E_ref_hf_data, order=2, softening=1e-5)

新增：ctmc_mainloop 的一阶 / 二阶展开版本，保留其全部位置参数。
仅新增参考场、阶数及软化常数这几个关键字参数。

直接更新原 rt.traj_data，返回 Vector{Bool} 格式的 traj_filter_flag。
rt.start_point_cc、rt.weight_cc 和已有的 PMD/HHG 累加结果均不修改。
因此原来的 add_to_pmd、add_to_hhg、轨线分析等调用可以沿用。
每次只记录所选阶数的完整轨线；一阶、二阶若需同时保留，请使用两个 rt。
参考轨线及修正项仅在积分时保存在单轨线工作区，不额外分配全体轨线数组。
"""
function ctmc_perturb_mainloop(
    F1, rt::ctmc_rt, t_num, trajs_num, tid_cc, Δt,
    E_data, E_hf_data, Z, filter_threshold;
    E_ref_data, E_ref_hf_data, order=2, softening=1e-5
)
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2"))
    isfinite(Δt) && Δt > 0 || throw(ArgumentError("Δt must be finite and positive"))
    isfinite(softening) && softening >= 0 || throw(ArgumentError("softening must be finite and nonnegative"))
    isfinite(Z) && Z >= 0 || throw(ArgumentError("Z must be finite and nonnegative"))
    isfinite(filter_threshold) && filter_threshold >= 0 ||
        throw(ArgumentError("filter_threshold must be finite and nonnegative"))
    1 <= t_num <= rt.t_num || throw(ArgumentError("t_num exceeds rt allocation"))
    0 <= trajs_num <= rt.trajs_num || throw(ArgumentError("trajs_num exceeds rt allocation"))
    length(tid_cc) >= trajs_num || throw(DimensionMismatch("tid_cc is too short"))
    all(i -> tid_cc[i] isa Integer && 1 <= tid_cc[i] <= t_num, 1:trajs_num) ||
        throw(ArgumentError("each start time index must lie in 1:t_num"))
    for (data, n) in ((E_data, t_num), (E_hf_data, t_num-1),
                      (E_ref_data, t_num), (E_ref_hf_data, t_num-1))
        length(data) == 3 && all(x -> length(x) >= n, data) ||
            throw(DimensionMismatch("field arrays must contain 3 components with enough time samples"))
    end

    traj_filter_flag = zeros(Bool, trajs_num)
    Threads.@threads for i = 1:trajs_num
        traj_filter_flag[i] = RK4_perturb_procedure(
            F1, i, rt.traj_data, rt.start_point_cc, tid_cc[i], t_num, Δt,
            E_data, E_hf_data, E_ref_data, E_ref_hf_data, Z, filter_threshold;
            order=order, softening=softening)
    end
    return traj_filter_flag
end

# ============================================================
# 对应 2026_10_21_ctmc_one_cycle_burst.jl 的调用方式：
#
# create_ctmc_rt、generate_start_point_random、update_start_point 不变。
# 将循环内原来的 ctmc_mainloop 调用替换为：
#
# traj_filter_flag = JuliaProp.ctmc_perturb_mainloop(
#     F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf,
#     Z, filter_threshold;
#     E_ref_data=Et_data_no_thz,
#     E_ref_hf_data=Et_data_hf_no_thz,
#     order=2,                 # 改为 1 即记录一阶近似轨线
#     softening=1e-5)          # 与脚本中 F1 分母的常数一致
#
# 后面的 add_to_pmd、add_to_hhg、calculate_asymptotic_momentum 等不变。
# 若直接 include 到 Main，请去掉 JuliaProp.；若模块已 export 新接口，
# 也可以按原有调用风格直接使用 ctmc_perturb_mainloop。
# 本节应追加到项目模块内的 ctmc.jl 末尾。
# ============================================================
