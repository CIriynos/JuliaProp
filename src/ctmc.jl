# index 1 ~ 6 --> px, py, pz, x, y, z
const PX_ID = 1; const PY_ID = 2; const PZ_ID = 3;
const X_ID = 4;  const Y_ID = 5;  const Z_ID = 6;
p2r(id) = id + 3
r2p(id) = id - 3


function RK4_procedure(i::Int64,
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
    tmp = norm(E)^2 + 1e-10
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
    F = abs(E) + 1e-10
    D = (4 * Z^3 / F)
    return (F * D^2 / (8 * pi * Z)) * exp(-2 * (2 * Ip) ^ (3/2) / (3 * F))
end

function W1(p⊥, E, Ip)
    F = abs(E) + 1e-10
    return sqrt(2 * Ip) / (pi * F) * exp(- (p⊥^2) * sqrt(2 * Ip) / F)
end

W(E, p⊥, Ip, Z) = W0(E, Ip, Z) * W1(p⊥, E, Ip)

function combine_trajs_data(ID, combined_data, traj_data, traj_filter_flag, weight_cc, tid_cc, trajs_num, t_num, delta_t)
    ai::Float64 = 0.0
    for i = 1: trajs_num
        if traj_filter_flag[i] == false
            for j = tid_cc[i]: t_num
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
    for j = tid: t_num
        if j - 2 >= 1 && j + 2 <= t_num
            acc_data[j] = four_order_difference(traj[P_ID][i], j, delta_t)
        elseif j == 2 || j == t_num - 1
            acc_data[j] = two_order_difference(traj[P_ID][i], j, delta_t)
        elseif j == 1
            acc_data[j] = (traj[P_ID][i][2] - traj[P_ID][i][1]) / delta_t
        elseif j == t_num
            acc_data[j] = (traj[P_ID][i][t_num] - traj[P_ID][i][t_num-1]) / delta_t
        end
    end
    return acc_data
end

# The function of Movement
F1(E, Z, x, y, z, rj) = -E - Z * rj / (x^2 + y^2 + z^2 + 1e-5) ^ (3 / 2)