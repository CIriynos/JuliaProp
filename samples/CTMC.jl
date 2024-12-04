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
println("Number of Threads: $(Threads.nthreads())")


# index 1 ~ 6 --> px, py, pz, x, y, z
const PX_ID = 1; const PY_ID = 2; const PZ_ID = 3;
const X_ID = 4;  const Y_ID = 5;  const Z_ID = 6;
p2r(id) = id + 3
r2p(id) = id - 3

# The function of Movement
F1(E, Z, x, y, z, rj) = -E - Z * rj / (x^2 + y^2 + z^2 + 0.1) ^ (3 / 2)

function RK4_procedure(i::Int64,
    traj_data::Vector{Vector{Vector{Float64}}}, start_point_cc::Vector{Vector{Float64}},
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
    return -Ip*E[1] / norm(E)^2, -Ip*E[2] / norm(E)^2, -Ip*E[3] / norm(E)^2
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

function add_to_pi_data(P_ID, pi_data, traj_data, traj_filter_flag, weight_cc, tid_cc, trajs_num, t_num)
    for i = 1: trajs_num
        if traj_filter_flag[i] == false
            for j = tid_cc[i]: t_num
                pi_data[j] += traj_data[P_ID][i][j] * weight_cc[i]
            end
        end
    end
end





#################################

# Core Parameters
Z = 1.0
Ip = 0.5
ϵ = 0.3

# Laser Parameters 1
ω1 = 0.05693
E1 = 0.0533761 * 1 / sqrt(ϵ^2 + 1)
nc1 = 15
Tp1 = 2pi * nc1 / ω1
tau1 = 500

# Laser Parameters 2
ω2 = ω1
E2 = 0.0533761 * ϵ / sqrt(ϵ^2 + 1)
nc2 = 15
Tp2 = 2pi * nc2 / ω2
tau2 = 500

# THZ Parameters
ω_thz = ω1 / 50
E3 = 0.000017 * 0
E_DC = 0.000017 * 0
Tp3 = 2pi / ω_thz
tau3 = -500

# CTMC Parameters
Δt = 0.05
t_min = 0.0
t_max = max(Tp1, Tp2) * 1.5
pv_max = 1.0
trajs_num = 20000
t_num = Int64((t_max - t_min) ÷ Δt)
ts = 0: Δt: t_max - Δt
ts_hf = Δt/2: Δt: t_max - Δt/2

# CTMC PMD Parameters
p_min = -1.0
p_max = 1.0
p_delta = 0.005
p_axis = p_min: p_delta: p_max

# Laser & THz Waveform
E_laser_1(t) = E1 * sin(ω1 * (t - tau1) / 2 / nc1)^2 * cos(ω1 * (t - tau1)) * (t > tau1 && t < tau1 + Tp1)
E_laser_2(t) = E2 * sin(ω2 * (t - tau2) / 2 / nc2)^2 * sin(ω2 * (t - tau2)) * (t > tau2 && t < tau2 + Tp2)
E_thz(t) = E_DC + E3 * sin(ω_thz * (t - tau3)) * (t > tau3 && t < tau3 + Tp3)
E_x(t) = E_laser_1(t) + E_thz(t)
E_y(t) = E_laser_2(t)
E_z(t) = 0.0

E_data = [E_x.(ts), E_y.(ts), E_z.(ts)]
E_hf_data = [E_x.(ts_hf), E_y.(ts_hf), E_z.(ts)]
E_norm_data = sqrt.(E_x.(ts).^2 .+ E_y.(ts).^2 .+ E_z.(ts).^2) .+ 1e-10
E_norm_data_no_THz = sqrt.((E_x.(ts) - E_thz.(ts)).^2 .+ E_y.(ts).^2) .+ 1e-10
plot([E_data[1] E_data[2] E_thz.(ts) .* 1e2])

# prepare for PMD / HHG
px_final = zeros(Float64, length(p_axis))
py_final = zeros(Float64, length(p_axis))
pz_final = zeros(Float64, length(p_axis))
pxy_final = zeros(Float64, (length(p_axis), length(p_axis)))
pxz_final = zeros(Float64, (length(p_axis), length(p_axis)))
px_data = zeros(Float64, t_num)
py_data = zeros(Float64, t_num)
pz_data = zeros(Float64, t_num)
ax_data = zeros(Float64, t_num)

# RK4 Data Allocation Reusable (It speet a lot of time)
trajs_num = 20000
traj_data = [[zeros(Float64, t_num) for i = 1: trajs_num] for j = 1: 6]



for m = 1: 5

# Random Start-Point Generator
tid_cc = zeros(Int64, trajs_num)
p = 1
while p <= trajs_num
    tid = Int64(rand(1: t_num))
    w = W.(E_norm_data[tid], 0, Ip, Z)
    if w > 1e-8
        tid_cc[p] = tid
        p += 1
    end
end
# tid_cc = rand(1: t_num, trajs_num)
pv_cc = rand(trajs_num) * pv_max
theta_cc = rand(trajs_num) * 2pi


# Preparation for Start Point 
weight_cc = W.(E_norm_data[tid_cc], pv_cc, Ip, Z)
start_point_cc = [zeros(Float64, trajs_num) for i = 1: 6]
for i = 1: trajs_num
    E_ti = (E_data[1][tid_cc[i]], E_data[2][tid_cc[i]], E_data[3][tid_cc[i]])
    exit_point = get_exit_point(E_ti, Ip)
    start_point_cc[X_ID][i] = exit_point[1]
    start_point_cc[Y_ID][i] = exit_point[2]
    start_point_cc[Z_ID][i] = exit_point[3]

    pv_vec = get_p_vec(E_ti, pv_cc[i], theta_cc[i])
    start_point_cc[PX_ID][i] = pv_vec[1]
    start_point_cc[PY_ID][i] = pv_vec[2]
    start_point_cc[PZ_ID][i] = pv_vec[3]
end


# RK4 Mainloop
filter_threshold = 5.0
traj_filter_flag = zeros(Bool, trajs_num)
Threads.@threads for i = 1: trajs_num
    flag = RK4_procedure(i, traj_data, start_point_cc, tid_cc[i], t_num, Δt, E_data, E_hf_data, Z, filter_threshold)
    traj_filter_flag[i] = flag
end
println("RK4 ended.")


# Add to PMD
for i = 1: trajs_num
    idx = Int64((traj_data[PX_ID][i][t_num] - p_min) ÷ p_delta)
    idy = Int64((traj_data[PY_ID][i][t_num] - p_min) ÷ p_delta)
    idz = Int64((traj_data[PZ_ID][i][t_num] - p_min) ÷ p_delta)
    if  idx < 1 || idx > length(p_axis) ||
        idy < 1 || idy > length(p_axis) ||
        idz < 1 || idz > length(p_axis)
        continue
    end
    px_final[idx] += weight_cc[i]
    py_final[idy] += weight_cc[i]
    pxy_final[idx, idy] += weight_cc[i]
    pxz_final[idx, idz] += weight_cc[i]
end

# Add to HHG
add_to_pi_data(PX_ID, px_data, traj_data, traj_filter_flag, weight_cc, tid_cc, trajs_num, t_num)
add_to_pi_data(PY_ID, py_data, traj_data, traj_filter_flag, weight_cc, tid_cc, trajs_num, t_num)
add_to_pi_data(PZ_ID, pz_data, traj_data, traj_filter_flag, weight_cc, tid_cc, trajs_num, t_num)


end

# ####

# HHG
ax_data = get_derivative_two_order(px_data, Δt)
ay_data = get_derivative_two_order(py_data, Δt)
plot(norm.(fft(px_data))[1:500], yscale=:log10)

hhg_delta_k = 2pi / t_num / Δt
hhg_k_linspace = [hhg_delta_k * i for i = 1: t_num]
hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(ts, 0, t_max)
hhg_spectrum_x = fft(ax_data .* hhg_windows_data)
hhg_spectrum_y = fft(ay_data .* hhg_windows_data)

plot(hhg_k_linspace[1:500] / ω1,
    [norm.(hhg_spectrum_x)[1:500] norm.(hhg_spectrum_y)[1:500]],
    yscale=:log10,
    ylimit=(1e-5, 1e3))

# plot(hhg_k_linspace[1:500] / ω1, norm.(hhg_spectrum_y)[1:500], yscale=:log10)


Plots.heatmap(p_axis, p_axis, pxy_final, color=:jet1, size=(500, 420))