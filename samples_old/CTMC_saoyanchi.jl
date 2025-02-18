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


######################################

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
    for j = tid: t_num
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

# The function of Movement
F1(E, Z, x, y, z, rj) = -E - Z * rj / (x^2 + y^2 + z^2 + 1e-5) ^ (3 / 2)











#################################

hhgs = []
e_figs = []
shg_id::Float64 = 0.0
hhg_k_linspace = 1:10
spectrum_range = 1:10

for tau_id = 1: 16

# Core Parameters
Z = 1.0
Ip = 0.5
ϵ = 0.5

# Laser Parameters 1
ω1 = 0.057
E1 = 0.053 * 1 * 1 / sqrt(ϵ^2 + 1)
nc1 = 15
Tp1 = 2pi * nc1 / ω1
tau1 = 500

# Laser Parameters 2
ω2 = ω1
E2 = 0.053 * 1 * ϵ / sqrt(ϵ^2 + 1)
nc2 = 15
Tp2 = 2pi * nc2 / ω2
tau2 = 500

# THZ Parameters
ω_thz = ω1 / 25
E3 = 0.00002 * 1
E_DC = 0.00002 * 1
Tp3 = 2pi / ω_thz
tau_lst = [tau1 - 2pi/ω_thz; range(tau1 - 2pi/ω_thz + tau1, tau1 + nc1*2pi/ω1 - tau1, 14); tau1 + nc1*2pi/ω1]
tau3 = tau_lst[tau_id]

# CTMC Parameters
Δt = 0.2
t_min = 0.0
t_max = Tp1 + 2 * tau1
t_num = Int64((t_max - t_min) ÷ Δt)
ts = 0: Δt: t_max - Δt
ts_hf = Δt/2: Δt: t_max - Δt/2
pv_max = 1.0

# CTMC PMD Parameters
p_min = -1.0
p_max = 1.0
p_delta = 0.005
p_axis = p_min: p_delta: p_max

# CTMC Filter Parameters
filter_threshold = 2.0


# Laser & THz Waveform
E_laser_1(t) = E1 * sin(ω1 * (t - tau1) / 2 / nc1)^2 * sin(ω1 * (t - tau1)) * (t > tau1 && t < tau1 + Tp1)
E_laser_2(t) = E2 * sin(ω2 * (t - tau2) / 2 / nc2)^2 * cos(ω2 * (t - tau2)) * (t > tau2 && t < tau2 + Tp2)
E_thz(t) = E_DC + E3 * sin(ω_thz * (t - tau3)) * (t > tau3 && t < tau3 + Tp3)
E_x(t) = E_laser_1(t) + E_thz(t)
E_y(t) = E_laser_2(t)
E_z(t) = 1e-20

E_data = [E_x.(ts) .+ 1e-20, E_y.(ts) .+ 1e-20, E_z.(ts) .+ 1e-20]
E_hf_data = [E_x.(ts_hf), E_y.(ts_hf), E_z.(ts)]
E_norm_data = sqrt.(E_x.(ts).^2 .+ E_y.(ts).^2 .+ E_z.(ts).^2) .+ 1e-20
E_norm_data_no_thz = sqrt.((E_laser_1.(ts)).^2 .+ E_y.(ts).^2) .+ 1e-20
push!(e_figs, plot([E_data[1] E_data[2] E_thz.(ts) .* 1e2]))


# RK4 Data Allocation Reusable (It spends a lot of time)
trajs_num = t_num
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
# phase_fig = plot()


m = 1
# for m = 1: 1


# Uniform Start-Point Generator
theta_m_list = [0; range(0, 2*pi, 6); range(0, 2*pi, 12); range(0, 2*pi, 18)]
pv_m_list = [1e-5; ones(6) .* 0.2; ones(12) .* 0.4; ones(18) .* 0.8]
tid_cc = 1: trajs_num
pv_cc = ones(trajs_num) .* pv_m_list[m]
theta_cc = ones(trajs_num) .* theta_m_list[m]


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
traj_filter_flag = zeros(Bool, trajs_num)
Threads.@threads for i = 1: trajs_num
    flag = RK4_procedure(i, traj_data, start_point_cc, tid_cc[i], t_num, Δt, E_data, E_hf_data, Z, filter_threshold)
    traj_filter_flag[i] = flag
end
println("RK4 ended.")


# Calculate the Asymptotic Momentum
asym_px_data = zeros(Float64, trajs_num)
asym_py_data = zeros(Float64, trajs_num)
asym_pz_data = zeros(Float64, trajs_num)
asym_filter_flag = zeros(Bool, trajs_num)
for i = 1: trajs_num
    xf = traj_data[X_ID][i][t_num]
    yf = traj_data[Y_ID][i][t_num]
    zf = traj_data[Z_ID][i][t_num]
    pxf = traj_data[PX_ID][i][t_num]
    pyf = traj_data[PY_ID][i][t_num]
    pzf = traj_data[PZ_ID][i][t_num]
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


# Add to PMD
for i = 1: trajs_num
    idx = Int64((asym_px_data[i] - p_min) ÷ p_delta)
    idy = Int64((asym_py_data[i] - p_min) ÷ p_delta)
    idz = Int64((asym_pz_data[i] - p_min) ÷ p_delta)
    if  idx < 1 || idx > length(p_axis) ||
        idy < 1 || idy > length(p_axis) ||
        idz < 1 || idz > length(p_axis)
        continue
    end
    if asym_filter_flag[i] == true
        continue
    end
    px_final[idx] += weight_cc[i]
    py_final[idy] += weight_cc[i]
    pxy_final[idx, idy] += weight_cc[i]
    pxz_final[idx, idz] += weight_cc[i]
end


# Add to HHG
combine_trajs_data(PX_ID, ax_data, traj_data, traj_filter_flag, weight_cc, tid_cc, trajs_num, t_num, Δt)
combine_trajs_data(PY_ID, ay_data, traj_data, traj_filter_flag, weight_cc, tid_cc, trajs_num, t_num, Δt)
combine_trajs_data(PZ_ID, az_data, traj_data, traj_filter_flag, weight_cc, tid_cc, trajs_num, t_num, Δt)


# # Trajs Analyse
# # HHG
# hhg_delta_k = 2pi / t_num / Δt
# hhg_k_linspace = [hhg_delta_k * (i - 1) for i = 1: t_num]
# spectrum_range = 1: Int64(floor(ω1 * 10 / hhg_delta_k))
# hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
# hhg_windows_data = hhg_windows_f.(ts, 10, t_max - 10)
# shg_id = Int64(ceil((ω1 * 2) ÷ hhg_delta_k) + 2)
# base_id = Int64(ceil(ω1 ÷ hhg_delta_k) + 2)

# ax_data_single = zeros(Float64, t_num)
# shg_yield_data = zeros(ComplexF64, t_num)
# for i = 1: trajs_num
#     if traj_filter_flag[i] == true
#         continue
#     end
#     ax_data_single = traj_get_acc(PX_ID, traj_data[PX_ID][i], tid_cc[i], t_num, Δt)
#     shg_yield_data[i] = fft(ax_data_single .* hhg_windows_data)[shg_id] .* weight_cc[i]
# end

# norm_fig = plot([norm.(shg_yield_data) E_data[1] .* 1e-3],
#     labels=["|F(2ω)|(t)" "E_x(t)"])
# phase_fig = scatter!(phase_fig, angle.(shg_yield_data) .+ pi, norm.(shg_yield_data) .^ 4,
#     proj = :polar,
#     markerstrokewidth = 0,
#     markeralpha = 0.5,
#     markersize = 1)
# filtered_ids = [id for id = 1: trajs_num if traj_filter_flag[id] == true]
# scatter!(norm_fig, filtered_ids, zeros(length(filtered_ids)))

# id1 = Int64((2 * (nc1 ÷ 2 + 0) * π / ω1) ÷ Δt)
# id2 = Int64((2 * (nc1 ÷ 2 + 0.5) * π / ω1) ÷ Δt)
# scatter!(norm_fig, [id1 id2], [0, 0])
# ax_data_single_1 = traj_get_acc(PX_ID, traj_data[PX_ID][id1], tid_cc[id1], t_num, Δt)
# ax_data_single_2 = traj_get_acc(PX_ID, traj_data[PX_ID][id2], tid_cc[id2], t_num, Δt)
# tmp1 = fft(ax_data_single_1 .* hhg_windows_data)[shg_id]
# tmp2 = fft(ax_data_single_2 .* hhg_windows_data)[shg_id]
# plot(ax_data_single_2)


# rad2deg(angle(tmp1))
# rad2deg(angle(tmp2))
# end



############ <===3

# HHG
hhg_delta_k = 2pi / t_num / Δt
hhg_k_linspace = [hhg_delta_k * (i - 1) for i = 1: t_num]
spectrum_range = 1: Int64(floor(ω1 * 8 / hhg_delta_k))
hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(ts, 0, t_max)
shg_id = Int64(ceil((ω1 * 2) ÷ hhg_delta_k) + 1)
base_id = Int64(ceil(ω1 ÷ hhg_delta_k) + 1)

hhg_spectrum_x = fft(ax_data .* hhg_windows_data)
hhg_spectrum_y = fft(ay_data .* hhg_windows_data)
plot([ax_data E_data[1] .* 1e-2])

# plot(hhg_k_linspace[spectrum_range] / ω1,
#     [norm.(hhg_spectrum_x)[spectrum_range] norm.(hhg_spectrum_y)[spectrum_range]],
#     yscale=:log10,
#     ylimit=(1e-12, 1e3))

p = plot(hhg_k_linspace[spectrum_range] / ω1,
    [norm.(hhg_spectrum_x)[spectrum_range]],
    yscale=:log10,
    ylimit=(1e-7, 1e3))
plot!(p, [1, 1], [1e-10, 1e10])
plot!(p, [2, 2], [1e-10, 1e10])
plot!(p, [3, 3], [1e-10, 1e10])
p

p2 = plot(hhg_k_linspace[spectrum_range] / ω1, 
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
    ylimit=(1e-6, 1e3))

push!(hhgs, norm.(hhg_spectrum_x))

# plot!(p2, [shg_id, shg_id], [1e-10, 1e10])
# plot!(p2, [base_id, base_id], [1e-10, 1e10])


# id1 = Int64((2 * (nc1 ÷ 2 + 0) * π / ω1) ÷ Δt)
# id2 = Int64((2 * (nc1 ÷ 2 + 0.5) * π / ω1) ÷ Δt)
# e_fft_1 = fft(E_data[1] .* [i >= id1 for i = 1: t_num])
# e_fft_2 = fft(E_data[1] .* [i >= id2 for i = 1: t_num])
# plot([E_data[1] .* [i >= id1 for i = 1: t_num] E_data[1] .* [i >= id2 for i = 1: t_num]] )
# tmpp = plot([norm.(e_fft_1)[1:100] norm.(e_fft_2)[1:100]], yscale=:log10)
# plot!(tmpp, [base_id, base_id], [1e-2,1e2])
# rad2deg(angle.(e_fft_1)[shg_id])
# rad2deg(angle.(e_fft_2)[shg_id])
# (id2 - id1) * Δt
# pi / ω1

# fft(E_data[1])
# barp = plot(norm.(fft(E_data[1]))[1:100], yscale=:log10)
# plot!(barp, [base_id, base_id], [1e-2,1e2])

# # PMD
# Plots.heatmap(p_axis, p_axis, pxy_final, color=:jet1, size=(500, 420))


end

p2 = plot(hhg_k_linspace[spectrum_range] / ω1, 
    1e1 * [hhgs[1][spectrum_range] hhgs[6][spectrum_range] hhgs[8][spectrum_range] hhgs[12][spectrum_range] hhgs[16][spectrum_range]],
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
    ylimit=(1e-6, 1e3))

hhg_ctmc = [hhgs[i][49] for i = 1: 16]
plot(hhg_ctmc)