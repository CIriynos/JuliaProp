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

# 10_31_CTMC.jl
# 1. 写一份CTMC

function RK4_one_step(y1, y2, f1, f2, t, h)
    @fastmath k1 = f1(t, y2)
    @fastmath k2 = f1(t + h/2, y2 + h * k1 / 2)
    @fastmath k3 = f1(t + h/2, y2 + h * k2 / 2)
    @fastmath k4 = f1(t + h, y2 + h * k3)
    @fastmath y1 = y1 + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    @fastmath k1 = f2(t, y1)
    @fastmath k2 = f2(t + h/2, y1 + h * k1 / 2)
    @fastmath k3 = f2(t + h/2, y1 + h * k2 / 2)
    @fastmath k4 = f2(t + h, y1 + h * k3)
    @fastmath y2 = y2 + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return y1, y2
end

# function RK4_2f(y1_init, y2_init, f1, f2, t0, h, steps)
#     y1 = y1_init
#     y2 = y2_init
#     for i = 2: steps
#         t = t0 + h * (i - 1)
#         @fastmath k1 = f1(t, y2)
#         @fastmath k2 = f1(t + h/2, y2 + h * k1 / 2)
#         @fastmath k3 = f1(t + h/2, y2 + h * k2 / 2)
#         @fastmath k4 = f1(t + h, y2 + h * k3)
#         @fastmath y1 = y1 + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

#         @fastmath k1 = f2(t, y1)
#         @fastmath k2 = f2(t + h/2, y1 + h * k1 / 2)
#         @fastmath k3 = f2(t + h/2, y1 + h * k2 / 2)
#         @fastmath k4 = f2(t + h, y1 + h * k3)
#         @fastmath y2 = y2 + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
#     end
# end

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


# Define Basic Parameters
ω = 0.05
ω_thz = ω / 10
tau = 0
E0 = 0.05
gamma = ω / E0
nc = 15
Ec = 0.0002
Δt = 0.05
tmin = 0
tmax = 2*nc*pi/ω
pv_max = 1.0

E_thz(t) = (Ec) * sin(ω_thz*(t-tau)/2)^2 * sin(ω_thz*(t-tau)) * (t-tau > 0 && t-tau <(2*pi/ω_thz)) * flap_top_windows_f(t, 0, 2*nc*pi/ω, 1/8)
E_fs(t) = E0 * sin(ω*t/2/nc)^2 * cos(ω*t) * (t < 2*nc*pi/ω)
E(t) = E_fs(t) + E_thz(t)
t_linspace = 0: Δt: 2*nc*pi/ω
# plot(t_linspace, [E_thz.(t_linspace) * 100 E_fs.(t_linspace)])


# Classical Movement
# dp/dt = -F(t) - Zr/r^3;  dr/dt = p
Ex(t) = 0.0; Ey(t) = 0.0; Ez(t) = E(t);
f1_x(t, x) = -Ex(t) - 0 / (x^2 + 1e-6); f2_x(t, px) = px;
f1_y(t, y) = -Ey(t) - 0 / (y^2 + 1e-6); f2_y(t, py) = py;
f1_z(t, z) = -Ez(t) - 0 / (z^2 + 1e-6); f2_z(t, pz) = pz;

# Monte Carlo Parameters
ctmc_t_num = 1000
ctmc_pv_num = 1
ctmc_pv_phi_num = 1
traject_t0_id_list = Int64.(floor.(rand(ctmc_t_num) * tmax))
traject_pv_id_list = rand(ctmc_pv_num) * 2*pv_max .- pv_max
traject_pv_phi_id_list = rand(ctmc_pv_phi_num) * 2*2pi .- pv_max


# filter those (nearly) impossible trajs
traj_collection_x = Tuple{Int64, Float64, Float64}[]
traj_collection_y = Tuple{Int64, Float64, Float64}[]
traj_collection_z = Tuple{Int64, Float64, Float64}[]
FILTER_THRESHOLD = 1e-8
for t0 in traject_t0_id_list
    for pv in traject_pv_id_list
        if W(E.(t0), pv, 0.5, 1) < FILTER_THRESHOLD
            continue
        end
        for phi in traject_pv_phi_id_list
            # For Linearly Polarized Laser
            px0 = pv * sin(phi)
            py0 = pv * cos(phi)
            z0 = -0.5 / E(t0)
            push!(traj_collection_x, (t0, px0, 0.0))
            push!(traj_collection_y, (t0, py0, 0.0))
            push!(traj_collection_z, (t0, 0.0, z0))
        end
    end
end

# CTMC Mainloop
traj_nums = length(traj_collection_x)
t_num = length(t_linspace)
p_final_record = []
for i = 1: traj_nums
    t0_id = traj_collection_x[i][1]
    px = traj_collection_x[i][2]
    py = traj_collection_y[i][2]
    pz = 0.0
    x = 0.0
    y = 0.0
    z = traj_collection_z[i][3]
    for j = t0_id: t_num
        crt_t = t_linspace[j]
        px, x = RK4_one_step(px, x, f1_x, f2_x, crt_t, Δt)
        py, y = RK4_one_step(py, y, f1_y, f2_y, crt_t, Δt)
        pz, z = RK4_one_step(pz, z, f1_z, f2_z, crt_t, Δt)
    end
    push!(p_final_record, (px, py, pz))
end

# scatter(p_final_record)

l1 = [px for (px, py, pz) in p_final_record]
l2 = [py for (px, py, pz) in p_final_record]
l3 = [pz for (px, py, pz) in p_final_record]
scatter(l2)

# Check Monte Carlo
# fig = plot()
# plot!(fig, t_linspace, E.(t_linspace))
# l1 = [t0 for (t0, x0) in traj_collection_x]
# l2 = [0 for (t0, x0) in traj_collection_x]
# scatter!(fig, l1, l2, mc=:red, ms=2, ma=0.5)