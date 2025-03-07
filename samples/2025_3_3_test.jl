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
using SpecialFunctions
println("Number of Threads: $(Threads.nthreads())")




ω = 0.05693
ω_thz = ω / 20
E0 = 0.0533
Ip = 0.5
K0 = 0.5 / ω
nc = 15
Δt = 1.0
E0_thz = 0.00002 * 0
E_dc = 0.0005
tau_thz = 0
tau_fs = 500
Tp = 2*nc*pi/ω
eps = 0.5


Et_fs_envelop(t) = E0 * exp(-(t - tau_fs - Tp/2)^2/(Tp/2)^2)^2
Et_fs_x(t) = 1 / sqrt(eps^2 + 1) * Et_fs_envelop(t) * cos(ω*(t-tau_fs) + 0.5pi)
Et_fs_y(t) = eps / sqrt(eps^2 + 1) * Et_fs_envelop(t) * cos(ω*(t-tau_fs) + pi)
Et_thz(t) = (E0_thz) * sin(ω_thz*(t-tau_thz)/2)^2 * sin(ω_thz*(t-tau_thz)) * (t-tau_thz > 0 && t-tau_thz <(2*pi/ω_thz))
E_applied(t) = Et_thz(t) + E_dc
Et_x(t) = Et_fs_x(t) + E_applied(t)
Et_y(t) = Et_fs_y(t)
F_act(t) = sqrt(Et_x(t) ^ 2 + Et_y(t) ^ 2)

l0(t) = sqrt(Et_fs_x(t)^2 + Et_fs_y(t)^2)
l1(t) = Et_fs_x(t) / (sqrt(Et_fs_x(t)^2 + Et_fs_y(t)^2))
l2(t) = Et_fs_y(t)^2 / (2 * sqrt(Et_fs_x(t)^2 + Et_fs_y(t)^2)^3)
F_asymp(t) = l0(t) + l1(t) * E_applied(t) + l2(t) * E_applied(t)^2

tmax = tau_fs + Tp + 500
ts = 0: Δt: tmax
plot(ts, [Et_x.(ts) Et_y.(ts)])
plot(ts, [F_act.(ts) F_asymp.(ts)])

W(F) = 4/F * exp(-2/(3*F))

function W_taylor(E1x, E1y, E2)
    E1 = sqrt(E1x^2 + E1y^2)
    α1 = (2 - 3 * E1) / (3 * E1^2)
    α2 = (9 * E1^2 - 12 * E1 + 2) / (9 * E1^4)
    β1 = E1x / E1
    β2 = E1y^2 / (2 * E1^3)
    return W(E1) * (1 + α1*β1*E2 + (α1*β2 + α2*β1^2) * E2^2)
end

err=1e-40
plot([W.(F_act.(ts)) W_taylor.(Et_fs_x.(ts), Et_fs_y.(ts), E_applied.(ts))])

dd = sum((W.(F_act.(ts)) .- W_taylor.(Et_fs_x.(ts), Et_fs_y.(ts), E_applied.(ts))) .^ 2) / sum(W.(F_act.(ts)).^2)
ee = abs((sum(abs.(W.(F_act.(ts)))) - sum(abs.(W_taylor.(Et_fs_x.(ts), Et_fs_y.(ts), E_applied.(ts)))))) / sum(abs.(W.(F_act.(ts))))

function W_taylor_demo(E1x, E1y)
    E1 = @. sqrt(E1x^2 + E1y^2)
    α1 = @. (2 - 3 * E1) / (3 * E1^2)
    α2 = @. (9 * E1^2 - 12 * E1 + 2) / (9 * E1^4)
    β1 = @. E1x / E1
    β2 = @. E1y^2 / (2 * E1^3)
    l = length(E1x)
    plot([ones(l)  α1.*β1 (α1.*β2.+α2.*β1.^2)])
end

W_taylor_demo(0.01: 0.001: 0.05, 0.05)