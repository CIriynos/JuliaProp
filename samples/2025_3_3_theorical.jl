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
ω_thz = ω / 2
E0 = 0.0533
Ip = 0.5
K0 = 0.5 / ω
nc = 12
Δt = 0.1  
E0_thz = 0.00002
E_dc = 0.00002
tau_fs = 2000
Tp = 2*nc*pi/ω
eps = 0.5
tau_thz = tau_fs + Tp / 2 - (pi / ω_thz)

tmax = tau_fs + Tp + 2000
ts = 0: Δt: tmax

# Et_fs_envelop(t) = E0 * exp(-(t - tau_fs - Tp/2)^2/(Tp/2)^2)^2
Et_fs_envelop(t) = E0 * sin(ω*(t-tau_fs)/2/nc)^2 * ((t-tau_fs) > 0 && t-tau_fs < (2*nc*pi/ω))
Et_fs_x(t) = 1 / sqrt(eps^2 + 1) * Et_fs_envelop(t) * cos(ω*(t-tau_fs) + 0.5pi)
Et_fs_y(t) = eps / sqrt(eps^2 + 1) * Et_fs_envelop(t) * cos(ω*(t-tau_fs) + pi)
Et_thz(t) = (E0_thz) * sin(ω_thz*(t-tau_thz)/2)^2 * sin(ω_thz*(t-tau_thz)) * (t-tau_thz > 0 && t-tau_thz <(2*pi/ω_thz))
E_applied(t) = (Et_thz(t) + E_dc) * flap_top_windows_f(t, 0, tmax, 1/8)
Et_x(t) = (Et_fs_x(t) + E_applied(t))
Et_y(t) = Et_fs_y(t)
F_total(t) = sqrt(Et_x(t) ^ 2 + Et_y(t) ^ 2) .+ 1e-50
F_1(t) = sqrt(Et_fs_x(t) ^ 2 + Et_fs_y(t) ^ 2) .+ 1e-50

l0(t) = sqrt(Et_fs_x(t)^2 + Et_fs_y(t)^2)
l1(t) = Et_fs_x(t) / (sqrt(Et_fs_x(t)^2 + Et_fs_y(t)^2))
l2(t) = Et_fs_y(t)^2 / (2 * sqrt(Et_fs_x(t)^2 + Et_fs_y(t)^2)^3)
F_asymp(t) = l0(t) + l1(t) * E_applied(t) + l2(t) * E_applied(t)^2

N = length(ts)
plot(ts, [Et_x.(ts) Et_y.(ts) E_applied.(ts) .* 500])
plot(ts, [F_total.(ts) F_asymp.(ts)])

W(F) = 4/F * exp(-2/(3*F))

W_γ1(E1, E1x, E1y) = ((2 - 3 * E1) / (3 * E1^2)) * (E1x / E1)
W_γ2(E1, E1x, E1y) = ((2 - 3 * E1) / (3 * E1^2)) * (E1y^2 / (2 * E1^3)) + ((9 * E1^2 - 12 * E1 + 2) / (9 * E1^4)) * (E1x / E1)^2

function W_taylor(E1x, E1y, E2)
    E1 = sqrt(E1x^2 + E1y^2)
    α1 = (2 - 3 * E1) / (3 * E1^2)
    α2 = (9 * E1^2 - 12 * E1 + 2) / (9 * E1^4)
    β1 = E1x / E1
    β2 = E1y^2 / (2 * E1^3)
    return W(E1) * (1 + α1*β1*E2 + (α1*β2 + α2*β1^2) * E2^2)
end


plot([W.(F_total.(ts)) W_taylor.(Et_fs_x.(ts), Et_fs_y.(ts), E_applied.(ts))])
# dd = sum((W.(F_total.(ts)) .- W_taylor.(Et_fs_x.(ts), Et_fs_y.(ts), E_applied.(ts))) .^ 2) / sum(W.(F_act.(ts)).^2)
# ee = abs((sum(abs.(W.(F_total.(ts)))) - sum(abs.(W_taylor.(Et_fs_x.(ts), Et_fs_y.(ts), E_applied.(ts)))))) / sum(abs.(W.(F_act.(ts))))


# for FFT
hhg_delta_k = 2pi / N / Δt
hhg_k_linspace = [hhg_delta_k * (i - 1) for i = 1: N]
spectrum_range = 1: Int64(floor(ω * 8 / hhg_delta_k) + 1)
hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(ts, 0, tmax)
shg_id = Int64(floor((ω * 2) ÷ hhg_delta_k) + 1)
base_id = Int64(floor(ω ÷ hhg_delta_k) + 1)


# 1 (biggest)
Wt_data = W.(F_1.(ts))
tmp1 = @. Et_fs_x(ts) * exp(-im * 2 * ω * ts)
H1 = reverse(get_integral(tmp1, Δt))
h1_data = @. W_γ1(F_1(ts), Et_fs_x(ts), Et_fs_y(ts)) * Wt_data * H1
Gw1 = fft(h1_data)
E_input = E_applied.(reverse(ts))
res1 = conv(E_input, h1_data)
plot(norm.(res1))
plot(hhg_k_linspace[spectrum_range] ./ ω, norm.(Gw1)[spectrum_range], yscale=:log10)


# 2
Wt_data = W.(F_1.(ts))
Int_Wt_data = get_integral(Wt_data, Δt) .* flap_top_windows_f.(ts, 0, tmax, 1/4)
h2_data = @. exp(-im * 2 * ω * ts) * (Int_Wt_data)
Gw2 = fft(h2_data)
E_input = E_applied.(reverse(ts))
res2 = conv(E_input, h2_data)
plot(norm.(res2))
plot(hhg_k_linspace[spectrum_range] ./ ω, [norm.(Gw2)[spectrum_range]], yscale=:log10)


# 3
Wt_data = W.(F_1.(ts))
tmp3 = @. Et_fs_x(ts) * exp(-im * 2 * ω * ts)
H3 = reverse(get_integral(tmp3, Δt))
h3_data = @. W_γ2(F_1(ts), Et_fs_x(ts), Et_fs_y(ts)) * Wt_data * H3
Gw3 = fft(h3_data)
E_input = E_applied.(reverse(ts))
res3 = conv(E_input .^ 2, h3_data)
plot(norm.(res3))
plot(hhg_k_linspace[spectrum_range] ./ ω, norm.(Gw3)[spectrum_range], yscale=:log10)


res = @. res1 + res2 + res3
plot(norm.(res))
# plot(hhg_k_linspace[spectrum_range] ./ ω, [norm.(Gw1)[spectrum_range] norm.(Gw2)[spectrum_range] norm.(Gw3)[spectrum_range]], yscale=:log10)

plot(1:50, [-20 .* log10.(norm.(Gw1)[1] ./ norm.(Gw1)[1:50] .+ 1e-50)])



# function bell_polynomial(m::Int, k::Int, a::Vector)
#     if m == 0 && k == 0
#         return 1
#     elseif k == 0 || k > m
#         return 0
#     end

#     t = m - k + 1
#     if length(a) < t
#         error("a must have at least $(t) elements")
#     end

#     solutions = generate_solutions(m, k, t)
#     m_fact = factorial(m)
#     total = 0

#     for sol in solutions
#         denom = 1
#         product_a = 1
#         for i in 1:t
#             ji = sol[i]
#             denom *= factorial(ji)
#             product_a *= (a[i] / factorial(i))^ji
#         end
#         total += m_fact / denom * product_a
#     end

#     return total
# end

# function generate_solutions(m::Int, k::Int, t::Int)
#     solutions = Vector{Vector{Int}}()
#     current = zeros(Int, t)

#     function backtrack(i::Int, rem_k::Int, rem_m::Int)
#         if i == 0
#             if rem_k == 0 && rem_m == 0
#                 push!(solutions, copy(current))
#             end
#             return
#         end
#         max_j = min(rem_k, rem_m ÷ i)
#         for j in 0:max_j
#             current[i] = j
#             backtrack(i-1, rem_k - j, rem_m - i*j)
#         end
#         current[i] = 0  # 回溯
#     end

#     backtrack(t, k, m)
#     return solutions
# end
# a = [1, 2, 3, 4]  # 对应 a₁=1, a₂=2, a₃=3, a₄=4
# result = bell_polynomial(5, 2, a)


# # ChatGPT's Taylor Method with exp
# F0 = 0.05


# # 计算 ln(W(F)) 关于 F 在 F0 附近的泰勒展开系数
# function Ak(k, F0)
#     if k == 0
#         return log(4) - log(F0) - 2 / (3 * F0)
#     else
#         return (-1)^k * ((1/(k * F0^k)) - (2 / (3 * F0^(k + 1))))
#     end
# end

# W_asymp(F) = exp(Ak(0, F0) + Ak(1, F0) * (F - F0) + Ak(2, F0) * (F - F0)^2 + Ak(3, F0) * (F - F0)^3)

# function Cm(m, A_list)
#     cm = 0.0
#     for k = 1: m
#         cm += (1 / (factorial(k))) * bell_polynomial(m, k, A_list[2:(m-k+1)+1])
#     end
#     cm *= exp(A_list[1]) # A0
#     return cm
# end

# n = 3
# Ak_list = [Ak(k, F0) for k = 0: n]
# Cm_list = [Cm(m, Ak_list) for m = 1: n]

# function W_asymp_2(F, n)
#     ans = W.(0.05)
#     for m = 1: n - 1
#         ans += Cm_list[m] * (F - F0) ^ (m)
#     end
#     return ans
# end

# W_asymp_2(F) = W.(0.05) + C1 * (F - F0) + C2 * (F - F0)^2 + C3 * (F - F0)^3

# Fs = 0.04: 0.001: 0.05
# plot(Fs, [W.(Fs) W_asymp.(Fs) W_asymp_2.(Fs, 3)])