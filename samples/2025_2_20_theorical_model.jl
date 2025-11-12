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


# Research on Ionz. Rate 

theory_plt_list = []
theory_plt_list_1 = []
theory_record_shg = []
theory_record_shg_1 = []
theory_hhg_data_list = []
conv_record = []
wt_datas = []
et_figs = []
theory_mid_thz = []
tau_list = []
hhg_k_linspace = []
shg_id = 0

# scheme 1 [S.V. Popruzhenko]
g(γ) = (3 / 2γ) * ((1 + 1 / (2 * γ^2)) * asinh(γ) - sqrt(1 + γ^2) / (2γ))
dawson_int(x) = 0.5 * sqrt(pi) * exp(-x^2) * erfi(x)
β(γ) = (2 * γ) / sqrt(1 + γ^2)
c1(γ) = asinh(γ) - γ / sqrt(1 + γ ^ 2)
nth(γ) = K0 * (1 + 1 / (2 * γ^2))
γ_real(F) = ω * sqrt(Ip * 2) / F

function w_sr(t, E_laser, E_envelop)
    F = abs(E_laser(t)) + 1e-50
    γ = γ_real(abs(E_envelop(t)) + 1e-50)
    res = 0.0
    for n = 1: Int64(floor(nth(γ)) + 1) + 1
        if n > nth(γ)
            res += dawson_int(sqrt(β(γ) * (n - nth(γ)))) * exp(-2 * g(γ) / (3 * F) - 2 * c1(γ) * (n - nth(γ)))
        end
    end
    Q = (2 / F) ^ 2 * (1 + 2 * exp(1)^-1 * γ) ^ (-2)
    return res * (2 / pi) * Ip * K0 ^ (-3/2) * β(γ) ^ (1/2) * Q
end

# scheme 2 (ADK)
# ADK_f(F) = 4 / F * exp(-2 * g(ω * sqrt(Ip * 2) / F) / (3 * F))
ADK_f(F) = 4 / F * exp(-2 / (3 * F))

# scheme 3 (Keldysh)
Keldysh_ionz_rate(γ, Ip, ω) = exp(-(2 * Ip / ω) * ((1 + 1 / (2 * γ^2)) * asinh(γ) - sqrt(1 + γ^2) / (2γ)))


ω = 0.05693
ω_thz = ω / 20
E0 = 0.0533
Ip = 0.5
K0 = 0.5 / ω
nc = 15
Δt = 1.0
E0_thz = 0.0001
E_dc = 0.0001


samples_num = 64
# tau_id = 1
for tau_id in 1: samples_num

tau_fs = 2000
tau_list = get_1c_thz_delay_list_ok(ω, tau_fs, nc, ω_thz, samples_num=samples_num)
tau_thz = tau_list[tau_id]
println("tau_thz = $tau_thz")

# Et_fs_envelop(t) = E0 * sin(ω*(t-tau_fs)/2/nc)^2 * (t-tau_fs > 0 && t-tau_fs < 2*nc*pi/ω)
Tp = 2*nc*pi/ω
Et_fs_envelop(t) = E0 * exp(-(t - tau_fs - Tp/2)^2/(Tp/2)^2)^2
Et_thz_envelop(t) = E0_thz * exp(-(t - tau_thz - pi/ω_thz)^2/(pi/ω_thz)^2)^2
Et_fs(t) = Et_fs_envelop(t) * cos(ω*(t-tau_fs) + 0.5pi)
# Et_thz(t) = Et_thz_envelop(t) * sin(ω_thz*(t-tau_thz))
Et_thz(t) = (E0_thz) * sin(ω_thz*(t-tau_thz)/2)^2 * sin(ω_thz*(t-tau_thz)) * (t-tau_thz > 0 && t-tau_thz <(2*pi/ω_thz))
E_applied(t) = (Et_thz(t) + E_dc)
E(t) = Et_fs(t) + E_applied(t)
tmax = tau_fs + Tp + 2000
t_linspace = 0: Δt: tmax
e_fig = plot(t_linspace, [E_applied.(t_linspace) * 500 E.(t_linspace)])
push!(et_figs, e_fig)


# Get Weight
W(t) = ADK_f(((abs(E(t)) + 1e-50) / E0) ^ 1.0 * E0)
W_no_thz(t) = ADK_f(((abs(Et_fs(t)) + 1e-50) / E0) ^ 1.0 * E0)
# W(t) = Keldysh_ionz_rate(γ_real(abs(E(t)) + 1e-50), Ip, ω) * 2.5e1
Wt_data = W.(t_linspace)
push!(wt_datas, Wt_data)


# for FFT
N = length(t_linspace)
hhg_delta_k = 2pi / N / Δt
hhg_k_linspace = [hhg_delta_k * (i - 1) for i = 1: N]
spectrum_range = 1: Int64(floor(ω * 10 / hhg_delta_k) + 1)
hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
hhg_windows_data = hhg_windows_f.(t_linspace, 0, tmax)
shg_id = Int64(floor((ω * 2) ÷ hhg_delta_k) + 1)
base_id = Int64(floor(ω ÷ hhg_delta_k) + 1)


# Method 1 (My Analyse)
H_data = fft(Wt_data .* 1)
G_data = fft(E.(t_linspace) .* 1)
H_super = H_data ./ (im * (hhg_k_linspace .+ hhg_delta_k))
res = (conv(H_super, G_data)[1: N]) ./ (length(hhg_k_linspace))
p1 = plot(hhg_k_linspace[spectrum_range] ./ ω, 
    [norm.(res)[spectrum_range]], yscale=:log10, ylimit=(1e-7, 1e3))
# p1 = H_data


# method 2 (Directly)
Eyield = E.(t_linspace) .* get_integral(Wt_data, Δt)
Gyield = fft(Eyield .* hhg_windows_data)
p2 = plot(hhg_k_linspace[spectrum_range] ./ ω, norm.(Gyield)[spectrum_range], yscale=:log10, ylimit=(1e-7, 1e3))


# record
push!(theory_plt_list, p2)
push!(theory_plt_list_1, p1)
push!(theory_record_shg, norm(Gyield[shg_id]))
# push!(theory_record_shg_1, norm(res[shg_id]))
# push!(conv_record, conv(H_super, G_data))
push!(theory_hhg_data_list, Gyield)
push!(theory_mid_thz, E_applied(tau_fs + nc * pi / ω))

end

theory_plt_list[1]
# theory_plt_list_1[1]

# plot(tau_list, theory_record_shg)

unify(data) = (data .- minimum(data)) ./ (maximum(data) - minimum(data))
p2 = plot(unify(tau_list), unify(theory_record_shg))
plot!(p2, unify(tau_list), unify(theory_mid_thz))



# p3 = plot!(p3, unify(theory_record_shg))

# p3 = p
# p3 = plot!(p3, theory_plt_list[1])

# theory_plt_list[1]
# plot([norm.(hhg_data_x_list[1])[1: 150] norm.(theory_hhg_data_list[1])[1: 150] .* 3], yscale=:log10,
#     labels = ["TDSE" "CTMC*"]) 