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


# Research on Ionz. Rate 

# tau_list = [-400, 100, 300, 500, 1000]
# -500, 100, 300, 500, 1000
p = plot()
p2 = plot()
rcd_shg = []
wt_datas = []
et_figs = []


for tau_id in 1: 5

ω = 0.05693
ω_thz = ω / 5
E0 = 0.0533
gamma = ω / E0
nc = 15
Δt = 0.01
E0_thz = 0.00001 * 10
E_c = 0.00001 * 10

tau_lst = [-2pi/ω_thz, nc*pi/ω - 1.5pi/ω_thz,
    nc*pi/ω - pi/ω_thz, nc*pi/ω - 0.5pi/ω_thz, nc*2pi/ω]
tau = tau_lst[tau_id]
println("tau = $tau")

E_thz(t) = (E0_thz) * sin(ω_thz*(t-tau)) * (t-tau > 0 && t-tau <(2*pi/ω_thz))
E_fs(t) = E0 * sin(ω*t/2/nc)^2 * cos(ω*t + pi) * (t < 2*nc*pi/ω)
E(t) = E_fs(t) + (E_thz(t) + E_c) * flap_top_windows_f(t, 0, 2*nc*pi/ω, 1/8)
t_linspace = 0: Δt: 2*nc*pi/ω
e_fig = plot(t_linspace, [E_thz.(t_linspace) * 200 E_fs.(t_linspace)])
push!(et_figs, e_fig)

Wk(gamma, Ip, omega) = exp(-(2*Ip/omega) * ((1 + 0.5/gamma^2) * asinh(gamma) - sqrt(1 + gamma^2) / 2 / gamma))
Wk2(F, Ip, omega) = Wk(omega * sqrt(2*Ip) / F, Ip, omega)
ADK_f(F) = 4 / F * exp(-2 / (3 * F))
# W(t) = ADK_f(abs(E(t)) + 1e-10)
W(t) = Wk2(abs(E(t)) + 1e-10, 0.5, ω)
Wt_data = W.(t_linspace)
Wt_int_data = get_integral(Wt_data, Δt)
Wt_data_saturn = @. Wt_data * exp(-Wt_int_data)
push!(wt_datas, Wt_data_saturn)

# for FFT
N = length(t_linspace)
Δω = 2pi / N / Δt
ω_linspace = [(i==1) ? (Δω) : (Δω * (i - 1)) for i = 1: N]
hhg_window_f(t, tmax) = sin(t / tmax * pi) ^ 2
shg_place = Int64(floor(ω * 2 / Δω))
println("shg_place = $shg_place")


# # Method 1 (My Analyse)
# H_data = fft(W.(t_linspace) .* hhg_window_f.(t_linspace, last(t_linspace)))
# G_data = fft(E.(t_linspace) .* hhg_window_f.(t_linspace, last(t_linspace)))
# plot(ω_linspace[1:100], norm.(H_data)[1:100])
# plot(ω_linspace[1:100], norm.(G_data)[1:100])

# H_super = H_data ./ (im * ω_linspace)
# res = conv(H_super, G_data)[1:N] .+ sqrt(2pi) .* Ec .* H_super
# plot!(p, ω_linspace[1:200], norm.(res)[1:200], yscale=:log10)


# method 2 (Directly)
Eyield = E.(t_linspace) .* get_integral(Wt_data_saturn, Δt)
Gyield = fft(Eyield .* hhg_window_f.(t_linspace, last(t_linspace)))
plot!(p2, ω_linspace[1:200], norm.(Gyield)[1:200], yscale=:log10)
#plot(ω_linspace[1:100], [norm.(res)[1:100] * 1e-5 norm.(Gyield)[1:100]], yscale=:log10)

push!(rcd_shg, norm.(Gyield)[shg_place])

end

p2
plot(rcd_shg)

# x0 = 13
# Wx(x) = 6 * x * exp(-x)
# Wx_1(x) = (1-x) * 6 * exp(-x)
# xs = 5:0.01:14
# plot(xs, [Wx.(xs) Wx_1.(xs)])