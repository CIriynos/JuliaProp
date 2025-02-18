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


p = plot()
p2 = plot()
rcd_shg = []
wt_datas = []
et_figs = []

# tau_id = 1
for tau_id = 1: 5

ω = 0.05693
ω_thz = ω / 20
E0 = 0.0533 * 1
gamma = ω / E0
nc = 15
Δt = 0.1
E0_thz = 0.00002 * 1
E_c = 0.00002 * 1

tau_fs = 500
tau_lst = tau_fs .+ [-2pi/ω_thz, nc*pi/ω - 1.5pi/ω_thz,
    nc*pi/ω - pi/ω_thz, nc*pi/ω - 0.5pi/ω_thz, nc*2pi/ω]
tau_thz = tau_lst[tau_id]
println("tau_thz = $tau_thz")

t_min = 0
t_max = 2*nc*pi/ω + 2*tau_fs
t_linspace = t_min: Δt: t_max

E_dc(t) = E_c
E_thz(t) = (E0_thz) * sin(ω_thz*(t-tau_thz)) * (t-tau_thz > 0 && t-tau_thz <(2*pi/ω_thz))
E_fs(t) = E0 * sin(ω*(t-tau_fs)/2/nc)^2 * cos(ω*(t-tau_fs) + pi) * ((t-tau_fs) > 0 && (t-tau_fs) < 2*nc*pi/ω)
E_add(t) = (E_thz(t) + E_dc(t)) * flap_top_windows_f(t, t_min, t_max, 1/6)
E(t) = E_fs(t) + E_add(t)

e_fig = plot(t_linspace, [E_add.(t_linspace) * 200 E_fs.(t_linspace)])
push!(et_figs, e_fig)


Wk(gamma, Ip, omega) = exp(-(2*Ip/omega) * ((1 + 0.5/gamma^2) * asinh(gamma) - sqrt(1 + gamma^2) / 2 / gamma))
Wk2(F, Ip, omega) = Wk(omega * sqrt(2*Ip) / F, Ip, omega)
ADK_f(F) = 4 / F * exp(-2 / (3 * F))
W(t) = ADK_f(abs(E(t)) + 1e-10)
W_no_thz(t) = ADK_f(abs(E_fs(t)) + 1e-10)
Wt_data = W.(t_linspace)
Wt_data_no_thz = W_no_thz.(t_linspace)
Wt_int_data = get_integral(Wt_data, Δt)
Wt_data_saturn = @. Wt_data * exp(-Wt_int_data)
# Wt_special = @. Wt_data * (1.0 .+ Wt_int_data .* 50)
# plot([Wt_data Wt_special])

push!(wt_datas, Wt_data_saturn)


# for FFT
N = length(t_linspace)
Δω = 2pi / N / Δt
ω_linspace = [(i==1) ? (Δω) : (Δω * (i - 1)) for i = 1: N]
hhg_window_f(t, tmax) = sin(t / tmax * pi) ^ 2
shg_place = Int64(floor(ω * 2 / Δω))
println("shg_place = $shg_place")
spectrum_range = 1: Int64(floor(ω * 10 / Δω))


# Method 1 (My Analyse)
H_data = fft(Wt_data .* hhg_window_f.(t_linspace, last(t_linspace)))
G_data = fft(E.(t_linspace) .* hhg_window_f.(t_linspace, last(t_linspace)))
plot(ω_linspace[1:200], norm.(H_data)[1:200],  yscale=:log10)
plot(ω_linspace[1:200], norm.(G_data)[1:200],  yscale=:log10)

H_super = H_data ./ (im * ω_linspace)
res = conv(H_super, G_data)[1:N] .+ sqrt(2pi) .* E_c .* H_super
res1 = conv(H_super, G_data)[1:N]
res2 = sqrt(2pi) .* E_c .* H_super
plot(norm.(H_super)[spectrum_range] .^ 2, yscale=:log10)
plot!(p, ω_linspace[spectrum_range], [norm.(res)[spectrum_range] .^ 2], yscale=:log10)


# method 2 (Directly)
Eyield = E.(t_linspace) .* get_integral(Wt_data, Δt)
plot(Eyield)
Gyield = fft(Eyield .* hhg_window_f.(t_linspace, last(t_linspace)))
plot!(p2, ω_linspace[spectrum_range], norm.(Gyield)[spectrum_range] .^ 2,
    yscale=:log10)

end

p
p2

# plot(ω_linspace[spectrum_range], 
#     [norm.(hhg_spectrum_x)[spectrum_range] norm.(res)[spectrum_range] .* 5e-5 norm.(Gyield)[spectrum_range] .* 0.5],
#     yscale=:log10, ylimit=(1e-7, 1e3))