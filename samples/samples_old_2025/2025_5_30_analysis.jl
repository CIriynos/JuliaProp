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
using Interpolations
# pythonplot()
default(fontfamily = "Helvetica",titlefontsize=12,
guidefontsize=10,
tickfontsize=10,
legendfontsize=10,)

# example_name = "2025_5_27_ctmc_hhg"
# L1 = retrieve_obj(example_name, "L1")
# L2 = retrieve_obj(example_name, "L2")
example_name = "2025_5_25_ctmc_pc"
L1 = retrieve_obj(example_name, "py_ave_list_collection")
L2 = retrieve_obj("2025_6_1", "actual_At_list_collection")
P_total_list = retrieve_mat("2025_6_7", "P_total_data")

# CTMC Parameters
Δt = 0.2
filter_threshold = 2.0
p_min = -1.0
p_max = 1.0
p_delta = 0.005

# Define the Laser. 
E_fs = 0.0533 * 1
E_thz = 0.00002
E_dc = 0.0
# nc = 2
nc_thz = 2
ω_fs = 0.05693
ω_thz = ω_fs / 4
tau_fs = 0

tasks_num = length(L1[1])
max_tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, 10, ω_thz, samples_num=length(L1[1]), nc_thz=nc_thz)
max_ts = max_tau_list[1]: Δt: max_tau_list[tasks_num]
interp_res_1 = zeros(Float64, length(max_ts))
interp_res_2 = zeros(Float64, length(max_ts))
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, 2, ω_thz, samples_num=length(L1[1]), nc_thz=nc_thz)

spec_data = zeros(Float64, length(max_ts), length(L1))
ag_data = zeros(Float64, length(max_ts), length(L1))
check_data = zeros(Float64, length(max_ts), length(L1))
contrast_ratio = zeros(Float64, length(L1))
cutoff_freq_list_cc = []


p1s = []
p2s = []
p3s = []
k = 1
for nc = 1: 0.5: 10
    for cep = 0: 0.125pi: 2pi
        tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz, samples_num=length(L1[1]), nc_thz=nc_thz)

        a1 = (L1[k] .- L1[k][1]) .* P_total_list[k]
        a2 = (L2[k] .- L2[k][1])
        contrast_ratio[k] = (maximum(L1[k]) - minimum(L1[k])) / ( (maximum(L1[k]) + minimum(L1[k])))
        # a1 .*= (contrast_ratio[k]) .^ 1.0
        interp_cub_1 = cubic_spline_interpolation(range(tau_list[2], tau_list[tasks_num-1], tasks_num-2), a1[2: (tasks_num-1)])
        interp_cub_2 = cubic_spline_interpolation(range(tau_list[2], tau_list[tasks_num-1], tasks_num-2), a2[2: (tasks_num-1)])

        ts = tau_list[2]: (Δt): tau_list[tasks_num-1]
        b = (length(max_ts) - length(ts)) ÷ 2
        interp_res_1[1 + b: (b + length(ts))] .= (interp_cub_1(ts))
        interp_res_2[1 + b: (b + length(ts))] .= (interp_cub_2(ts))
                
        hhg_delta_k = 2pi / length(max_ts) / Δt
        hhg_k_linspace = [hhg_delta_k * (i - 1) for i = 1: length(max_ts)]
        spec1 = (fft(interp_res_1 .* hanning(length(max_ts))))
        spec2 = (fft(interp_res_2 .* hanning(length(max_ts))))
        base_id = Int64((ω_thz) ÷ hhg_delta_k) + 1
        upper_limit_of_hhg = ω_fs
        rg = 1: (Int64((upper_limit_of_hhg) ÷ hhg_delta_k) + 1)
        display_rg = 1: (Int64((13.0 * ω_thz) ÷ hhg_delta_k) + 1)
        norm_cp_data = 10 .* log10.(norm.(spec1) ./ norm.(spec2))
        angle_cp_data = unwrap(angle.(spec1) .- angle.(spec2))
        angle_cp_data .-= angle_cp_data[2]
        angle_cp_data[1] = 0.0
        
        p1 = plot(hhg_k_linspace[display_rg] ./ (ω_fs / 375), [norm.(spec1)[display_rg] norm.(spec2)[display_rg]], yscale=:log10, ylimits=(1e-5, 1e3))
        p2 = plot(hhg_k_linspace[display_rg] ./ (ω_fs / 375), [norm_cp_data[display_rg]], ylimits=(-10, 10))
        p3 = plot(hhg_k_linspace[display_rg] ./ (ω_fs / 375), [angle_cp_data[display_rg]], ylimits=(-5pi, 5pi))
        push!(p1s, p1)
        push!(p2s, p2)
        push!(p3s, p3)
        check_data[:, k] .= interp_res_2
        spec_data[:, k] .= norm_cp_data
        ag_data[:, k] .= angle_cp_data

        interp_cub_f = cubic_spline_interpolation(0: hhg_delta_k: upper_limit_of_hhg, norm_cp_data[rg])
        xs = 0: (hhg_delta_k / 10): (upper_limit_of_hhg - hhg_delta_k)
        ys = interp_cub_f(xs)
        cutoff_id_list = []
        critical_cut_off_id = last(rg)
        for i = 2: length(xs)
            if (ys[i] <= -3.0 && ys[i - 1] >= -3.0) || (ys[i] >= -3.0 && ys[i - 1] <= -3.0)
                push!(cutoff_id_list, i)
            end
            if ys[i] <= -10.0
                critical_cut_off_id = min(i ÷ 10, critical_cut_off_id)
            end
        end
        cutoff_freq_list = @. (cutoff_id_list - 1) * (hhg_delta_k / 10) / (ω_fs / 375)
        push!(cutoff_freq_list_cc, cutoff_freq_list)

        k += 1
    end
end


hhg_delta_k = 2pi / length(max_ts) / Δt
id = (Int64((ω_fs * 1) ÷ hhg_delta_k) + 1)
start_id = 2    # SHG: 2   PC: 1
ncs = 1: 0.01: 10
Tps = @. 2 * ncs * pi / ω_fs
delta_t_s = @. (2/pi) * Tps * asin(1 / sqrt(2)) 
delta_ω_s = (5.54 ./ delta_t_s) ./ (ω_fs / 375)
plot(ncs, delta_ω_s)


# 1
phi_cep_id = 9
rg1 = (0 + phi_cep_id): 17*2: 323
d1 = spec_data[start_id: id, rg1]
for i = 1: length(rg1)
    # d1[:, i] .+= log10(contrast_ratio[rg1[i]]) .* 10
end
mm = maximum(d1)

xs = 1: 1: 10
ys = start_id: id
cubic_itp_1 = cubic_spline_interpolation((ys, xs), d1)
xi = 1: 0.05: 10
yi = start_id: 0.1: id
z1 = cubic_itp_1(yi, xi)
least_display_limit = -30
fig1 = heatmap(xi, yi .* hhg_delta_k ./ (ω_fs / 375), max.(z1 .- mm),
    clim=(least_display_limit, 0), xlabel="Numbers of Cycles", ylabel="frequency (THz)",
    linewidth=0.4, size=(350, 250), cmap=:jet1)

tmp = yi .* hhg_delta_k ./ (ω_fs / 375)
plot!(fig1, ncs, min.(delta_ω_s, maximum(tmp)), linestyle=:dash, color=:green,
    xlims = (1, 10), ylims = (minimum(tmp), maximum(tmp)), leg=false)

foo = max.(z1 .- mm, least_display_limit) .>= -3.0
# bar = ones(Int64, size(foo)[2]) .* size(foo)[1]
bar = zeros(Int64, size(foo)[2])
for i = 1: size(foo)[2]
    for j = 1: (size(foo)[1] - 1)
        if foo[j, i] == true && foo[j + 1, i] == false
            bar[i] = j
        end
    end
end
plot!(fig1, xi, bar .* hhg_delta_k ./ (ω_fs / 375) ./ 7, linestyle=:dash, color=:blue,
    xlims = (1, 10), ylims = (minimum(tmp), maximum(tmp)), leg=false, right_margin = 8 * Plots.mm)
fig1


# # 2
# given_nc = 4
# d2 = spec_data[start_id: id, (1+17*(2 * (given_nc - 1))): (1+17*(2 * (given_nc - 1))+16)]
# xs = 1: 17
# ys = start_id: id
# cubic_itp_2 = cubic_spline_interpolation((ys, xs), d2)
# xi = 1: 0.05: 17
# yi = start_id: 0.1: id
# z2 = cubic_itp_2(yi, xi)
# println("max z2 = $(maximum(z2))")
# contourf(range(0, 360, length(xi)), yi .* hhg_delta_k ./ (ω_fs / 375), max.(z2, -15),
#     levels=30, clim=(-15, 0), xlabel="Φ_CEP (degree)", ylabel="frequency (THz)")

# # 3
# given_nc = 6
# d3 = ag_data[start_id:id, (1+17*(2 * (given_nc - 1))): (1+17*(2 * (given_nc - 1))+16)]
# xs = 1: 17
# ys = start_id: id
# cubic_itp_3 = cubic_spline_interpolation((ys, xs), d3)
# xi = 1: 0.05: 17
# yi = start_id: 0.1: id
# z3 = cubic_itp_3(yi, xi)
# contourf(range(0, 360, length(xi)), yi .* hhg_delta_k ./ (ω_fs / 375), z3, levels=20, color=:hsv,
#     xlabel="Φ_CEP (degree)", ylabel="frequency (THz)", clim=(-pi, pi), size=(400, 300), right_margin = 8 * Plots.mm)
