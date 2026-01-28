import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW


field_plt_list = []
hhg_plt_list = []
hhg_data_x_list = []
shg_yields = []
thz_data = []
tau_list = []
hhg_k_linspace = []
mid_Efs_record = []
hhg_t_data = []
tau_thz_mid = 0.0
shg_id = 0


# Basic Parameters
Nr = 5000
Δr = 0.2 / 2
l_num = 60
Δt = 0.05 / 2
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = rmax * 0.8
a0 = 0.5
po_func(r) = -1.56295 * (r^2 + a0) ^ (-0.5) * flap_top_windows_f(r, 0, Ri_tsurf, 1/4, left_flag=false) * exp(-0.1 * (r^2 + a0) ^ (0.5))
absorb_func = absorb_boundary_r(rmax, Ri_tsurf)

for task_id = 1: 11

# # Create Physical World & Runtime
# pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
# rt = create_tdse_rt_sh(pw);

# # Initial Wave
# init_wave_list = itp_fdsh(pw, rt, err=1e-8);
# crt_shwave = deepcopy(init_wave_list[1]);
# en = get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5
# println("Energy of ground state: ", en)

# Define the Laser. 
E_fs = 0.0533
E_thz = 0.00002
E_dc = 0.00002
ω_fs = 0.05693
ω_thz = ω_fs / 25
nc = 15
tau_fs = 0
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz, samples_num=16)
tau_thz = tau_list[task_id]

Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.8, phase1=0.5pi, phase2=0.0)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="sin2", phase1=0.5pi)
E_applied(t) = (Ex_thz(t) + E_dc) * flap_top_windows_f(t, 0, tmax, 1/2)
At_datas, Et_datas, ts, steps = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light, appendix_steps=1)
plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts)


# # Propagation
# crt_shwave = deepcopy(init_wave_list[1])
# hhg_integral_t_1, hhg_integral_t_2, hhg_integral_t_3 = tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);
# # hhg_integral_t_1, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw, rt, At_data_x, steps, Ri_tsurf);

# # Store Data Manually
# example_name = "2025_1_17_test_$(task_id)"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "hhg_integral_t_1", hhg_integral_t_1)
#     write(file, "hhg_integral_t_2", hhg_integral_t_2)
#     write(file, "hhg_integral_t_3", hhg_integral_t_3)
# end

# Retrieve Data.
example_name = "2025_6_8_$(task_id)"
hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t_1")

# HHG
p, hhg_data_x, hhg_data_y, base_id, hhg_k_linspace = get_hhg_spectrum_xy(hhg_integral_t, Et_datas[1], Et_datas[2], tau_fs, tmax, ω_fs, ts, Δt, max_display_rate=10)

# recording
shg_id = base_id * 2
tau_thz_mid = get_exactly_coincided_delay(ω_fs, tau_fs, nc, ω_thz)
push!(hhg_plt_list, p)
push!(hhg_data_x_list, hhg_data_x)
push!(shg_yields, (hhg_data_x[shg_id]))
push!(field_plt_list, plot_fs_thz_figure(Ex_fs, Ey_fs, E_applied, ts))
push!(hhg_t_data, hhg_integral_t)

Ex_thz_tmp, Ey_thz_tmp, Ez_thz_tmp, tmax_tmp = light_pulse(ω_thz, E_thz, 1, 0, pulse_shape="sin2", phase1=0.5pi)
At_datas_tmp, Et_datas_tmp, ts_tmp = create_tdata(tmax_tmp, 0, Δt, Ex_thz_tmp, no_light, no_light)
thz_data = [Et_datas_tmp[1], ts_tmp]

At_THz_datas, Et_THz_datas, = create_tdata(tmax, 0, Δt, t -> E_applied(t), no_light, no_light, appendix_steps=1)
push!(mid_Efs_record, Et_THz_datas[1][Int64(floor((tau_fs + nc * pi / ω_fs) ÷ Δt) + 1)])

Tp = 2 * nc * pi / ω_fs
Ex_thz_tmp, Ey_thz_tmp, Ez_thz_tmp, tmax_tmp = light_pulse(ω_thz, E_thz, 1, 0, pulse_shape="sin2", phase1=0.5pi)
At_datas_tmp, Et_datas_tmp, ts_tmp = create_tdata(tmax_tmp + Tp/2, -Tp/2, Δt, Ex_thz_tmp, no_light, no_light)
thz_data = [Et_datas_tmp[1], ts_tmp]

end

dd = zeros(Float64, length(hhg_k_linspace), 11)
for i = 1: 11
    dd[:, i] .= norm.(hhg_data_x_list[i][1:length(hhg_k_linspace)])
end

default(fontfamily = "Helvetica",titlefontsize=10,
guidefontsize=10,
tickfontsize=10,
legendfontsize=10, dpi=1000)
C(g::ColorGradient) = RGB[g[z] for z=LinRange(0,1,11)]
cc = cgrad(:vik) |> C
lincolor_scheme = [cc[1] cc[2] cc[3] cc[4] cc[5] cc[6] cc[7] cc[8] cc[9] cc[10] cc[11]]
# label_scheme = ["𝛜 = 0" "𝛜 = 0.1" "𝛜 = 0.2" "𝛜 = 0.3" "𝛜 = 0.4" "𝛜 = 0.5" "𝛜 = 0.6" "𝛜 = 0.7" "𝛜 = 0.8" "𝛜 = 0.9"]
label_scheme = ["𝛜 = 0" "" "𝛜 = 0.2" "" "𝛜 = 0.4" "" "𝛜 = 0.6" "" "𝛜 = 0.8" "" "𝛜 = 1.0" ]

# p = plot(hhg_k_linspace, dd, yscale=:log10, ylimits=(1e-7, 1e2), xaxis=2:2:10,
#     linecolor=lincolor_scheme, lw=1.5, label="", xlabel="ω / ω0",
#     ylabel="Yield (arb.u.)", size=(300, 200), seriescolor=:roma, colorbar_title="1", line_z = 0:0.1:1.0)

x = hhg_k_linspace
y = [norm.(hhg_data_x_list[i][1:length(hhg_k_linspace)]) for i in 1:11]
z = 0: 0.1: 1.0
p = plot()
foreach(i->plot!(x, y[i], c=:jet, line_z=z[i], yscale=:log10, ylimits=(1e-7, 1e2), xaxis=2:2:10,
    lw=1.0, label="", xlabel="ω / ω0", ylabel="Yield (arb.u.)",
    size=(350, 250), colorbar_title="\nEllipticity", right_margin=Plots.mm * 5), 1:11)
p

d1 = shg_yields

plot(0:0.1:1.0, [norm.(d2) norm.(d1) norm.(d1 .- d2)])
plot(0:0.1:1.0, [(norm.(d1 .- d2) ./ (norm.(d1 .- d2) .+ norm.(d2)))],
    size=(280, 250), lw=1.5, xlabel="Ellipticity", ylabel="Contrast Ratio", label="")

# plot((norm.(d1 .- d2) ./ (norm.(d1 .- d2) .+ norm.(d2))), 0:0.1:1.0,
#     size=(180, 268), lw=1.5, xlabel="Contrast Ratio", label="",
#     xaxis=0:0.5:1.0, yaxis=0: 0.2: 1.0)

# plot(hhg_k_linspace, norm.(hhg_data_x_list[11][1:length(hhg_k_linspace)]), yscale=:log10, ylimits=(1e-7, 1e2))
# plot(tau_list, shg_yields)
# plot(tau_list, mid_Efs_record)

# unify(data) = (data .- minimum(data)) ./ (maximum(data) - minimum(data))
# s = 0.095
# shg_yields_scaled = s * (shg_yields .- shg_yields[1])
# p2 = plot(unify(tau_list), shg_yields_scaled)
# plot!(p2, unify(thz_data[2]), (reverse(thz_data[1])))
# plot!(p2, unify(tau_list), mid_Efs_record .- mid_Efs_record[1])

# hhg_plt_list[14]
# p2


# f = open("3_27_hhg_data_11.txt", "w+")
# for i = 1: length(hhg_t_data[11])
#     write(f, "$(real(hhg_t_data[11][i]))\n")
# end
# close(f)


# unify(data) = (data .- minimum(data)) ./ (maximum(data) - minimum(data))
# p2 = plot(unify(tau_list), unify(shg_yields))
# plot!(p2, unify(thz_data[2]), unify(-thz_data[1]))


# f = open("saoyanchi_data.txt", "w+")
# for i = 1: length(tau_list)
#     write(f, "$(tau_list[i]) $(shg_yields[4])\n")
# end
# close(f)
