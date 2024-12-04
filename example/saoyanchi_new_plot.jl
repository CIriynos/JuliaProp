import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5

Nr = 25000
Δr = 0.2
l_num = 40
Δt = 0.05
Z = 1.0
po_func_r = coulomb_potiential_zero_fixed(Rco=250.0)
rmax = Nr * Δr
Ri_tsurf = 4500.0
absorb_func = absorb_boundary_r(rmax, Ri_tsurf)

pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);
init_wave_list = itp_fdsh(pw, rt, err=1e-9);
get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5


# saoyanchi
# tau_range = [0, 1500.0]
# tau_range = [-700: 200: 1500; -600: 200: 1400]
# tau_range = -700: 200: 1500
# tau_range = [-700: 100: 700; 900: 200: 1500]
# tau_range = [-700: 100: 1500; 1700: 200: 2100]
tau_range = [300, 1700]

p = Vector{Any}(undef, length(tau_range))
expected_kx_list = zeros(Float64, length(tau_range))
expected_ky_list = zeros(Float64, length(tau_range))
expected_kz_list = zeros(Float64, length(tau_range))

for (example_id, tau_int) in enumerate(tau_range)

tau = Float64(tau_int)

# init wave
crt_shwave = deepcopy(init_wave_list[1]);

# two-color laser here
eps1 = 0.3
E_peak1 = 0.0533
E_peak2 = 0.000177
# E_peak2 = 0.0

ω1 = 0.05693
ω2 = ω1 / 10
E1x = E_peak1 * (1.0 / sqrt(eps1 ^ 2 + 1.0))
E2x = 0.0
E1y = E_peak1 * (eps1 / sqrt(eps1 ^ 2 + 1.0))
E2y = E_peak2
nc1 = 15.0
nc2 = 1.0
phase_1x = 0.5pi
phase_2x = 0.0
phase_1y = 0.0
phase_2y = 0.0    # ???
tau_fs = (tau < 0.0) * abs(tau)
tau_thz = (tau >= 0.0) * abs(tau)

Ex(t) = (E1x) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1x) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1)) + 
        (E2x) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2x) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

Ey(t) = (E1y) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1y) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1)) + 
        (E2y) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2y) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

steps_1 = Int64((2 * nc1 * pi / ω1) ÷ Δt + 1) + Int64(tau_fs ÷ Δt + 1)
steps_2 = Int64((2 * nc2 * pi / ω2) ÷ Δt + 1) + Int64(tau_thz ÷ Δt + 1)
# fs: 1655.5  thz: 1103.65
# -1500 -> 2000
steps = max(steps_1, steps_2)
t_linspace = create_linspace(steps, Δt)

Et_data_x = Ex.(t_linspace)
Et_data_y = Ey.(t_linspace)
Et_data_z = zeros(Float64, steps)

At_data_x = get_integral(Et_data_x, Δt)
At_data_y = get_integral(Et_data_y, Δt)
At_data_z = get_integral(Et_data_z, Δt)

# define k space
k_delta = 0.002
kmin = 0.002
kmax = 1.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))
k_space_2 = create_k_space(k_linspace, theta_linspace(180), phi_linspace(360))

# propagation
# phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# # no need here, maybe.
# # a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);

# δa_lm = isurf_rest_part(crt_shwave, k_space.k_r_range, last(t_linspace), Ri_tsurf, pw, rt)

# example_name = "saoyanchi_new_$tau_int"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
#     write(file, "delta_a_lm", hcat(δa_lm...))
# end

# retrieve data.
example_name = "saoyanchi_new_$tau_int"
crt_shwave = retrieve_obj(example_name, "crt_shwave")
phi_record = retrieve_obj(example_name, "phi_record")
dphi_record = retrieve_obj(example_name, "dphi_record")
δa_lm = retrieve_obj(example_name, "delta_a_lm")

p[example_id] = tsurf_plot_xy_momentum_spectrum(δa_lm, k_space, pw, kr_flag=false, k_min=0.05)
# expected_kx, expected_ky, expected_kz = tsurf_get_average_momentum_parallel(δa_lm, k_space, pw, k_min=0.0)
# expected_kx_list[example_id] = expected_kx
# expected_ky_list[example_id] = expected_ky
# expected_kz_list[example_id] = expected_kz

# expected_kx, expected_ky, expected_kz = tsurf_get_average_momentum_parallel(δa_lm, k_space_2, pw, k_min=0.0)
# expected_kx_list[example_id] = expected_kx
# expected_ky_list[example_id] = expected_ky
# expected_kz_list[example_id] = expected_kz

println("Now $tau_int")

end

using LaTeXStrings;

plot(-0.75: 0.01: 0.75, [normalize(p[2][1] ./ p[2][2]) / 10 normalize(p[1][1] ./ p[1][2]) / 10],
        guidefont=Plots.font(14, "Times"),
        tickfont=Plots.font(14, "Times"),
        legendfont=Plots.font(10),
        margin = 5 * Plots.mm,
        xlabel="Projection in the x",
        ylabel = "Weight",
        label = [L"THz(τ_2)" L"THz(τ_1)"],
        linewidth = 1.5
)

tmp = [normalize(p[2][1] ./ p[2][2]) / 10 normalize(p[1][1] ./ p[1][2]) / 10]
f = open("dtt", "w")
write(f, tmp)
close(f)


# x_range = -0.75: 0.01: 0.75
# data1 = normalize(p[2][1] ./ p[2][2]) / 10
# data2 = normalize(p[1][1] ./ p[1][2]) / 10
# f = open("xproj_data.txt", "w+")
# for i = 1: length(p[1][1])
#         write(f, "$(x_range[i]) $(data1[i]) $(data2[i])\n")
# end
# close(f)


# mm = -350 + 1900
# plot(tau_range[1:23] .+ mm, abs.(expected_ky_list)[1:23],
#         xlabel="Time delay τ (a.u.)",
#         ylabel="Average momentum ky (a.u.)",
#         guidefont=Plots.font(14, "Times"),
#         tickfont=Plots.font(14, "Times"),
#         legendfont=Plots.font(14, "Times"),
#         margin = 5 * Plots.mm,
#         legend = false,
#         linewidth = 2.5
#         )


# p1 = scatter(tau_range[1:23] .+ mm, (abs.(expected_ky_list)[1:23] .- 0.004) * 3.4,
#         label="TDSE",
#         markershape = :diamond,
#         markersize = 5)
# ω1 = 0.05693
# ω2 = ω1 / 10
# t_linspace = create_linspace(Int64((2 * 1 * pi / ω2) ÷ Δt + 1), 0.05)
# E_THz(t) = (0.00017) * sin(ω2 * (t) / 2 / 1)^2 * sin(ω2 * (t)) * (t > 0 && t < (2 * 1 * pi / ω2))
# Et_data_thz = E_THz.(t_linspace)
# At_data_thz = get_integral(Et_data_thz, Δt)
# plot!(p1, range(tau_range[1] + 230 + mm, tau_range[23] - 330 + mm, length(t_linspace)), At_data_thz,
#         guidefont=Plots.font(14, "Times"),
#         tickfont=Plots.font(14, "Times"),
#         legendfont=Plots.font(14, "Times"),
#         margin = 5 * Plots.mm,
#         xlabel="Time delay τ(a.u.)",
#         ylabel = "Vector potiential A (a.u.)",
#         label = "THz",
#         linewidth = 2.5
#         )


# example_name = "saoyanchi_new_plot_data"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "expected_kx_list", expected_kx_list)
#     write(file, "expected_ky_list", expected_ky_list)
#     write(file, "expected_kz_list", expected_kz_list)
# end


# using Plots
# tau_range = [-700: 100: 1500; 1700: 200: 2100]
# expected_ky_list = retrieve_mat("saoyanchi_new_plot_data", "expected_ky_list")
# len = length(tau_range)
# plot(tau_range, [abs.(expected_ky_list[1:len])])
# title!("ky average with delay τ")
# xlabel!("τ")
# ylabel!("P(ky)")