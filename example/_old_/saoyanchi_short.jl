import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5

Nr = 40000
Δr = 0.2
l_num = 25
Δt = 0.05
Z = 1.0
po_func_r = coulomb_potiential_zero_fixed(Rco=100.0)
rmax = Nr * Δr
absorb_func = absorb_boundary_r(rmax)
Ri_tsurf = 6500.0

pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);
init_wave_list = itp_fdsh(pw, rt, err=1e-9);

# saoyanchi
# tau_range = -500.0: 500.0: 1000.0
tau_range = [-500.0, 1000.0]
p = Vector{Any}(undef, length(tau_range))
expected_kx_list = zeros(Float64, length(tau_range))
expected_ky_list = zeros(Float64, length(tau_range))
expected_kz_list = zeros(Float64, length(tau_range))

for (example_id, tau) in enumerate(tau_range)

# init wave
crt_shwave = deepcopy(init_wave_list[1]);

# two-color laser here
eps1 = 0.3
E_peak1 = 0.0533
# E_peak2 = 0.000177 * 2
E_peak2 = 0.0

ω1 = 0.05693
ω2 = ω1 / 5
E1x = E_peak1 * (1.0 / sqrt(eps1 ^ 2 + 1.0))
E2x = 0.0
E1y = E_peak1 * (eps1 / sqrt(eps1 ^ 2 + 1.0))
E2y = E_peak2
nc1 = 6.0
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

steps_1 = Int64((2 * nc1 * pi / ω1) ÷ Δt) + Int64(tau_fs ÷ Δt)
steps_2 = Int64((2 * nc2 * pi / ω2) ÷ Δt) + Int64(tau_thz ÷ Δt)
# fs: 1655.5  thz: 1103.65

steps = max(steps_1, steps_2)
t_linspace = create_linspace(steps, Δt)

Et_data_x = Ex.(t_linspace)
Et_data_y = Ey.(t_linspace)
Et_data_z = zeros(Float64, steps)

At_data_x = get_integral(Et_data_x, Δt)
At_data_y = get_integral(Et_data_y, Δt)
At_data_z = get_integral(Et_data_z, Δt)

# define k space
k_delta = 0.01
kmin = 0.01
kmax = 1.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))

# propagation
# phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# # tsurf + isurf
# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);

# # store data
# example_name = "saoyanchi_short_$example_id"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
#     write(file, "a_tsurff_vec", a_tsurff_vec)
# end

# retrieve data.
example_name = "saoyanchi_short_$example_id"
crt_shwave = retrieve_obj(example_name, "crt_shwave")
phi_record = retrieve_obj(example_name, "phi_record")
dphi_record = retrieve_obj(example_name, "dphi_record")
a_tsurff_vec = retrieve_mat(example_name, "a_tsurff_vec")

p[example_id] = tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true)

expected_kx, expected_ky, expected_kz = tsurf_get_average_momentum_vector_parallel(a_tsurff_vec, k_space, k_min=0.1)
expected_kx_list[example_id] = expected_kx
expected_ky_list[example_id] = expected_ky
expected_kz_list[example_id] = expected_kz

println("Now $example_id")

end