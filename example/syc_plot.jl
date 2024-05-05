import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5

Nr = 10000
Δr = 0.2
l_num = 30
Δt = 0.05
Z = 1.0
po_func_r = coulomb_potiential_zero_fixed(Rco=100.0)
rmax = Nr * Δr
absorb_func = absorb_boundary_r(rmax)

pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);
init_wave_list = itp_fdsh(pw, rt, err=1e-9);

# saoyanchi
tau_range = [-800.0, -500.0, -400.0, -350.0, -300.0, -250.0, -200.0, -150.0, -100.0, -50.0, 0.0,
        50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 700.0, 800.0, 900.0,
        1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0][1:31]

p = Vector{Any}(undef, length(tau_range))
expected_kx_list = zeros(Float64, length(tau_range))
expected_ky_list = zeros(Float64, length(tau_range))
expected_kz_list = zeros(Float64, length(tau_range))
phi_record_norm_list = zeros(Float64, length(tau_range))

for (example_id, tau) in enumerate(tau_range)

# init wave
crt_shwave = deepcopy(init_wave_list[1]);

# two-color laser here
eps1 = 0.3
E_peak1 = 0.0533
E_peak2 = 0.000177

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

# Ax(t) = (E1x / ω1) * sin(ω1 * (t - tau_fs) / 2 / nc1)^2 * sin(ω1 * (t - tau_fs) + phase_1x) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1)) + 
#         (E2x / ω2) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2x) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))
# Ay(t) = (E1y / ω1) * sin(ω1 * (t - tau_fs) / 2 / nc1)^2 * sin(ω1 * (t - tau_fs) + phase_1y) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1)) + 
#         (E2y / ω2) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2y) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

Ex(t) = (E1x) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1x) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1)) + 
        (E2x) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2x) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

Ey(t) = (E1y) * sin(ω1 * (t - tau_fs) / 2 / nc1) ^2 * sin(ω1 * (t - tau_fs) + phase_1y) * (t - tau_fs > 0 && t - tau_fs < (2 * nc1 * pi / ω1)) + 
        (E2y) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2y) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))

steps_1 = Int64((2 * nc1 * pi / ω1) ÷ Δt) + Int64(tau_fs ÷ Δt)
steps_2 = Int64((2 * nc2 * pi / ω2) ÷ Δt) + Int64(tau_thz ÷ Δt)
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

k_linspace_2 = 0.05: 0.01: 1.0
k_space_2 = create_k_space(k_linspace_2, fixed_theta(pi/2), phi_linspace(180))


# propagation
Ri_tsurf = 1500.0

# phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# no need here, maybe.
# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);

# δa_lm = isurf_rest_part(crt_shwave, k_space.k_r_range, last(t_linspace), Ri_tsurf, pw, rt)


########################
# retrieve data.
example_name = "saoyanchi_$example_id"
crt_shwave = retrieve_obj(example_name, "crt_shwave")
phi_record = retrieve_obj(example_name, "phi_record")
dphi_record = retrieve_obj(example_name, "dphi_record")
δa_lm = retrieve_obj(example_name, "delta_a_lm")

# p[example_id] = tsurf_plot_xy_momentum_spectrum(δa_lm, k_space, pw, kr_flag=true, k_min=0.05)
expected_kx, expected_ky, expected_kz = tsurf_get_average_momentum_parallel(δa_lm, k_space, pw, k_min=0.0)
expected_kx_list[example_id] = expected_kx
expected_ky_list[example_id] = expected_ky
expected_kz_list[example_id] = expected_kz


# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space_2, TSURF_MODE_ELLI);
# p[example_id] = tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space_2, kr_flag=true)
# expected_kx, expected_ky, expected_kz = tsurf_get_average_momentum_vector_parallel(a_tsurff_vec, k_space_2)
# expected_kx_list[example_id] = expected_kx
# expected_ky_list[example_id] = expected_ky
# expected_kz_list[example_id] = expected_kz

# phi_record_norm_list[example_id] = sum(norm.(phi_record))

println("Now example_id=$example_id finished.")

end

# p[1]
plot(tau_range, [abs.(expected_ky_list)], xlabel="τ (a.u.)", ylabel="Expected value of ky")

# plot(phi_record_norm_list)
# result: since example_id=21, tsurf must be used.

# A_thz(t) = (E2y / ω2) * sin(ω2 * (t - tau_thz) / 2 / nc2)^2 * sin(ω2 * (t - tau_thz) + phase_2y) * (t - tau_thz > 0 && t - tau_thz < (2 * nc2 * pi / ω2))
# At_data_thz = A_thz.(t_linspace)
# plot(t_linspace, At_data_thz, xlabel="time (a.u.)", label="THz pulse A(t)")

# Et_data_y = get_derivative_two_order(At_data_y, Δt)
# At_data_y_2 = get_integral(Et_data_y, Δt)
# plot([At_data_y_2 At_data_y])
# plot(Et_data_y)
