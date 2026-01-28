import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5

Nr = 3000
Δr = 0.2
l_num = 15
Δt = 0.05
Z = 1.0
po_func_r = coulomb_potiential_zero_fixed_plus(Rco=25.0)
# po_func_r = coulomb_potiential_zero_fixed(Rco=25.0)
rmax = Nr * Δr
absorb_func = absorb_boundary_r(rmax, 450.0, pow_value=8.0, max_value=10.0)

pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# init wave
init_wave_list = itp_fdsh(pw, rt, err=1e-9);
crt_shwave = deepcopy(init_wave_list[1]);

# Define laser here.
E0 = 0.0533799
omega = 0.114
nc = 2.0
steps = Int64((2 * nc * pi / omega) ÷ Δt)
t_linspace = create_linspace(steps, Δt)

Ax(t) = (E0 / omega) * (sin(omega * t / 2.0 / nc) ^ 2) * sin(omega * t) * (t < (2 * nc * pi / omega))
Ay(t) = (E0 / omega) * (sin(omega * t / 2.0 / nc) ^ 2) * cos(omega * t) * (t < (2 * nc * pi / omega))
At_data_x = Ax.(t_linspace)
At_data_y = Ay.(t_linspace)
At_data_z = zeros(Float64, steps)

# define k space
k_delta = 0.01
kmin = 0.01
kmax = 1.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 200
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))

# propagation
Ri_tsurf = 450.0
@time phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

Ri_tsurf = 450.0
a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);
tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true)


# δa_lm = isurf_rest_part(crt_shwave, k_space.k_r_range, last(t_linspace), Ri_tsurf, pw, rt)
# tsurf_plot_xy_momentum_spectrum(δa_lm, k_space, pw, kr_flag=true, k_min=0.05)
# expected_kx, expected_ky, expected_kz = tsurf_get_average_momentum_parallel(δa_lm, k_space, pw, k_min=0.0)

# a_tsurff_lm_vec = tsurf_sh_vector(pw, phi_record, dphi_record, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space.k_collection, TSURF_MODE_ELLI)
# tsurf_plot_xy_momentum_spectrum(a_tsurff_lm_vec, k_space, pw, kr_flag=true)

# store data
example_name = "attoclock"
h5open("./data/$example_name.h5", "w") do file
    write(file, "crt_shwave", hcat(crt_shwave...))
    write(file, "phi_record", hcat(phi_record...))
    write(file, "dphi_record", hcat(dphi_record...))
    write(file, "a_tsurff_vec", a_tsurff_vec)
end


########################
# retrieve data.
crt_shwave = retrieve_obj("attoclock", "crt_shwave")
phi_record = retrieve_obj("attoclock", "phi_record")
dphi_record = retrieve_obj("attoclock", "dphi_record")
a_tsurff_vec = retrieve_mat("attoclock", "a_tsurff_vec")

# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);
tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true)
