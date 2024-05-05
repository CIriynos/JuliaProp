import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5

Nr = 4000
Δr = 0.2
l_num = 15
Δt = 0.05
Z = 1.0
po_func_r = coulomb_potiential_zero_fixed(Rco=50.0)
rmax = Nr * Δr
absorb_func = absorb_boundary_r(rmax, 600.0)

pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# init wave
init_wave_list = itp_fdsh(pw, rt, err=1e-9);
crt_shwave = deepcopy(init_wave_list[1]);

# Define laser here.
E0 = 0.0533799
omega = 0.114
nc = 2.0
steps = Int64((2 * nc * pi / omega) ÷ Δt) * 1
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
Ri_tsurf = 600.0
phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);

tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true)

tsurf_get_average_momentum_vector_parallel(a_tsurff_vec, k_space)