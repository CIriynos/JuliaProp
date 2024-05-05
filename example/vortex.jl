import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5

Nr = 10000
Δr = 0.2
l_num = 25
Δt = 0.05
Z = 1.0
po_func_r = coulomb_potiential_zero_fixed()
rmax = Nr * Δr
absorb_func = absorb_boundary_r(rmax)

pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# init wave
init_wave_list = itp_fdsh(pw, rt, err=1e-9);
crt_shwave = deepcopy(init_wave_list[1]);

# two-color laser here
ω1 = 0.114
ω2 = 0.190
E1x = 0.0267
E2x = 0.0267
E1y = 0.0267
E2y = 0.0267
nc1 = 6.0
nc2 = 10.0
phase_1x = 0.0
phase_2x = 0.0
phase_1y = 0.5pi
phase_2y = 1.5pi
tau = 331.0

Ax(t) = (E1x / ω1) * sin(ω1 * t / 2 / nc1)^2 * sin(ω1 * t + phase_1x) * (t > 0 && t < (2 * nc1 * pi / ω1)) + 
    (E2x / ω2) * sin(ω2 * (t - tau) / 2 / nc2)^2 * sin(ω2 * (t - tau) + phase_2x) * (t - tau > 0 && t - tau < (2 * nc2 * pi / ω2))
Ay(t) = (E1y / ω1) * sin(ω1 * t / 2 / nc1)^2 * sin(ω1 * t + phase_1y) * (t > 0 && t < (2 * nc1 * pi / ω1)) + 
    (E2y / ω2) * sin(ω2 * (t - tau) / 2 / nc2)^2 * sin(ω2 * (t - tau) + phase_2y) * (t - tau > 0 && t - tau < (2 * nc2 * pi / ω2))

steps_1 = Int64((2 * nc1 * pi / ω1) ÷ Δt)
steps_2 = Int64((2 * nc2 * pi / ω2) ÷ Δt) + Int64(tau ÷ Δt)
steps = max(steps_1, steps_2)
t_linspace = create_linspace(steps, Δt)

At_data_x = Ax.(t_linspace)
At_data_y = Ay.(t_linspace)
At_data_z = zeros(Float64, steps)

# define k space
k_delta = 0.01
kmin = 0.01
kmax = 0.6
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))

# propagation
Ri_tsurf = 1000.0

# phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);

# example_name = "vortex"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
#     write(file, "a_tsurff_vec", a_tsurff_vec)
# end


########################
# retrieve data.

crt_shwave = retrieve_obj("vortex", "crt_shwave")
phi_record = retrieve_obj("vortex", "phi_record")
dphi_record = retrieve_obj("vortex", "dphi_record")
a_tsurff_vec = retrieve_mat("vortex", "a_tsurff_vec")

# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);
tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true)