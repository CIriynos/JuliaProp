import Pkg
Pkg.activate(".")

using Revise
using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using Printf

Nr = 10000
Δr = 0.2
l_num = 40
Δt = 0.05
Z = 1.9
po_func_r = coulomb_potiential_helium_zero_fixed_plus(Rco = 100.0)
rmax = Nr * Δr  # rmax=1000.0
absorb_func = absorb_boundary_r(rmax, 1600.0, pow_value=8.0, max_value=50.0)

pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# init wave
init_wave_list = itp_fdsh(pw, rt, err=1e-9);
crt_shwave = deepcopy(init_wave_list[1]);

get_energy_sh(init_wave_list[1], rt, pw.shgrid) # -0.944

# Define laser here.
# E0 = 0.0534     # 0.0534 <=> I = 2.0 (×10^14 W⋅cm-2)
E0 = 0.0534
omega = 0.062   # λ = 735 nm
eps = 0.87
Ip = 0.944      # ion energy
gamma = omega * sqrt(2 * Ip) / E0
laser_duration = 289    # 7 fs
nc = Int64(floor(0.5 * omega * laser_duration / (π - 2 * asin(sqrt(1 / exp(1))))))  # evaluate nc by laser's duration (1/e criterion)
steps = Int64((2 * nc * pi / omega) ÷ Δt)
actual_duration = steps * Δt
t_linspace = create_linspace(steps, Δt)

E0x = E0 * (1.0 / sqrt(eps ^ 2 + 1.0))
E0y = E0 * (eps / sqrt(eps ^ 2 + 1.0))

Ax(t) = (E0x / omega) * (sin(omega * t / 2.0 / nc) ^ 2) * sin(omega * t) * (t < (2 * nc * pi / omega))
Ay(t) = (E0y / omega) * (sin(omega * t / 2.0 / nc) ^ 2) * cos(omega * t) * (t < (2 * nc * pi / omega))
At_data_x = Ax.(t_linspace)
At_data_y = Ay.(t_linspace)
At_data_z = zeros(Float64, steps)

# define k space
k_delta = 0.01
kmin = 0.1
kmax = 2.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))

# propagation
Ri_tsurf = 1500.0
# phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);

# # tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true)

# # store data
# gamma_str = @sprintf("%.2f", gamma)
# example_name = "ac_plus_gamma_$gamma_str"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
#     write(file, "a_tsurff_vec", a_tsurff_vec)
# end


########################
# retrieve data.
gamma_str = @sprintf("%.2f", gamma)
example_name = "ac_plus_gamma_$gamma_str"
crt_shwave = retrieve_obj(example_name, "crt_shwave")
phi_record = retrieve_obj(example_name, "phi_record")
dphi_record = retrieve_obj(example_name, "dphi_record")
a_tsurff_vec = retrieve_mat(example_name, "a_tsurff_vec")

# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);
tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true)
