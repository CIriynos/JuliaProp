import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# 12_7_lvlaoshi_mission


# Basic Parameters
grid_ratio = 1
Nr = 10000 * grid_ratio
Δr = 0.2 / grid_ratio
l_num = 200
Δt = 0.05 / grid_ratio
Z = 1.0
rmax = Nr * Δr  # 2000
Ri_tsurf = 1800.0
po_func_r_shit(r) = -1.0 * (r^2 + 0.01) ^ (-0.5) * flap_top_windows_f(r, 0, Ri_tsurf, 1/4, left_flag=false)
absorb_func = absorb_boundary_r(rmax, Ri_tsurf, pow_value=8.0, max_value=100.0)


# Create Physical World and Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r_shit, Z, absorb_func)
rt = create_tdse_rt_sh(pw);


# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-8);
crt_shwave = deepcopy(init_wave_list[1]);
en = get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5
println("Energy of ground state: ", en)


# Define the Laser. 
eps = 0.3
ω1 = 0.05
gamma = 0.3
Ip = 0.5
E1 = ω1 * sqrt(2 * Ip) / gamma   # get E0 from gamma
E1x = E1 * (1.0 / sqrt(eps ^ 2 + 1.0))
E1y = E1 * (-eps / sqrt(eps ^ 2 + 1.0))
nc1 = 8
Tp = 2 * nc1 * pi / ω1

Ax(t) = (E1x / ω1) * (sin(ω1 * t / 2.0 / nc1) ^ 8) * sin(ω1 * (t - Tp/2)) * (t >= 0 && t <= Tp)
Ay(t) = (E1y / ω1) * (sin(ω1 * t / 2.0 / nc1) ^ 8) * cos(ω1 * (t - Tp/2)) * (t >= 0 && t <= Tp)

steps = Int64(Tp ÷ Δt)
t_linspace = create_linspace(steps, Δt)
At_data_x = Ax.(t_linspace)
At_data_y = Ay.(t_linspace)
plot(t_linspace, [At_data_x At_data_y],
    labels=["Ax" "Ay"], xlabel="Time a.u.",
    ylabel="Vector Potential A(t) a.u.")

# # Propagation
# crt_shwave = deepcopy(init_wave_list[1])
# phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# # Store Data
# example_name = "12_7_lvlaoshi_mission"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
# end

# Define k Space
k_delta = 0.01
kmin = 0.01
kmax = 2.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))

# Retrieve Data.
example_name = "12_7_lvlaoshi_mission"
crt_shwave = retrieve_obj(example_name, "crt_shwave")
phi_record = retrieve_obj(example_name, "phi_record")
dphi_record = retrieve_obj(example_name, "dphi_record")

δa_lm = isurf_rest_part(crt_shwave, k_space.k_r_range, last(t_linspace), Ri_tsurf, pw, rt)

tsurf_plot_xy_momentum_spectrum(δa_lm, k_space, pw, kr_flag=false, k_min=0.2)
