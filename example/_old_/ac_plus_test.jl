import Pkg
Pkg.activate(".")

using Revise
using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using Printf
println("Number of Threads: $(Threads.nthreads())")

ps = []
angles = []
a_phi_list = []
ttt_list = []
input_sheet = [(1, 2), (1, 8), (2, 2), (2, 8), (4, 2), (4, 8), (6, 2), (6, 8)]

for (ratio, sinpow) in input_sheet

# Basic Parameters
Nr = 10000
Δr = 0.2
l_num = 50
Δt = 0.05
Z = 1.0
Ri_tsurf = 1500.0
po_func_r = coulomb_potiential_zero_fixed_COS(1200.0, 1500.0)
rmax = Nr * Δr  # rmax = 2000.0
absorb_func = absorb_boundary_r(rmax, 1500.0, pow_value=8.0, max_value=100.0)
# pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
# rt = create_tdse_rt_sh(pw);


# # Initial Wave
# init_wave_list = itp_fdsh(pw, rt, err=1e-9);
# crt_shwave = deepcopy(init_wave_list[1]);
# get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5


# Define Laser Here.
sin_envelope_pow = sinpow    # sin(...)^2 or sin(...)^8
gamma = 2.5             # gamma = omega * sqrt(2 * Ip) / E0
rate = 0.0532 / 2.0     # E = 0.0534 <=> I = 2.0 (×10^14 W⋅cm-2), E0 = rate * I0
omega = 0.062           # λ = 735 nm
eps = 0.87
# Ip = 0.944            # ion energy
Ip = 0.5
E0 = omega * sqrt(2 * Ip) / gamma   # get E0 from gamma
I0 = E0 / rate
laser_duration = 289 * ratio    # 7 fs 
nc = Int64(floor(0.5 * omega * laser_duration / (π - 2 * asin((1 / 2) ^ (1 / sin_envelope_pow)))))  # evaluate nc by laser's duration (1/2 criterion)
steps = Int64((2 * nc * pi / omega) ÷ Δt)
actual_duration = steps * Δt
t_linspace = create_linspace(steps, Δt)

E0x = E0 * (1.0 / sqrt(eps ^ 2 + 1.0))
E0y = E0 * (eps / sqrt(eps ^ 2 + 1.0))
Ax(t) = (E0x / omega) * (cos(omega * t / 2.0 / nc - pi / 2) ^ sin_envelope_pow) * sin(omega * (t - nc*pi/omega)) * (t < (2 * nc * pi / omega))
Ay(t) = (E0y / omega) * (cos(omega * t / 2.0 / nc - pi / 2) ^ sin_envelope_pow) * cos(omega * (t - nc*pi/omega)) * (t < (2 * nc * pi / omega))

At_data_x = Ax.(t_linspace)
At_data_y = Ay.(t_linspace)
At_data_z = zeros(Float64, steps)


# Define k Space
k_delta = 0.01
kmin = 0.01
kmax = 2.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))


# # Start Propagation
# phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);
# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);
# tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true)

# # Store Data
# gamma_str = @sprintf("%.2f", gamma)
# rmax_str = @sprintf("%.2f", rmax)
# example_name = "ac_plus_test_sin$(sin_envelope_pow)_L$(l_num)_Rmax$(rmax_str)_gamma$(gamma_str)_ratio$(ratio)"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
#     write(file, "a_tsurff_vec", a_tsurff_vec)
# end


# Retrieve Data.
gamma_str = @sprintf("%.2f", gamma)
rmax_str = @sprintf("%.2f", rmax)
example_name = "ac_plus_test_sin$(sin_envelope_pow)_L$(l_num)_Rmax$(rmax_str)_gamma$(gamma_str)_ratio$(ratio)"
crt_shwave = retrieve_obj(example_name, "crt_shwave")
phi_record = retrieve_obj(example_name, "phi_record")
dphi_record = retrieve_obj(example_name, "dphi_record")
a_tsurff_vec = retrieve_mat(example_name, "a_tsurff_vec")

# Plot and Get the Angle
p = tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=false)
a_vec_phi = zeros(Float64, Nk_phi+1)
ttt = zeros(Float64, length(k_linspace))
for (p, k_vec) in enumerate(k_space.k_collection)
    i, _, k = k_space.ijk_mapping[p]
    # println(k)
    # if k_vec[1] < 0.05
    #     continue
    # end
    a_vec_phi[k] += norm(a_tsurff_vec[p]) ^ 2
    if k == 60
        ttt[i] = norm(a_tsurff_vec[p]) ^ 2
    end
end
angle = 270 - rad2deg(phi_linspace(Nk_phi)[argmax(a_vec_phi)])

push!(ps, p)
push!(angles, angle)
push!(a_phi_list, a_vec_phi)
push!(ttt_list, ttt)

end



k_delta = 0.01
kmin = 0.01
kmax = 2.0
k_linspace = kmin: k_delta: kmax
Nk_phi = 360
k_space = create_k_space(k_linspace, fixed_theta(pi/2), phi_linspace(Nk_phi))
# Retrieve Data.
# gamma_str = @sprintf("%.2f", gamma)
# rmax_str = @sprintf("%.2f", rmax)
example_name = "ac_plus_H_sin2_L80_Rmax2000.00_gamma1.50"
# example_name = "ac_plus_H_sin2_L100_Rmax2000.00_gamma0.7"
crt_shwave = retrieve_obj(example_name, "crt_shwave")
phi_record = retrieve_obj(example_name, "phi_record")
dphi_record = retrieve_obj(example_name, "dphi_record")
a_tsurff_vec = retrieve_mat(example_name, "a_tsurff_vec")

# Plot and Get the Angle
tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true)

a_vec_phi = zeros(Float64, Nk_phi+1)
ttt = zeros(Float64, length(k_linspace))
for (p, k_vec) in enumerate(k_space.k_collection)
    i, _, k = k_space.ijk_mapping[p]
    # println(k)
    # if k_vec[1] < 0.05
    #     continue
    # end
    a_vec_phi[k] += norm(a_tsurff_vec[p]) ^ 2
    if k == 60
        ttt[i] = norm(a_tsurff_vec[p]) ^ 2
    end
end
angle = 270 - rad2deg(phi_linspace(Nk_phi)[argmax(a_vec_phi)])



# # Check the Laser Field
# Et_data_x = -get_derivative_two_order(At_data_x, Δr)
# Et_data_y = -get_derivative_two_order(At_data_y, Δr)
# plot(At_data_x, At_data_y)
# GR.setarrowsize(1)
# len = length(Et_data_x)
# f1 = plot(Et_data_x[1:100:len], Et_data_y[1:100:len], arrow=(:closed, 2.0))
# # plot!(f1, At_data_x[1:100:len] ./ 32, At_data_y[1:100:len] ./ 32, arrow=(:closed, 2.0))
# plot(t_linspace, [Et_data_x, Et_data_y], title="sin $(sin_envelope_pow)", label=["Ex" "Ey"])
# plot(t_linspace, [At_data_x, At_data_y], title="sin $(sin_envelope_pow)", label=["Ax" "Ay"])
