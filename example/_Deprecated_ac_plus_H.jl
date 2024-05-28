import Pkg
Pkg.activate(".")

using Revise
using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using Printf

println("Number of Threads:")
println(Threads.nthreads())

# Iterating All Tasks with gamma
p = []
angles = []
gamma_list = reverse(0.5: 0.25: 3.0)
gamma_list_organized = [gamma_list[1:2:length(gamma_list)]; gamma_list[2:2:length(gamma_list)]]
gamma_list_organized = [gamma for gamma in gamma_list_organized if gamma > 1.5]


for gamma in gamma_list_organized

    l_num_by_gamma = 100
    if gamma > 1.5
        l_num_by_gamma = 60
    elseif gamma > 1.0
        l_num_by_gamma = 80
    else
        l_num_by_gamma = 100
    end

    # Basic Parameters
    Nr = 5000
    Δr = 0.2
    l_num = l_num_by_gamma
    Δt = 0.05
    Z = 1.0
    po_func_r = coulomb_potiential_zero_fixed_plus(Rco = 40.0)
    rmax = Nr * Δr  # rmax = 1000.0
    absorb_func = absorb_boundary_r(rmax, 800.0, pow_value=8.0, max_value=100.0)
    pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
    rt = create_tdse_rt_sh(pw);

    # Initial Wave
    init_wave_list = itp_fdsh(pw, rt, err=1e-9);
    crt_shwave = deepcopy(init_wave_list[1]);
    get_energy_sh(init_wave_list[1], rt, pw.shgrid) # He:-0.944  H:-0.5

    # Define Laser Here.
    rate = 0.0532 / 2.0     # E = 0.0534 <=> I = 2.0 (×10^14 W⋅cm-2)
    # E0 = rate * I0
    omega = 0.062           # λ = 735 nm
    eps = 0.87
    # Ip = 0.944            # ion energy
    Ip = 0.5
    # gamma = omega * sqrt(2 * Ip) / E0
    E0 = omega * sqrt(2 * Ip) / gamma   # get E0 from gamma
    I0 = E0 / rate
    laser_duration = 289    # 7 fs
    nc = Int64(floor(0.5 * omega * laser_duration / (π - 2 * asin(sqrt(1 / 2)))))  # evaluate nc by laser's duration (1/2 criterion)
    # nc = Int64(floor(0.5 * omega * laser_duration / (π - 2 * asin((1 / 2) ^ (1/8)))))
    steps = Int64((2 * nc * pi / omega) ÷ Δt)
    actual_duration = steps * Δt
    t_linspace = create_linspace(steps, Δt)

    E0x = E0 * (1.0 / sqrt(eps ^ 2 + 1.0))
    E0y = -E0 * (eps / sqrt(eps ^ 2 + 1.0))

    Ax(t) = (E0x / omega) * (sin(omega * t / 2.0 / nc) ^ 2) * sin(omega * t) * (t < (2 * nc * pi / omega))
    Ay(t) = (E0y / omega) * (sin(omega * t / 2.0 / nc) ^ 2) * cos(omega * t) * (t < (2 * nc * pi / omega))
    # Ax(t) = (E0x / omega) * (sin(omega * t / 2.0 / nc) ^ 8) * sin(omega * t) * (t < (2 * nc * pi / omega))
    # Ay(t) = (E0y / omega) * (sin(omega * t / 2.0 / nc) ^ 8) * cos(omega * t) * (t < (2 * nc * pi / omega))

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
    Ri_tsurf = 800.0
    
    # phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);
    # a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space, TSURF_MODE_ELLI);
    # tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true)

    # # store data
    # gamma_str = @sprintf("%.2f", gamma)
    # example_name = "ac_plus_H_sin2_L$(l_num)_gamma$(gamma_str)_cos2po"
    # h5open("./data/$example_name.h5", "w") do file
    #     write(file, "crt_shwave", hcat(crt_shwave...))
    #     write(file, "phi_record", hcat(phi_record...))
    #     write(file, "dphi_record", hcat(dphi_record...))
    #     write(file, "a_tsurff_vec", a_tsurff_vec)
    # end


    # retrieve data.
    gamma_str = @sprintf("%.2f", gamma)
    example_name = "ac_plus_H_sin2_L$(l_num)_gamma$(gamma_str)_cos2po"
    crt_shwave = retrieve_obj(example_name, "crt_shwave")
    phi_record = retrieve_obj(example_name, "phi_record")
    dphi_record = retrieve_obj(example_name, "dphi_record")
    a_tsurff_vec = retrieve_mat(example_name, "a_tsurff_vec")

    println(length(crt_shwave))
    push!(p, tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=false))

    # Look the Angles
    println("Now $gamma_str")
    a_vec_phi = zeros(Float64, Nk_phi+1)
    for (p, k_vec) in enumerate(k_space.k_collection)
        i, _, k = k_space.ijk_mapping[p]
        # println(k)
        a_vec_phi[k] += norm(a_tsurff_vec[p]) ^ 2
    end
    plot(a_vec_phi)
    push!(angles, rad2deg(phi_linspace(Nk_phi)[argmax(a_vec_phi)]))

end

p[1]
angles[1]


# p[]
# gamma_list_organized[1]
# # plot(gamma_list_organized[3:5], (angles[3:5] .- 270))

# ids = [3, 9, 4, 10, 5, 11]
# plot(gamma_list_organized[ids], (angles[ids] .- 270))
# p[]


# plot(angles .- 270)
# p[2]
# p[4]


# look
# a_vec_phi = zeros(Float64, Nk_phi+1)
# for (p, k_vec) in enumerate(k_space.k_collection)
#     i, _, k = k_space.ijk_mapping[p]
#     # println(k)
#     a_vec_phi[k] += norm(a_tsurff_vec[p]) ^ 2
# end
# rad2deg(phi_linspace(Nk_phi)[argmax(a_vec_phi)])

# Et_data_x = -get_derivative_two_order(At_data_x, Δr)
# Et_data_y = -get_derivative_two_order(At_data_y, Δr)
# plot(At_data_x, At_data_y)
# GR.setarrowsize(1)
# len = length(Et_data_x)
# f1 = plot(Et_data_x[1:100:len], Et_data_y[1:100:len], arrow=(:closed, 2.0))
# plot!(f1, At_data_x[1:100:len] ./ 32, At_data_y[1:100:len] ./ 32, arrow=(:closed, 2.0))
# plot(t_linspace, [Et_data_x, Et_data_y], title="sin2", label=["Ex" "Ey"])