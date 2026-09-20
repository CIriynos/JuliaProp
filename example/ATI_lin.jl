import Pkg
Pkg.activate(".")

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW


# Basic Parameters
Nr =            10000            # number of radial grid points
Δr =            0.2             # radial grid step size
l_num =         50              # number of angular momentum components
Δt =            0.05            # time step size
Z =             1.0             # nuclear charge
# po_func(r)  = -1 / r
# po_func =       coulomb_potiential_zero_fixed_plus(Rco=100.0) # potential function
po_func(r) =    -1 / r * exp(- r * r / (20.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
# po_func(r) =    -1 / r * exp(- r * r / (5.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential

rmax =          Nr * Δr     
Ri_tsurf =      rmax * 0.7      # radius for t-surf method
absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function


# Create Physical World & Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func)
rt = create_tdse_rt_sh(pw);


# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-30);           # get initial wavefunctions by imaginary time propagation
crt_shwave = deepcopy(init_wave_list[1]);               # set the current wavefunction as the ground state
en = get_energy_sh(init_wave_list[1], rt, pw.shgrid)    # get energy of the initial wavefunction (For H atom, should be -0.5 a.u.)

# define laser field
E_fs =          0.05               # peak electric field of the fs pulse
ω_fs =          0.057 * 0.6            # angular frequency of the fs pulse
nc =            12                  # number of optical cycles in the fs pulse
Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, 0, ellipticity=0.0, phase1=0.5pi)        # create the light pulse from the given parameters (+ ellipticity)
At_datas, Et_datas, ts, steps = create_tdata(tmax, 0, Δt, no_light, no_light, t -> Ex_fs(t), appendix_steps=1)
                                                                                        # create the vector potential and electric field data for propagation

At_data_x = At_datas[1]
At_data_y = At_datas[2]
At_data_z = At_datas[3]

# define k space
k_delta = 0.01
kmin = k_delta
kmax = 2.0
ks = kmin: k_delta: kmax
Nk_theta = 200
k_space = create_k_space(ks, (theta_linspace(Nk_theta)), fixed_theta(pi/2))

###########################

# # Main Propagation Loop of TDSE with t-surf Recording
# phi_record, dphi_record = tdseln_sh_mainloop_record_optimized(crt_shwave, pw, rt, At_data_z, steps, Ri_tsurf);

# # i-surf procedure
# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, ts, k_space, TSURF_MODE_PL, is_tsurf_added=false);

# # Store Data Manually
# example_name = "ATI_lin"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
#     write(file, "a_tsurff_vec", a_tsurff_vec)
# end

########################

# retrieve data.
crt_shwave = retrieve_obj("ATI_lin", "crt_shwave")
phi_record = retrieve_obj("ATI_lin", "phi_record")
dphi_record = retrieve_obj("ATI_lin", "dphi_record")
a_tsurff_vec = retrieve_mat("ATI_lin", "a_tsurff_vec")

# plot the figure
tsurf_plot_xz_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=false, kr_min=0.05)


kr_flag = false
kr_min = 0.02
Nk_r = length(k_space.k_r_range)
Nk_theta = length(k_space.k_theta_range)
k_r_range = k_space.k_r_range
k_theta_range = k_space.k_theta_range

pes_mat = zeros(Float64, Nk_r, Nk_theta)
for (p, k_vec) in enumerate(k_space.k_collection)
    i, j, _ = k_space.ijk_mapping[p]
    pes_mat[i, j] = (kr_flag == true ? norm(k_vec) : 1.0) * norm(a_tsurff_vec[p]) ^ 2 * (norm(k_vec) > kr_min)
end

gr()
# colormap = cgrad([:white, :black, :red, :blue, :white])
colormap = cgrad([:white, palette(:jet1, 6)...], rev = true)
ddd = clamp.(abs.(log10.(pes_mat ./ maximum(pes_mat))), 0, 3)
fig = Plots.heatmap(k_theta_range, k_r_range, ddd; color = colormap, projection = :polar)
Plots.heatmap!(fig, k_theta_range .- pi, k_r_range, ddd[:, end:-1:1]; color = colormap, projection = :polar)

colormap_2 = cgrad(["#710000", "#C60001", "#FB0001", "#FE3A01", "#FF7A00", "#FEAB01", "#FEAB01", "#FEAB01", "#D5FF17", "#91FF60", "#4AFFAE", "#02FFFC", "#00A3FE", "#003DFF", "#2120FF", "#8C8CFE", :white], rev=true)
heatmap([k_theta_range .+ π/2; k_theta_range .+ 3π/2], k_r_range[1: end ÷ 2], [norm.(pes_mat) norm.(pes_mat)[:, end:-1:1]][1: end ÷ 2, :], projection=:polar, color=colormap_2)

heatmap([k_theta_range .+ π/2; k_theta_range .+ 3π/2], k_r_range[1: end ÷ 2], [norm.(pes_mat)[:, end:-1:1] norm.(pes_mat)][1: end ÷ 2, :], projection=:polar, color=colormap_2)



gr()
# colormap = cgrad([:white, :black, :red, :blue, :white])
colormap = cgrad([:white, palette(:jet1, 6)...], rev = true)
ddd = clamp.(abs.(log10.(pes_mat ./ maximum(pes_mat))), 0, 3)
# fig = Plots.heatmap(k_theta_range, k_r_range, ddd; color = colormap, projection = :polar)
# Plots.heatmap!(fig, k_theta_range .- pi, k_r_range, ddd[:, end:-1:1]; color = colormap, projection = :polar)

fig = Plots.heatmap(k_theta_range, k_r_range, ddd[:, end:-1:1]; color = colormap, projection = :polar)
Plots.heatmap!(fig, k_theta_range .- pi, k_r_range, ddd; color = colormap, projection = :polar)
