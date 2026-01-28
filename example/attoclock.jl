import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW


# Basic Parameters
Nr =            5000            # number of radial grid points
Δr =            0.2             # radial grid step size
l_num =         25              # number of angular momentum components
Δt =            0.05            # time step size
Z =             1.0             # nuclear charge
po_func =       coulomb_potiential_zero_fixed_plus(Rco=50.0) # potential function
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

# Define the Laser pulse
E_fs =          0.05338             # peak electric field of the fs pulse
ω_fs =          0.114               # angular frequency of the fs pulse
nc =            2.0                 # number of optical cycles in the fs pulse
steps = Int64((2 * nc * pi / ω_fs) ÷ Δt)
                                    # number of steps for TDSE evolution

# create the Laser pulse data
Ax(t) = (E_fs / ω_fs) * (sin(ω_fs * t / 2.0 / nc) ^ 2) * sin(ω_fs * t) * (t < (2 * nc * pi / ω_fs))
Ay(t) = (E_fs / ω_fs) * (sin(ω_fs * t / 2.0 / nc) ^ 2) * cos(ω_fs * t) * (t < (2 * nc * pi / ω_fs))
ts = create_linspace(steps, Δt)
At_data_x = Ax.(ts)
At_data_y = Ay.(ts)
At_data_z = zeros(Float64, steps)

# define k space
k_delta = 0.01
kmin = 0.01
kmax = 1.0
ks = kmin: k_delta: kmax
Nk_phi = 200
k_space = create_k_space(ks, fixed_theta(pi/2), phi_linspace(Nk_phi))

###########################

# # Main Propagation Loop of TDSE with t-surf Recording
# phi_record, dphi_record = tdse_elli_sh_mainloop_record_xy_optimized(crt_shwave, pw, rt, At_data_x, At_data_y, steps, Ri_tsurf);

# # i-surf procedure
# a_tsurff_vec = isurf_sh_vector(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, ts, k_space, TSURF_MODE_ELLI);

# # Store Data Manually
# example_name = "attoclock"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
#     write(file, "a_tsurff_vec", a_tsurff_vec)
# end

########################

# retrieve data.
crt_shwave = retrieve_obj("attoclock", "crt_shwave")
phi_record = retrieve_obj("attoclock", "phi_record")
dphi_record = retrieve_obj("attoclock", "dphi_record")
a_tsurff_vec = retrieve_mat("attoclock", "a_tsurff_vec")

# plot the figure
tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space, kr_flag=true, kr_min=0.08)

# Note:
# This figure should be as the same as Fig.1 in Paper: V. Tulsky and D. Bauer / Computer Physics Communications 251 (2020) 107098 