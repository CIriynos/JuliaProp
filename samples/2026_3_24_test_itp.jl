import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW



# Basic Parameters
Nr =            5000 * 1            # number of radial grid points
Δr =            0.2 / 1             # radial grid step size
l_num =         50                  # number of angular momentum components
Δt =            0.05 / 1            # time step size
Z =             1.0                 # nuclear charge
po_func(r) =    -1 / r              # potential function
rmax =          Nr * Δr     
absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function
Ri_tsurf        = rmax * 0.7        # radius for t-surf method


# Create Physical World & Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# Initial Wave
# k_num = 100
k_num = 500
init_wave_list = itp_fdsh(pw, rt, k=k_num, err=1e-8);           # get initial wavefunctions by imaginary time propagation
crt_shwave = deepcopy(init_wave_list[1]);               # set the current wavefunction as the ground state
for i in 1: k_num
    println("Energy of the $i-th state: ", get_energy_sh(init_wave_list[i], rt, pw.shgrid))
end

# Store Data Manually
# example_name = "2026_3_24_test_itp"
example_name = "2026_3_24_test_itp_k_num_$k_num"
h5open("./data/$example_name.h5", "w") do file
    for i in 1:k_num
        write(file, "energy_state_$i", hcat(init_wave_list[i]...))
    end
end

###########################

# example_name = "2026_3_24_test_itp"
example_name = "2026_3_24_test_itp_k_num_$k_num"
eigen_states = Vector{shwave_t}(undef, k_num)

for i in 1: k_num
    eigen_states[i] = retrieve_obj(example_name, "energy_state_$i")
    println("Energy of the $i-th state: ", get_energy_sh(eigen_states[i], rt, pw.shgrid))
end

