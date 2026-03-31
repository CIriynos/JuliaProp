import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW



# Basic Parameters
Nr =            20000 * 1            # number of radial grid points
Δr =            0.2 / 1             # radial grid step size
l_num =         25                  # number of angular momentum components
Δt =            0.05 / 1            # time step size
Z =             1.0                 # nuclear charge
po_func(r) =    -1 / r              # potential function
rmax =          Nr * Δr     
absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function
Ri_tsurf        = rmax * 0.7        # radius for t-surf method

# get the Ip of the system
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func);
rt = create_tdse_rt_sh(pw);
Ip = -get_energy_sh(itp_fdsh(pw, rt, k=1, err=1e-15)[1], rt, pw.shgrid)

max_energy_level = 10
k_num = sum([i^2 for i in 1: max_energy_level])
ek_list = [-Ip / n^2 for n in 1: max_energy_level]
# occ_list = get_k_occupation_list(k_num, pw.l_num)

example_name = "2026_3_28_test_itp_new_approach"

rs = get_linspace(pw.shgrid.rgrid)
h5open("./data/$example_name.h5", "w") do file
    k = 1
    for (n, ek) in enumerate(ek_list)
        pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func, delta_t_im = 2 / (-ek))
        rt = create_tdse_rt_sh(pw);

        for l in 0: min(n-1, l_num-1)
            for m in -l: l
                id = get_index_from_lm(l, m, l_num)
                init_wave = create_empty_shwave(pw.shgrid)
                @. init_wave[id] = rs * exp(-rs * n)
                itp_fdsh_single(pw, rt, init_wave, id, err=1e-10, log_info=false)
                en = get_energy_sh(init_wave, rt, pw.shgrid)

                write(file, "energy_state_k_$k", (init_wave[id]))
                println("(k = $k) Energy of the state with n=$n, l=$l, m=$m: ", en)

                k += 1
            end
        end
    end
end