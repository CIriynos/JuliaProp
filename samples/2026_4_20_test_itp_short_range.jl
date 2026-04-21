import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW


# Basic Parameters
Nr =            20000 * 2           # number of radial grid points
Δr =            0.2 / 2             # radial grid step size
l_num =         50                  # number of angular momentum components
Δt =            0.05 / 2            # time step size
Z =             1.0                  # nuclear charge
# po_func(r) =    -1 / r              # potential function
po_func(r) =    -1 / r * exp(- r * r / (20.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
rmax =          Nr * Δr     
absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function
Ri_tsurf        = rmax * 0.7        # radius for t-surf method


# get the Ip of the system
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func);
rt = create_tdse_rt_sh(pw);
max_k = 4
ori_wave_list = itp_fdsh_special(pw, rt, k=max_k, err=1e-5)   # a special version of the imaginary time propagation for getting the Ip of the system, which uses a more strict error threshold and logs the energy during the iteration

ek_list = []
for k = 1: max_k
    ek = get_energy_sh(ori_wave_list[k], rt, pw.shgrid)
    if ek > 0
        continue
    end
    push!(ek_list, ek)
    println("Energy of the state with k=$k: ", ek)
end


# example_name = "2026_4_20_test_itp_short_range"
example_name = "2026_4_20_test_itp_short_range_dense" # ratio = 2

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