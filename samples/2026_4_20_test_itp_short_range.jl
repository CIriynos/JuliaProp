import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW


# Basic Parameters
ratio = 1
# Nr =            20000 * ratio           # number of radial grid points
Nr =            40000 * ratio           # number of radial grid points
Δr =            0.2 / ratio             # radial grid step size
l_num =         200                  # number of angular momentum components
Δt =            0.05 / ratio            # time step size
Z =             1.0                  # nuclear charge
# po_func(r) =    -1 / r              # potential function
po_func(r) =    -1 / r * exp(- r * r / (20.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
rmax =          Nr * Δr     
absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function
Ri_tsurf        = rmax * 0.7        # radius for t-surf method


# get the Ip of the system
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func);
# rt = create_tdse_rt_sh(pw, m_zero_flag=true);
# max_k = 4
# ori_wave_list = itp_fdsh_special(pw, rt, k=max_k, err=1e-5)   # a special version of the imaginary time propagation for getting the Ip of the system, which uses a more strict error threshold and logs the energy during the iteration

# ek_list = []
# for k = 1: max_k
#     ek = get_energy_sh(ori_wave_list[k], rt, pw.shgrid)
#     if ek > 0
#         continue
#     end
#     push!(ek_list, ek)
#     println("Energy of the state with k=$k: ", ek)
# end

# Energy of the state with k=1: -0.49643369499243595
# Energy of the state with k=2: -0.11338866728255607
# Energy of the state with k=3: -0.03459184076770206
# Energy of the state with k=4: -0.002465088974768024

ek_list = [-0.49643369499243595, -0.11338866728255607, -0.03459184076770206, -0.002465088974768024]

rs = get_linspace(pw.shgrid.rgrid)

pw = nothing
rt = nothing
GC.gc()

# example_name = "2026_4_20_test_itp_short_range"
example_name = "2026_4_20_test_itp_short_range_large_r_box"

m_zero_flag = true

h5open("./data/$example_name.h5", "w") do file
    k = 1
    for (n, ek) in enumerate(ek_list)
        pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func, delta_t_im = 2 / (-ek))
        rt = create_tdse_rt_sh(pw, m_zero_flag=true);

        for l in 0: min(n-1, l_num-1)
            for m in -l: l
                if m_zero_flag == true && m != 0
                    id = get_index_from_lm(l, m, l_num)
                    init_wave = create_empty_shwave(pw.shgrid)
                    write(file, "energy_state_k_$k", (init_wave[id]))
                else
                    id = get_index_from_lm(l, m, l_num)
                    init_wave = create_empty_shwave(pw.shgrid)
                    @. init_wave[id] = rs * exp(-rs * n)
                    itp_fdsh_single(pw, rt, init_wave, id, err=1e-10, log_info=false)
                    en = get_energy_sh(init_wave, rt, pw.shgrid)

                    write(file, "energy_state_k_$k", (init_wave[id]))
                    println("(k = $k) Energy of the state with n=$n, l=$l, m=$m: ", en)
                end
                k += 1
            end
        end

        pw = nothing
        rt = nothing
        GC.gc()
    end
end