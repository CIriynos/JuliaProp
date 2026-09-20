import Pkg
Pkg.activate(".")
# using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

function main()

    # Basic Parameters
    ratio = 1
    Nr =            20000 * ratio        # number of radial grid points
    Δr =            0.2 / ratio          # radial grid step size
    l_num =         50                  # number of angular momentum components
    Δt =            0.05 / ratio         # time step size
    Z =             1.0                  # nuclear charge
    po_func(r) =    -1 / r
    # po_func(r) =    -1 / r * exp(- r * r / (5.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
    rmax =          Nr * Δr     
    absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function
    Ri_tsurf        = rmax * 0.7        # radius for t-surf method

    # get the Ip of the system
    pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func);
    rt = create_tdse_rt_sh(pw, m_zero_flag=true);
    init_wave = create_empty_shwave(pw.shgrid)

    # max_k = 2
    max_k = 5

    ek_list = []
    for k = 1: max_k
        init_wave = create_empty_mzero_shwave(pw.shgrid)
        rs = get_linspace(pw.shgrid.rgrid)
        @. init_wave[k] = rs * exp(-rs * k)
        itp_fdsh_single(pw, rt, init_wave, k, err=1e-15, log_info=false)
        ek = get_energy_sh_so(init_wave, rt, k)

        # ek = -0.5 / (k ^ 2)
        push!(ek_list, ek)
        println("Energy of the state with k=$k: ", ek)

        init_wave = nothing 
        GC.gc(true)
        ccall(:malloc_trim, Cint, (Csize_t,), 0)
    end

    rs = get_linspace(pw.shgrid.rgrid)
    pw = nothing
    rt = nothing
    GC.gc(true)
    ccall(:malloc_trim, Cint, (Csize_t,), 0)

    # example_name = "2026_6_15_itp_short_range_5"
    example_name = "2026_6_15_itp"

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
                        itp_fdsh_single(pw, rt, init_wave, id, log_info=false, mininum_loop_times=2000)
                        en = get_energy_sh(init_wave, rt, pw.shgrid)

                        write(file, "energy_state_k_$k", (init_wave[id]))
                        println("(k = $k) Energy of the state with n=$n, l=$l, m=$m: ", en)
                    end
                    k += 1

                    init_wave = nothing 
                    GC.gc(true)
                    ccall(:malloc_trim, Cint, (Csize_t,), 0)
                end
            end

            println("ended procedure.")
            pw = nothing
            rt = nothing
            GC.gc(true)
            ccall(:malloc_trim, Cint, (Csize_t,), 0)
            println("finished memory clear.")
        end
    end
end

main()