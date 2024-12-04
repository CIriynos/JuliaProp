import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW


# Basic Parameters
grid_ratio = 1
Nr = 5000 * grid_ratio
Δr = 0.2 / grid_ratio
l_num = 60
Δt = 0.05 / grid_ratio
Z = 1.0
rmax = Nr * Δr  # 1000
Ri_tsurf = 800.0
# po_func_r_shit(r) = -1.0 / r * flap_top_windows_f(r, 0, Ri_tsurf, 1/4, left_flag=false)
po_func_r_shit(r) = -1.33 * (r^2 + 0.5) ^ (-0.5) * flap_top_windows_f(r, 0, Ri_tsurf, 1/4, left_flag=false)
absorb_func = absorb_boundary_r(rmax, Ri_tsurf, pow_value=8.0, max_value=100.0)


# Create Physical World and Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r_shit, Z, absorb_func)
rt = create_tdse_rt_sh(pw);


# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-8, k=10);
crt_shwave = deepcopy(init_wave_list[1]);
for k = 1: 10
    en = get_energy_sh(init_wave_list[k], rt, pw.shgrid) # He:-0.944  H:-0.5
    println("Energy of state $(k): ", en)
end

# HHG 
for k1 = 1: 10
    for k2 = 1: 10

        dU_data = get_derivative_two_order(pw.po_data_r, pw.delta_r)
        # crt_shwave = deepcopy(init_wave_list[k]);
        crt_shwave = [init_wave_list[k1][id] .+ init_wave_list[k2][id] for id = 1: l_num ^ 2]
        integral_buffer = zeros(ComplexF64, pw.l_num ^ 2)
        result::ComplexF64 = 0

        Threads.@threads for id = 1: l_num ^ 2
            l = rt.lmap[id]
            m = rt.mmap[id]
            id1 = get_index_from_lm(l - 1, m - 1, pw.shgrid.l_num)
            id2 = get_index_from_lm(l + 1, m - 1, pw.shgrid.l_num)
            c1 = -sqrt((l + m + 1) * (l + m + 2) / ((2 * l + 1) * (2 * l + 3)))
            c2 = sqrt((l - m) * (l - m - 1) / ((2 * l + 1) * (2 * l - 1)))

            integral_buffer[id] = 0      # clear it first
            if id1 != -1    # if id1 (l - 1, m) is in bound, then add 
                for k = 1: pw.Nr
                    integral_buffer[id] += conj(crt_shwave[id][k]) * crt_shwave[id1][k] * dU_data[k] * c1 * pw.delta_r
                end
            end
            if id2 != -1    # if id2 (l + 1, m) is in bound, then add
                for k = 1: pw.Nr
                    integral_buffer[id] += conj(crt_shwave[id][k]) * crt_shwave[id2][k] * dU_data[k] * c2 * pw.delta_r
                end
            end
        end
        for id in iter_strategy
            result += integral_buffer[id]
        end
        println("result of k1=$k1, k2=$k2 : $result")
    end
end