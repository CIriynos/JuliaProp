
# ---------------------------
#   fd_elli functions below
# ---------------------------
get_B(η, tilde_flag) = (tilde_flag == true) ? ([-exp(-im * η) 1; exp(-im * η) 1] .* (1 / sqrt(2))) : ([-exp(im * η) 1; exp(im * η) 1] .* (1 / sqrt(2)))

a_expr(l, m) = sqrt((l + m) / ((2l + 1) * (2l - 1))) * (-m * sqrt(l + m - 1) - sqrt((l - m) * (l * (l - 1) - m * (m - 1))))
ã_expr(l, m) = sqrt((l - m) / ((2l + 1) * (2l - 1))) * (-m * sqrt(l - m - 1) + sqrt((l + m) * (l * (l - 1) - m * (m + 1))))

b_expr(l, m) = sqrt((l - m + 1) / ((2l + 1) * (2l + 3))) * (m * sqrt(l - m + 2) - sqrt((l + m + 1) * ((l + 1) * (l + 2) - m * (m - 1))))
b̃_expr(l, m) = sqrt((l + m + 1) / ((2l + 1) * (2l + 3))) * (m * sqrt(l + m + 2) + sqrt((l - m + 1) * ((l + 1) * (l + 2) - m * (m + 1))))

d_expr(l, m) = sqrt((l - m + 1) * (l - m + 2) / ((2l + 1) * (2l + 3)))
d̃_expr(l, m) = d_expr(l, -m)


@inline function get_R_lm(l, m, r, delta_t, At_abs, η, tilde_flag)
    if tilde_flag == true
        ξ = delta_t * At_abs / (8.0 * r)
        b̃lm = b̃_expr(l, m)
        tmp = (1 + ξ^2 * b̃lm^2)
        a = (1 - ξ^2 * b̃lm^2) / tmp
        b = (2 * ξ * exp(im * η) * b̃lm) / tmp
        c = (-2 * ξ * exp(-im * η) * b̃lm) / tmp
    else
        ξ = delta_t * At_abs / (8.0 * r)
        blm = b_expr(l, m)
        tmp = (1 + ξ^2 * blm^2)
        a = (1 - ξ^2 * blm^2) / tmp
        b = (2 * ξ * exp(-im * η) * blm) / tmp
        c = (-2 * ξ * exp(im * η) * blm) / tmp
    end
    return a, b, c
end
# a, b, c = get_R_lm(1, -1, 0.1, 0.01, 1, 0.0, false)
# adjoint([a b; c a]) ≈ inv([a b; c a])

function apply_Rlm_elli(vec1, vec2, rgrid, delta_t, At_abs, η, l, m, tilde_flag)
    a::Float64 = 0
    b::ComplexF64 = 0
    c::ComplexF64 = 0
    r1tmp::ComplexF64 = 0
    r2tmp::ComplexF64 = 0
    @inbounds for i = 1: rgrid.count
        rs = grid_index(rgrid, i)
        a, b, c = get_R_lm(l, m, rs, delta_t, At_abs, η, tilde_flag)
        r1tmp = a * vec1[i] + b * vec2[i]
        r2tmp = c * vec1[i] + a * vec2[i]
        vec1[i] = r1tmp
        vec2[i] = r2tmp
    end
end

function apply_pure_lmat(lmat, input1, input2, output1, output2)
    @. output1 = input1 * lmat[1, 1] + input2 * lmat[1, 2]
    @. output2 = input1 * lmat[2, 1] + input2 * lmat[2, 2]
end

function apply_Ylm_block_elli(rt::tdse_sh_rt, vec1, vec2, tmpvec1, tmpvec2, id, tilde_flag)
    if tilde_flag == true
        apply_pure_lmat(rt.B_tilde_elli, vec1, vec2, tmpvec1, tmpvec2)   # B̃
        # inv(Ỹ+lm) Ỹ_lm
        mul!(vec1, rt.Ylm_tilde_neg[id], tmpvec1)
        trimat_elimination(tmpvec1, rt.Ylm_tilde_pos[id], vec1, rt.A_add_list_scalar[id], rt.B_add_list[id])
        mul!(vec2, rt.Ylm_tilde_pos[id], tmpvec2)
        trimat_elimination(tmpvec2, rt.Ylm_tilde_neg[id], vec2, rt.A_add_list_scalar[id], rt.B_add_list[id])
        
        apply_pure_lmat(adjoint(rt.B_tilde_elli), tmpvec1, tmpvec2, vec1, vec2)  # B̃†
    else
        apply_pure_lmat(rt.B_elli, vec1, vec2, tmpvec1, tmpvec2)  # B
        # inv(Y+lm) Y_lm
        mul!(vec1, rt.Ylm_neg[id], tmpvec1)
        trimat_elimination(tmpvec1, rt.Ylm_pos[id], vec1, rt.A_add_list_scalar[id], rt.B_add_list[id])
        mul!(vec2, rt.Ylm_pos[id], tmpvec2)
        trimat_elimination(tmpvec2, rt.Ylm_neg[id], vec2, rt.A_add_list_scalar[id], rt.B_add_list[id])
        
        apply_pure_lmat(adjoint(rt.B_elli), tmpvec1, tmpvec2, vec1, vec2)  # B†
    end    
end


function fdsh_elli_one_step(crt_shwave, rt::tdse_sh_rt, shgrid, delta_t, At_abs, η, iter_strategy)

    l_num2 = shgrid.l_num * shgrid.l_num

    # update Ylm_pos/neg(tilde) and B, B_tilde
    Threads.@threads for id in iter_strategy
        l = Float64(rt.lmap[id])
        m = Float64(rt.mmap[id])
        tmp = -(delta_t * At_abs / 8) * d_expr(l, m)
        tmp_tilde = (delta_t * At_abs / 8) *  d̃_expr(l, m)
        @. rt.Ylm_pos[id] = rt.M1 + tmp * rt.D1
        @. rt.Ylm_neg[id] = rt.M1 - tmp * rt.D1
        @. rt.Ylm_tilde_pos[id] = rt.M1 + tmp_tilde * rt.D1
        @. rt.Ylm_tilde_neg[id] = rt.M1 - tmp_tilde * rt.D1
    end
    rt.B_elli = get_B(η, false)
    rt.B_tilde_elli = get_B(η, true)

    # apply ang mix
    for id in iter_strategy
        l = rt.lmap[id]
        m = rt.mmap[id]
        next_id = get_index_from_lm(l + 1, m - 1, shgrid.l_num)
        apply_Rlm_elli(crt_shwave[id], crt_shwave[next_id], shgrid.rgrid, delta_t, At_abs, η, l, m, false)
        apply_Ylm_block_elli(rt, crt_shwave[id], crt_shwave[next_id], rt.tmp_shwave[id], rt.tmp_shwave[next_id], id, false)
        next_id = get_index_from_lm(l + 1, m + 1, shgrid.l_num)
        apply_Rlm_elli(crt_shwave[id], crt_shwave[next_id], shgrid.rgrid, delta_t, At_abs, η, l, m, true)
        apply_Ylm_block_elli(rt, crt_shwave[id], crt_shwave[next_id], rt.tmp_shwave[id], rt.tmp_shwave[next_id], id, true)
    end

    # apply inv(W+)W-
    Threads.@threads for id = 1: l_num2
        l = rt.lmap[id]
        m = rt.mmap[id]
        if l == abs(m)
            mul!(rt.tmp_shwave[id], rt.W_neg_boost[l + 1], crt_shwave[id])
            trimat_elimination(crt_shwave[id], rt.W_pos_boost[l+1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
        else
            mul!(rt.tmp_shwave[id], rt.W_neg[l + 1], crt_shwave[id])
            trimat_elimination(crt_shwave[id], rt.W_pos[l+1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
        end
    end
    
    # apply ang mix
    for id in reverse(iter_strategy)
        l = rt.lmap[id]
        m = rt.mmap[id]
        next_id = get_index_from_lm(l + 1, m + 1, shgrid.l_num)
        apply_Ylm_block_elli(rt, crt_shwave[id], crt_shwave[next_id], rt.tmp_shwave[id], rt.tmp_shwave[next_id], id, true)
        apply_Rlm_elli(crt_shwave[id], crt_shwave[next_id], shgrid.rgrid, delta_t, At_abs, η, Float64(l), Float64(m), true)
        next_id = get_index_from_lm(l + 1, m - 1, shgrid.l_num)
        apply_Ylm_block_elli(rt, crt_shwave[id], crt_shwave[next_id], rt.tmp_shwave[id], rt.tmp_shwave[next_id], id, false)
        apply_Rlm_elli(crt_shwave[id], crt_shwave[next_id], shgrid.rgrid, delta_t, At_abs, η, Float64(l), Float64(m), false)
    end
end


function fdsh_elli_one_step_parallelized(crt_shwave, rt::tdse_sh_rt, shgrid, delta_t, At_abs, η, par_strategy)

    l_num2 = shgrid.l_num * shgrid.l_num
    total_line = [par_strategy[1]; par_strategy[2]; par_strategy[3]]

    # update Ylm_pos/neg(tilde) and B, B_tilde
    Threads.@threads for id in total_line
        l = Float64(rt.lmap[id])
        m = Float64(rt.mmap[id])
        tmp = -(delta_t * At_abs / 8) * d_expr(l, m)
        tmp_tilde = (delta_t * At_abs / 8) *  d̃_expr(l, m)
        @. rt.Ylm_pos[id] = rt.M1 + tmp * rt.D1
        @. rt.Ylm_neg[id] = rt.M1 - tmp * rt.D1
        @. rt.Ylm_tilde_pos[id] = rt.M1 + tmp_tilde * rt.D1
        @. rt.Ylm_tilde_neg[id] = rt.M1 - tmp_tilde * rt.D1
    end
    rt.B_elli = get_B(η, false)
    rt.B_tilde_elli = get_B(η, true)

    # apply ang mix
    for line in par_strategy
        Threads.@threads for id in line
            l = rt.lmap[id]
            m = rt.mmap[id]
            next_id = get_index_from_lm(l + 1, m - 1, shgrid.l_num)
            apply_Rlm_elli(crt_shwave[id], crt_shwave[next_id], shgrid.rgrid, delta_t, At_abs, η, l, m, false)
            apply_Ylm_block_elli(rt, crt_shwave[id], crt_shwave[next_id], rt.tmp_shwave[id], rt.tmp_shwave[next_id], id, false)
            next_id = get_index_from_lm(l + 1, m + 1, shgrid.l_num)
            apply_Rlm_elli(crt_shwave[id], crt_shwave[next_id], shgrid.rgrid, delta_t, At_abs, η, l, m, true)
            apply_Ylm_block_elli(rt, crt_shwave[id], crt_shwave[next_id], rt.tmp_shwave[id], rt.tmp_shwave[next_id], id, true)
        end
    end

    # apply inv(W+)W-
    Threads.@threads for id in total_line
        l = rt.lmap[id]
        m = rt.mmap[id]
        if l == abs(m)
            mul!(rt.tmp_shwave[id], rt.W_neg_boost[l + 1], crt_shwave[id])
            trimat_elimination(crt_shwave[id], rt.W_pos_boost[l+1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
        else
            mul!(rt.tmp_shwave[id], rt.W_neg[l + 1], crt_shwave[id])
            trimat_elimination(crt_shwave[id], rt.W_pos[l+1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
        end
    end
    
    # apply ang mix
    for line in reverse(par_strategy)
        Threads.@threads for id in line
            l = rt.lmap[id]
            m = rt.mmap[id]
            next_id = get_index_from_lm(l + 1, m + 1, shgrid.l_num)
            apply_Ylm_block_elli(rt, crt_shwave[id], crt_shwave[next_id], rt.tmp_shwave[id], rt.tmp_shwave[next_id], id, true)
            apply_Rlm_elli(crt_shwave[id], crt_shwave[next_id], shgrid.rgrid, delta_t, At_abs, η, Float64(l), Float64(m), true)
            next_id = get_index_from_lm(l + 1, m - 1, shgrid.l_num)
            apply_Ylm_block_elli(rt, crt_shwave[id], crt_shwave[next_id], rt.tmp_shwave[id], rt.tmp_shwave[next_id], id, false)
            apply_Rlm_elli(crt_shwave[id], crt_shwave[next_id], shgrid.rgrid, delta_t, At_abs, η, Float64(l), Float64(m), false)
        end
    end
end


function tdse_elli_sh_mainloop_record(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, At_data_abs, At_data_eta, steps, Ri_tsurf)
    shgrid = pw.shgrid
    delta_t = pw.delta_t
    phi_record = [zeros(ComplexF64, steps) for i = 1: shgrid.l_num ^ 2]
    dphi_record = [zeros(ComplexF64, steps) for i = 1: shgrid.l_num ^ 2]
    R_id = grid_reduce(shgrid.rgrid, Ri_tsurf)
    println("[TDSE] Start TDSE-SH for elliptical polarized laser. It may cost a plenty of time.")
    for i in 1: steps
        if (i - 1) % 200 == 0
            en = get_energy_sh(crt_shwave, rt, shgrid)
            println("[TDSE] Running TDSE-SH-elliptical. step $(i-1), energy = $en")
        end
        fdsh_elli_one_step_parallelized(crt_shwave, rt, shgrid, delta_t, At_data_abs[i], At_data_eta[i], rt.par_strategy)

        # record
        for id = 1: shgrid.l_num ^ 2
            phi_record[id][i] = crt_shwave[id][R_id]
            dphi_record[id][i] = four_order_difference(crt_shwave[id], R_id, shgrid.rgrid.delta)
        end
    end
    return phi_record, dphi_record
end


function tdse_elli_sh_mainloop_record_xy(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, Ax_data, Ay_data, steps, Ri_tsurf)
    shgrid = pw.shgrid
    delta_t = pw.delta_t
    At_data_abs = norm.(Ax_data .+ im .* Ay_data)
    At_data_eta = angle.(Ax_data .+ im .* Ay_data)
    phi_record = [zeros(ComplexF64, steps) for i = 1: shgrid.l_num ^ 2]
    dphi_record = [zeros(ComplexF64, steps) for i = 1: shgrid.l_num ^ 2]
    R_id = grid_reduce(shgrid.rgrid, Ri_tsurf)

    println("[TDSE] Start TDSE-SH for elliptical polarized laser. It may cost a plenty of time.")
    for i in 1: steps
        if (i - 1) % 200 == 0
            en = get_energy_sh(crt_shwave, rt, shgrid)
            println("[TDSE] Running TDSE-SH-elliptical. step $(i-1), energy = $en")
        end
        fdsh_elli_one_step_parallelized(crt_shwave, rt, shgrid, delta_t, At_data_abs[i], At_data_eta[i], rt.par_strategy)

        # record
        for id = 1: shgrid.l_num ^ 2
            phi_record[id][i] = crt_shwave[id][R_id]
            dphi_record[id][i] = four_order_difference(crt_shwave[id], R_id, shgrid.rgrid.delta)
        end
    end
    return phi_record, dphi_record
end



function tdse_elli_sh_mainloop_record_xy_optimized(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, Ax_data, Ay_data, steps, Ri_tsurf)
    shgrid = pw.shgrid
    delta_t = pw.delta_t
    At_data_abs = norm.(Ax_data .+ im .* Ay_data)
    At_data_eta = angle.(Ax_data .+ im .* Ay_data)
    phi_record = [zeros(ComplexF64, steps) for i = 1: shgrid.l_num ^ 2]
    dphi_record = [zeros(ComplexF64, steps) for i = 1: shgrid.l_num ^ 2]
    R_id = grid_reduce(shgrid.rgrid, Ri_tsurf)

    # optimization
    TDSE_ELLI_OPTIMIZE_THRESHOLD = 1e-15
    TDSE_ELLI_MIN_LM = 3
    visiting_ids = sort([rt.par_strategy[1]; rt.par_strategy[2]; rt.par_strategy[3]])
    optimized_par_strategy = deepcopy(rt.par_strategy)
    par_lens::Vector{Int64} = [0, 0, 0]
    ids_mask = zeros(Int64, shgrid.l_num ^ 2)   # 1 => selected.
    actual_par_strategy::Vector{Vector{Int64}} = [[0], [0], [0]]
    iter_strategy::Vector{Int64} = [0]

    # Start Mainloop
    println("[TDSE] Start TDSE-SH for elliptical polarized laser. It may cost a plenty of time.")
    for i in 1: steps

        # We do not need to get optimized_count every time, just sometimes.
        if (i - 1) % 10 == 0
            # we only select the valuable ids for calculating.
            ids_mask .= 0   # clear first
            par_lens .= 0

            Threads.@threads for id in visiting_ids
                l = rt.lmap[id]
                m = rt.mmap[id]
                tmp::Float64 = 0.0
                for k = 1: pw.Nr
                    tmp += real(conj(crt_shwave[id][k]) * crt_shwave[id][k])
                end

                if tmp > TDSE_ELLI_OPTIMIZE_THRESHOLD
                    ids_mask[id] = 1    # this id should be count in.
                end
                # if id == rt.par_strategy[1][1] || id == rt.par_strategy[2][1] || id == rt.par_strategy[3][1]
                #     ids_mask[id] = 1    # always count in them.
                # end
                if l < TDSE_ELLI_MIN_LM && abs(m) < TDSE_ELLI_MIN_LM
                    ids_mask[id] = 1    # always count in them.
                end
            end

            # filtering
            for (j, line) in enumerate(rt.par_strategy)
                for id in line
                    if ids_mask[id] == 1
                        par_lens[j] += 1
                        optimized_par_strategy[j][par_lens[j]] = id
                    end
                end
            end

            actual_par_strategy = [optimized_par_strategy[1][1: par_lens[1]], optimized_par_strategy[2][1: par_lens[2]],
                optimized_par_strategy[3][1: par_lens[3]]]
            iter_strategy = [optimized_par_strategy[1][1: par_lens[1]]; optimized_par_strategy[2][1: par_lens[2]];
                optimized_par_strategy[3][1: par_lens[3]]]
        end

        fdsh_elli_one_step_parallelized(crt_shwave, rt, shgrid, delta_t, At_data_abs[i], At_data_eta[i], actual_par_strategy)

        # record RI
        for id = 1: shgrid.l_num ^ 2
            phi_record[id][i] = crt_shwave[id][R_id]
            dphi_record[id][i] = four_order_difference(crt_shwave[id], R_id, shgrid.rgrid.delta)
        end

        # Printing Logging message
        if (i - 1) % 200 == 0
            en = get_energy_sh(crt_shwave, rt, shgrid)
            println("[TDSE] Running TDSE-SH-elliptical. step $(i-1), energy = $en")
            println("  |->  par_lens = $(par_lens[1]), $(par_lens[2]), $(par_lens[3])")
        end
    end
    
    return phi_record, dphi_record
end



function tdse_elli_sh_mainloop_record_xy_hhg_optimized(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, Ax_data, Ay_data, steps, Ri_tsurf; start_rcd_step::Int64 = 1, end_rcd_step::Int64 = 1000000)
    shgrid = pw.shgrid
    delta_t = pw.delta_t
    At_data_abs = norm.(Ax_data .+ im .* Ay_data)
    At_data_eta = angle.(Ax_data .+ im .* Ay_data)
    phi_record = [zeros(ComplexF64, steps) for i = 1: shgrid.l_num ^ 2]
    dphi_record = [zeros(ComplexF64, steps) for i = 1: shgrid.l_num ^ 2]
    R_id = grid_reduce(shgrid.rgrid, Ri_tsurf)

    # hhg
    dU_data = get_derivative_two_order(pw.po_data_r, pw.delta_r)
    hhg_integral_t_1 = zeros(ComplexF64, steps)
    hhg_integral_t_2 = zeros(ComplexF64, steps)
    hhg_integral_t_3 = zeros(ComplexF64, steps)
    integral_buffer_1 = zeros(ComplexF64, pw.l_num ^ 2)     # <lm|N+|l'm'>
    integral_buffer_2 = zeros(ComplexF64, pw.l_num ^ 2)     # <lm|N-|l'm'>
    integral_buffer_3 = zeros(ComplexF64, pw.l_num ^ 2)     # <lm|cosθ|l'm'>

    # optimization
    TDSE_ELLI_OPTIMIZE_THRESHOLD = 1e-15
    TDSE_ELLI_MIN_LM = 3
    visiting_ids = sort([rt.par_strategy[1]; rt.par_strategy[2]; rt.par_strategy[3]])
    optimized_par_strategy = deepcopy(rt.par_strategy)
    par_lens::Vector{Int64} = [0, 0, 0]
    ids_mask = zeros(Int64, shgrid.l_num ^ 2)   # 1 => selected.
    actual_par_strategy::Vector{Vector{Int64}} = [[0], [0], [0]]
    iter_strategy::Vector{Int64} = [0]

    # Start Mainloop
    println("[TDSE] Start TDSE-SH for elliptical polarized laser. It may cost a plenty of time.")
    for i in 1: steps

        # We do not need to get optimized_count every time, just sometimes.
        if (i - 1) % 20 == 0
            # we only select the valuable ids for calculating.
            ids_mask .= 0   # clear first
            par_lens .= 0

            Threads.@threads for id in visiting_ids
                l = rt.lmap[id]
                m = rt.mmap[id]
                tmp::Float64 = 0.0
                for k = 1: pw.Nr
                    tmp += real(conj(crt_shwave[id][k]) * crt_shwave[id][k])
                end

                if tmp > TDSE_ELLI_OPTIMIZE_THRESHOLD
                    ids_mask[id] = 1    # this id should be count in.
                end
                # if id == rt.par_strategy[1][1] || id == rt.par_strategy[2][1] || id == rt.par_strategy[3][1]
                #     ids_mask[id] = 1    # always count in them.
                # end
                if l < TDSE_ELLI_MIN_LM && abs(m) < TDSE_ELLI_MIN_LM
                    ids_mask[id] = 1    # always count in them.
                end
            end

            # filtering
            for (j, line) in enumerate(rt.par_strategy)
                for id in line
                    if ids_mask[id] == 1
                        par_lens[j] += 1
                        optimized_par_strategy[j][par_lens[j]] = id
                    end
                end
            end

            actual_par_strategy = [optimized_par_strategy[1][1: par_lens[1]], optimized_par_strategy[2][1: par_lens[2]],
                optimized_par_strategy[3][1: par_lens[3]]]
            iter_strategy = [optimized_par_strategy[1][1: par_lens[1]]; optimized_par_strategy[2][1: par_lens[2]];
                optimized_par_strategy[3][1: par_lens[3]]]
        end

        fdsh_elli_one_step_parallelized(crt_shwave, rt, shgrid, delta_t, At_data_abs[i], At_data_eta[i], actual_par_strategy)

        # record RI
        for id = 1: shgrid.l_num ^ 2
            phi_record[id][i] = crt_shwave[id][R_id]
            dphi_record[id][i] = four_order_difference(crt_shwave[id], R_id, shgrid.rgrid.delta)
        end

        # Printing Logging message
        if (i - 1) % 200 == 0
            en = get_energy_sh(crt_shwave, rt, shgrid)
            println("[TDSE] Running TDSE-SH-elliptical. step $(i-1), energy = $en")
            println("  |->  par_lens = $(par_lens[1]), $(par_lens[2]), $(par_lens[3])")
        end

        if i < start_rcd_step || i > end_rcd_step
            continue
        end

        # HHG Part
        Threads.@threads for id in iter_strategy
            l = rt.lmap[id]
            m = rt.mmap[id]
            id1 = get_index_from_lm(l - 1, m - 1, pw.shgrid.l_num)
            id2 = get_index_from_lm(l + 1, m - 1, pw.shgrid.l_num)
            id3 = get_index_from_lm(l - 1, m + 1, pw.shgrid.l_num)
            id4 = get_index_from_lm(l + 1, m + 1, pw.shgrid.l_num)
            c1 = -sqrt((l + m - 1) * (l + m) / ((2 * l - 1) * (2 * l + 1)))
            c2 = sqrt((l - m + 2) * (l - m + 1) / ((2 * l + 1) * (2 * l + 3)))
            c3 = sqrt((l - m - 1) * (l - m) / ((2 * l - 1) * (2 * l + 1)))
            c4 = -sqrt((l + m + 2) * (l + m + 1) / ((2 * l + 1) * (2 * l + 3)))

            integral_buffer_1[id] = 0      # clear it first
            integral_buffer_2[id] = 0      # clear it first
            if id1 != -1    # if id1 (l - 1, m - 1) is in bound, then add 
                for k = 1: pw.Nr
                    integral_buffer_1[id] += conj(crt_shwave[id][k]) * crt_shwave[id1][k] * dU_data[k] * c1 * pw.delta_r
                end
            end
            if id2 != -1    # if id2 (l + 1, m - 1) is in bound, then add
                for k = 1: pw.Nr
                    integral_buffer_1[id] += conj(crt_shwave[id][k]) * crt_shwave[id2][k] * dU_data[k] * c2 * pw.delta_r
                end
            end
            if id3 != -1    # if id3 (l - 1, m + 1) is in bound, then add
                for k = 1: pw.Nr
                    integral_buffer_2[id] += conj(crt_shwave[id][k]) * crt_shwave[id3][k] * dU_data[k] * c3 * pw.delta_r
                end
            end
            if id4 != -1    # if id4 (l + 1, m + 1) is in bound, then add
                for k = 1: pw.Nr
                    integral_buffer_2[id] += conj(crt_shwave[id][k]) * crt_shwave[id4][k] * dU_data[k] * c4 * pw.delta_r
                end
            end

            # z-direction
            id5 = get_index_from_lm(l - 1, m, pw.shgrid.l_num)
            id6 = get_index_from_lm(l + 1, m, pw.shgrid.l_num)

            integral_buffer_3[id] = 0      # clear it first
            if id5 != -1    # if id5 (l - 1, m) is in bound, then add
                for k = 1: pw.Nr
                    integral_buffer_3[id] += conj(crt_shwave[id][k]) * crt_shwave[id5][k] * dU_data[k] * c_expr(l - 1, m) * pw.delta_r
                end
            end
            if id6 != -1    # if id6 (l + 1, m) is in bound, then add
                for k = 1: pw.Nr
                    integral_buffer_3[id] += conj(crt_shwave[id][k]) * crt_shwave[id6][k] * dU_data[k] * c_expr(l, m) * pw.delta_r
                end
            end
        end
        for id in iter_strategy
            hhg_integral_t_1[i] += integral_buffer_1[id]
            hhg_integral_t_2[i] += integral_buffer_2[id]
            hhg_integral_t_3[i] += integral_buffer_3[id]
        end
    end
    
    return hhg_integral_t_1, hhg_integral_t_2, hhg_integral_t_3, phi_record, dphi_record
end






function fdsh_elli_one_step_parallelized_no_coupling(crt_shwave, rt::tdse_sh_rt, shgrid, delta_t, At_abs, η, par_strategy)

    total_line = [par_strategy[1]; par_strategy[2]; par_strategy[3]]
    # No need to apply ang mix anymore
    # ...
    # apply inv(W+)W-
    Threads.@threads for id in total_line
        l = rt.lmap[id]
        m = rt.mmap[id]
        if l == abs(m)
            mul!(rt.tmp_shwave[id], rt.W_neg_boost[l + 1], crt_shwave[id])
            trimat_elimination(crt_shwave[id], rt.W_pos_boost[l+1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
        else
            mul!(rt.tmp_shwave[id], rt.W_neg[l + 1], crt_shwave[id])
            trimat_elimination(crt_shwave[id], rt.W_pos[l+1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
        end
    end
    
    # No need to apply ang mix anymore
    # ...
end


# Newest Version ==> long_prop version
function tdse_elli_sh_mainloop_record_xy_hhg_long_prop(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, Ax_data, Ay_data, steps, steps_laser, Ri_tsurf; start_rcd_step::Int64 = 1, end_rcd_step::Int64 = 100000000)
    shgrid = pw.shgrid
    delta_t = pw.delta_t
    At_data_abs = norm.(Ax_data .+ im .* Ay_data)
    At_data_eta = angle.(Ax_data .+ im .* Ay_data)
    phi_record = [zeros(ComplexF64, steps) for i = 1: shgrid.l_num ^ 2]
    dphi_record = [zeros(ComplexF64, steps) for i = 1: shgrid.l_num ^ 2]
    R_id = grid_reduce(shgrid.rgrid, Ri_tsurf)

    # hhg
    dU_data = get_derivative_two_order(pw.po_data_r, pw.delta_r)
    hhg_integral_t = zeros(ComplexF64, steps)
    integral_buffer = zeros(ComplexF64, pw.l_num ^ 2)

    # optimization
    TDSE_ELLI_OPTIMIZE_THRESHOLD = 1e-15
    visiting_ids = sort([rt.par_strategy[1]; rt.par_strategy[2]; rt.par_strategy[3]])
    optimized_par_strategy = deepcopy(rt.par_strategy)
    par_lens::Vector{Int64} = [0, 0, 0]
    ids_mask = zeros(Int64, shgrid.l_num ^ 2)   # 1 => selected.
    actual_par_strategy::Vector{Vector{Int64}} = [[0], [0], [0]]
    iter_strategy::Vector{Int64} = [0]

    # Start Mainloop
    println("[TDSE] Start TDSE-SH for elliptical polarized laser. It may cost a plenty of time.")
    for i in 1: steps

        # We do not need to get optimized_count every time, just sometimes.
        if (i - 1) % 20 == 0
            # we only select the valuable ids for calculating.
            ids_mask .= 0   # clear first
            par_lens .= 0

            Threads.@threads for id in visiting_ids
                tmp::Float64 = 0.0
                for k = 1: pw.Nr
                    tmp += real(conj(crt_shwave[id][k]) * crt_shwave[id][k])
                end

                if tmp > TDSE_ELLI_OPTIMIZE_THRESHOLD
                    ids_mask[id] = 1    # this id should be count in.
                end
                if id == rt.par_strategy[1][1] || id == rt.par_strategy[2][1] || id == rt.par_strategy[3][1]
                    ids_mask[id] = 1    # always count in them.
                end
            end

            # filtering
            for (j, line) in enumerate(rt.par_strategy)
                for id in line
                    if ids_mask[id] == 1
                        par_lens[j] += 1
                        optimized_par_strategy[j][par_lens[j]] = id
                    end
                end
            end

            # update actual_par_strategy
            actual_par_strategy = [optimized_par_strategy[1][1: par_lens[1]], optimized_par_strategy[2][1: par_lens[2]],
                optimized_par_strategy[3][1: par_lens[3]]]
            iter_strategy = [optimized_par_strategy[1][1: par_lens[1]]; optimized_par_strategy[2][1: par_lens[2]];
                optimized_par_strategy[3][1: par_lens[3]]]
        end

        # if laser pulse is over, then we can propagate without Ylm coupling.
        if i > steps_laser
            fdsh_elli_one_step_parallelized_no_coupling(crt_shwave, rt, shgrid, delta_t, At_data_abs[i], At_data_eta[i], actual_par_strategy)
        else
            fdsh_elli_one_step_parallelized(crt_shwave, rt, shgrid, delta_t, At_data_abs[i], At_data_eta[i], actual_par_strategy)
        end

        # # record RI
        # for id = 1: shgrid.l_num ^ 2
        #     phi_record[id][i] = crt_shwave[id][R_id]
        #     dphi_record[id][i] = four_order_difference(crt_shwave[id], R_id, shgrid.rgrid.delta)
        # end

        # Printing Logging message
        if (i - 1) % 200 == 0
            en = get_energy_sh(crt_shwave, rt, shgrid)
            println("[TDSE] Running TDSE-SH-elliptical. step $(i-1), energy = $en")
            println("  |->  par_lens = $(par_lens[1]), $(par_lens[2]), $(par_lens[3])")
        end

        if i < start_rcd_step || i > end_rcd_step
            continue
        end

        # HHG Part
        Threads.@threads for id in iter_strategy
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
            hhg_integral_t[i] += integral_buffer[id]
        end
    end
    return hhg_integral_t, phi_record, dphi_record
end