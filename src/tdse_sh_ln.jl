# ---------------------------
#    fd_sh functions below
# ---------------------------
c_expr(l, m) = sqrt(((l + 1) ^ 2 - m ^ 2) / ((2 * l + 1) * (2 * l + 3)))

function calc_ground_state_population(crt_shwave::shwave_t, init_shwave::shwave_t, shgrid::GridSH)
    # P1s(t) = |Σ ϕ*00(rs,0)ϕ00(rs,t)∆r|^2, formula in book
    return (dot(init_shwave[1], crt_shwave[1])) ^ 2
end

function apply_ang_v2(piece1, piece2, rgrid, delta_t, At, l, m)
    gamma = 0.25 * At * delta_t * c_expr(l, m) * (l + 1)
    @inbounds @fastmath for i = 1: rgrid.count
        rs = grid_index(rgrid, i)
        a = (rs * rs - gamma * gamma) / (rs * rs + gamma * gamma)
        b = (2 * rs * gamma) / (rs * rs + gamma * gamma)
        r1tmp = a * piece1[i] - b * piece2[i]
        r2tmp = b * piece1[i] + a * piece2[i]
        piece1[i] = r1tmp
        piece2[i] = r2tmp
    end
end

function apply_pure_lmat_v2(lmat, input1, input2, output1, output2, adjoint_flag)
    if adjoint_flag == true
        @. output1 = input1 * conj(lmat[1, 1]) + input2 * conj(lmat[2, 1])
        @. output2 = input1 * conj(lmat[1, 2]) + input2 * conj(lmat[2, 2])
    else
        @. output1 = input1 * lmat[1, 1] + input2 * lmat[1, 2]
        @. output2 = input1 * lmat[2, 1] + input2 * lmat[2, 2]
    end
end

function apply_mix_v2(rt::tdse_sh_rt, crt_piece_1, crt_piece_2, tmp_piece_1, tmp_piece_2, id)
    apply_pure_lmat_v2(rt.B_pl, crt_piece_1, crt_piece_2, tmp_piece_1, tmp_piece_2, false)

    mul!(crt_piece_1, rt.Ylm_neg[id], tmp_piece_1)
    trimat_elimination(tmp_piece_1, rt.Ylm_pos[id], crt_piece_1, rt.A_add_list_scalar[id], rt.B_add_list[id])
    mul!(crt_piece_2, rt.Ylm_pos[id], tmp_piece_2)
    trimat_elimination(tmp_piece_2, rt.Ylm_neg[id], crt_piece_2, rt.A_add_list_scalar[id], rt.B_add_list[id])

    apply_pure_lmat_v2(rt.B_pl, tmp_piece_1, tmp_piece_2, crt_piece_1, crt_piece_2, true)
end

function fdshpl_one_step(crt_shwave, rt, shgrid, delta_t, At, At_next, m; optimizing_count::Int64 = 0)
    bundle_size = shgrid.l_num - abs(m) - optimizing_count
    head_ptr = get_index_from_lm(abs(m), m, shgrid.l_num)

    # update rmats
    Threads.@threads for j = 1: bundle_size - 1
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        tmp = 0.25 * delta_t * At * c_expr(l, m)
        @. rt.Ylm_pos[id] = rt.M1 + tmp * rt.D1
        @. rt.Ylm_neg[id] = rt.M1 - tmp * rt.D1
    end

    # apply ang mix, using new approach
    Threads.@threads for j = 1: 2: bundle_size - 1
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        apply_ang_v2(crt_shwave[id], crt_shwave[id + 1], shgrid.rgrid, delta_t, At, l, m)
        apply_mix_v2(rt, crt_shwave[id], crt_shwave[id + 1], rt.tmp_shwave[id], rt.tmp_shwave[id + 1], id)
    end
    Threads.@threads for j = 2: 2: bundle_size - 1
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        apply_ang_v2(crt_shwave[id], crt_shwave[id + 1], shgrid.rgrid, delta_t, At, l, m)
        apply_mix_v2(rt, crt_shwave[id], crt_shwave[id + 1], rt.tmp_shwave[id], rt.tmp_shwave[id + 1], id)
    end

    # for j = 1: 1: bundle_size - 1
    #     l = abs(m) + j - 1
    #     id = head_ptr + j - 1
    #     apply_ang_v2(crt_shwave[id], crt_shwave[id + 1], shgrid.rgrid, delta_t, At, l, m)
    #     apply_mix_v2(rt, crt_shwave[id], crt_shwave[id + 1], rt.tmp_shwave[id], rt.tmp_shwave[id + 1], id)
    # end

    # apply Hat
    Threads.@threads for j = 1: bundle_size
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        if j == 1
            mul!(rt.tmp_shwave[id], rt.W_neg_boost[l+1], crt_shwave[id])
            trimat_elimination(crt_shwave[id], rt.W_pos_boost[l+1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
        else
            mul!(rt.tmp_shwave[id], rt.W_neg[l+1], crt_shwave[id])
            trimat_elimination(crt_shwave[id], rt.W_pos[l+1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
        end
    end

    # update rmats
    Threads.@threads for j = 1: bundle_size - 1
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        tmp = 0.25 * delta_t * At_next * c_expr(l, m)
        @. rt.Ylm_pos[id] = rt.M1 + tmp * rt.D1
        @. rt.Ylm_neg[id] = rt.M1 - tmp * rt.D1
    end

    # apply mix ang
    Threads.@threads for j = 2: 2: bundle_size - 1
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        apply_mix_v2(rt, crt_shwave[id], crt_shwave[id + 1], rt.tmp_shwave[id], rt.tmp_shwave[id + 1], id)
        apply_ang_v2(crt_shwave[id], crt_shwave[id + 1], shgrid.rgrid, delta_t, At_next, l, m)
    end
    Threads.@threads for j = 1: 2: bundle_size - 1
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        apply_mix_v2(rt, crt_shwave[id], crt_shwave[id + 1], rt.tmp_shwave[id], rt.tmp_shwave[id + 1], id)
        apply_ang_v2(crt_shwave[id], crt_shwave[id + 1], shgrid.rgrid, delta_t, At_next, l, m)
    end

    # for j = bundle_size - 1: -1: 1
    #     l = abs(m) + j - 1
    #     id = head_ptr + j - 1
    #     apply_ang_v2(crt_shwave[id], crt_shwave[id + 1], shgrid.rgrid, delta_t, At, l, m)
    #     apply_mix_v2(rt, crt_shwave[id], crt_shwave[id + 1], rt.tmp_shwave[id], rt.tmp_shwave[id + 1], id)
    # end
end


function apply_R_length(piece1, piece2, rgrid, delta_t, Et, l, m)
    gamma = 1im * 0.25 * delta_t * Et * c_expr(l, m)
    @inbounds @fastmath for i = 1: rgrid.count
        rs = grid_index(rgrid, i)
        a = (1.0 + gamma^2 * rs^2) / (1.0 - gamma^2 * rs^2)
        b = (-2.0 * gamma * rs) / (1.0 - gamma^2 * rs^2)
        r1tmp = a * piece1[i] + b * piece2[i]
        r2tmp = b * piece1[i] + a * piece2[i]
        piece1[i] = r1tmp
        piece2[i] = r2tmp
    end
end

function fdshpl_length_gauge_one_step(crt_shwave, rt::tdse_sh_rt, shgrid, delta_t, Et, m; optimizing_count::Int64 = 0)
    bundle_size = shgrid.l_num - abs(m) - optimizing_count
    head_ptr = get_index_from_lm(abs(m), m, shgrid.l_num)

    # apply ang mix, using new approach
    Threads.@threads for j = 1: 2: bundle_size - 1
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        apply_R_length(crt_shwave[id], crt_shwave[id + 1], shgrid.rgrid, delta_t, Et, l, m)
    end
    Threads.@threads for j = 2: 2: bundle_size - 1
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        apply_R_length(crt_shwave[id], crt_shwave[id + 1], shgrid.rgrid, delta_t, Et, l, m)
    end

    # apply Hat
    Threads.@threads for j = 1: bundle_size
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        if j == 1
            mul!(rt.tmp_shwave[id], rt.W_neg_boost[l+1], crt_shwave[id])
            trimat_elimination(crt_shwave[id], rt.W_pos_boost[l+1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
        else
            mul!(rt.tmp_shwave[id], rt.W_neg[l+1], crt_shwave[id])
            trimat_elimination(crt_shwave[id], rt.W_pos[l+1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
        end
    end

    # apply mix ang
    Threads.@threads for j = 2: 2: bundle_size - 1
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        apply_R_length(crt_shwave[id], crt_shwave[id + 1], shgrid.rgrid, delta_t, Et, l, m)
    end
    Threads.@threads for j = 1: 2: bundle_size - 1
        l = abs(m) + j - 1
        id = head_ptr + j - 1
        apply_R_length(crt_shwave[id], crt_shwave[id + 1], shgrid.rgrid, delta_t, Et, l, m)
    end
end


function tdseln_sh_mainloop(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, At_data, steps::Int64; m::Int64 = 0)
    energy_list = Float64[]
    for i in 1: steps
        if (i - 1) % 200 == 0
            en = get_energy_sh_mbunch(crt_shwave, rt, pw.shgrid, m)
            norm_value = sum(map(rvec->dot(rvec, rvec), crt_shwave))
            push!(energy_list, en)
            println("step $(i-1) energy = $en, norm_value = $norm_value")
        end
        At = At_data[i]
        At_next = At_data[min(i + 1, steps)]
        fdshpl_one_step(crt_shwave, rt, pw.shgrid, pw.delta_t, At, At_next, m)
    end
    return energy_list
end


function tdseln_sh_mainloop_record(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, At_data, steps::Int64, Ri_tsurf::Float64; m::Int64 = 0)
    phi_record = [zeros(ComplexF64, steps) for i = 1: pw.l_num ^ 2]
    dphi_record = [zeros(ComplexF64, steps) for i = 1: pw.l_num ^ 2]
    R_id::Int64 = grid_reduce(pw.shgrid.rgrid, Ri_tsurf)
    for i in 1: steps
        if (i - 1) % 200 == 0
            en = get_energy_sh_mbunch(crt_shwave, rt, pw.shgrid, m)
            println("step $(i-1) energy = $en")
        end
        At = At_data[i]
        At_next = At_data[min(i + 1, steps)]
        fdshpl_one_step(crt_shwave, rt, pw.shgrid, pw.delta_t, At, At_next, m)

        # record
        for id = 1: pw.l_num ^ 2
            phi_record[id][i] = crt_shwave[id][R_id]
            dphi_record[id][i] = four_order_difference(crt_shwave[id], R_id, pw.delta_r)
        end
    end
    return phi_record, dphi_record
end


function tdseln_sh_mainloop_record_optimized(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, At_data, steps::Int64, Ri_tsurf::Float64; m::Int64 = 0)
    phi_record = [zeros(ComplexF64, steps) for i = 1: pw.l_num ^ 2]
    dphi_record = [zeros(ComplexF64, steps) for i = 1: pw.l_num ^ 2]
    R_id::Int64 = grid_reduce(pw.shgrid.rgrid, Ri_tsurf)

    # optimization
    TDSE_LN_OPTIMIZE_THRESHOLD = 1e-15
    head_ptr = get_index_from_lm(abs(m), m, pw.shgrid.l_num)
    bundle_size = pw.shgrid.l_num - abs(m)
    ids_mask = zeros(Int64, pw.shgrid.l_num ^ 2)   # 1 => selected.
    optimized_count::Int64 = 0

    for i in 1: steps
        # we only select the valuable ids
        # We do not need to get optimized_count every time, just sometimes.
        if (i - 1) % 200 == 0
            ids_mask .= 0   # clear first
            optimized_count = 0

            Threads.@threads for j = 1: bundle_size
                id = head_ptr + j - 1
                tmp::Float64 = real(dot(crt_shwave[id], crt_shwave[id]))
                if tmp > TDSE_LN_OPTIMIZE_THRESHOLD
                    ids_mask[id] = 1    # this id should be count in.
                end
                if id == head_ptr || id == head_ptr + 1
                    ids_mask[id] = 1    # always count in head_ptr
                end
            end

            # filtering
            for j = 1: bundle_size
                id = head_ptr + j - 1
                if ids_mask[id] == 0
                    optimized_count += 1
                end
            end

            # make sure that we can update the block at boundary + 1
            optimized_count = max(0, optimized_count - 1)
        end

        At = At_data[i]
        At_next = At_data[min(i + 1, steps)]
        fdshpl_one_step(crt_shwave, rt, pw.shgrid, pw.delta_t, At, At_next, m; optimizing_count=optimized_count)

        # record
        for id = 1: pw.l_num ^ 2
            phi_record[id][i] = crt_shwave[id][R_id]
            dphi_record[id][i] = four_order_difference(crt_shwave[id], R_id, pw.delta_r)
        end

        # Printing Logging message
        if (i - 1) % 200 == 0
            en = get_energy_sh_mbunch(crt_shwave, rt, pw.shgrid, m)
            println("[TDSE] Runing TDSE-SH-linear. step $(i-1) energy = $en ")
            println("     | block_size = $(bundle_size - optimized_count) ")
        end
    end
    return phi_record, dphi_record
end



function tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, At_data, steps::Int64, Ri_tsurf::Float64; m::Int64 = 0, start_rcd_step::Int64 = 1, end_rcd_step::Int64 = 1000000)
    phi_record = [zeros(ComplexF64, steps) for i = 1: pw.l_num ^ 2]
    dphi_record = [zeros(ComplexF64, steps) for i = 1: pw.l_num ^ 2]
    R_id::Int64 = grid_reduce(pw.shgrid.rgrid, Ri_tsurf)

    # hhg
    dU_data = get_derivative_two_order(pw.po_data_r, pw.delta_r)
    hhg_integral_t = zeros(ComplexF64, steps)
    integral_buffer = zeros(ComplexF64, pw.l_num)

    # optimization
    TDSE_LN_OPTIMIZE_THRESHOLD = 1e-15
    head_ptr = get_index_from_lm(abs(m), m, pw.shgrid.l_num)
    bundle_size = pw.shgrid.l_num - abs(m)
    ids_mask = zeros(Int64, pw.shgrid.l_num ^ 2)   # 1 => selected.
    optimized_count::Int64 = 0

    for i in 1: steps
        # we only select the valuable ids
        ids_mask .= 0   # clear first

        # We do not need to get optimized_count every time, just sometimes.
        if (i - 1) % 20 == 0
            optimized_count = 0

            Threads.@threads for j = 1: bundle_size
                id = head_ptr + j - 1
                tmp::Float64 = real(dot(crt_shwave[id], crt_shwave[id]))
                if tmp > TDSE_LN_OPTIMIZE_THRESHOLD
                    ids_mask[id] = 1    # this id should be count in.
                end
                if id == head_ptr || id == head_ptr + 1
                    ids_mask[id] = 1    # always count in head_ptr
                end
            end

            # filtering
            for j = 1: bundle_size
                id = head_ptr + j - 1
                if ids_mask[id] == 0
                    optimized_count += 1
                end
            end

            # make sure that we can update the block at boundary + 1
            optimized_count = max(0, optimized_count - 1)
        end

        At = At_data[i]
        At_next = At_data[min(i + 1, steps)]
        fdshpl_one_step(crt_shwave, rt, pw.shgrid, pw.delta_t, At, At_next, m; optimizing_count=optimized_count)

        # record
        for id = 1: pw.l_num ^ 2
            phi_record[id][i] = crt_shwave[id][R_id]
            dphi_record[id][i] = four_order_difference(crt_shwave[id], R_id, pw.delta_r)
        end

        # Printing Logging message
        if (i - 1) % 200 == 0
            en = get_energy_sh_mbunch(crt_shwave, rt, pw.shgrid, m)
            norm_value = sum(map(rvec->dot(rvec, rvec), crt_shwave))
            println("[TDSE] Runing TDSE-SH-linear. step $(i-1) energy = $en, norm_value = $norm_value")
            println("     | block_size = $(bundle_size - optimized_count) ")
        end

        if i < start_rcd_step || i > end_rcd_step
            continue
        end

        # HHG
        Threads.@threads for j = 1: bundle_size - optimized_count
            l = abs(m) + j - 1
            id = head_ptr + j - 1
            id1 = get_index_from_lm(l - 1, m, pw.shgrid.l_num)
            id2 = get_index_from_lm(l + 1, m, pw.shgrid.l_num)

            integral_buffer[l + 1] = 0      # clear it first
            # if id1 != -1    # if id1 (l - 1, m) is in bound, then add 
            #     @fastmath integral_buffer[l + 1] += dot(crt_shwave[id], crt_shwave[id1] .* dU_data) * pw.delta_r * c_expr(l - 1, m)
            # end
            # if id2 != -1    # if id2 (l + 1, m) is in bound, then add
            #     @fastmath integral_buffer[l + 1] += dot(crt_shwave[id], crt_shwave[id2] .* dU_data) * pw.delta_r * c_expr(l, m)
            # end
            if id1 != -1
                for k = 1: pw.Nr
                    integral_buffer[l + 1] += conj(crt_shwave[id][k]) * crt_shwave[id1][k] * dU_data[k] * pw.delta_r * c_expr(l - 1, m)
                end
            end
            if id2 != -1
                for k = 1: pw.Nr
                    integral_buffer[l + 1] += conj(crt_shwave[id][k]) * crt_shwave[id2][k] * dU_data[k] * pw.delta_r * c_expr(l, m)
                end
            end
        end
        for j = 1: bundle_size - optimized_count
            l = abs(m) + j - 1
            hhg_integral_t[i] += integral_buffer[l + 1]
        end
    end
    return hhg_integral_t, phi_record, dphi_record
end



function tdseln_sh_mainloop_length_gauge(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, Et_data, steps::Int64; m::Int64 = 0)
    init_shwave = copy(crt_shwave)
    norm_list = Float64[]
    gs_population_list = Float64[]
    energy_list = Float64[]
    for i in 1: steps
        if (i - 1) % 200 == 0
            en = get_energy_sh_mbunch(crt_shwave, rt, pw.shgrid, m)
            norm_value = sum(map(rvec->dot(rvec, rvec), crt_shwave))
            gs_ppls_value = calc_ground_state_population(crt_shwave, init_shwave, pw.shgrid)
            push!(norm_list, norm_value)
            push!(gs_population_list, gs_ppls_value)
            push!(energy_list, en)
            println("step $(i-1) energy = $en, gs_population = $gs_ppls_value, norm = $norm_value")
        end

        fdshpl_length_gauge_one_step(crt_shwave, rt, pw.shgrid, pw.delta_t, Et_data[i], m)
    end
    return energy_list
end


function tdseln_sh_mainloop_length_gauge_hhg(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, Et_data, steps::Int64; m::Int64 = 0, start_rcd_step::Int64 = 1, end_rcd_step::Int64 = 100000000)
    
    init_shwave = copy(crt_shwave)
    norm_list = Float64[]
    gs_population_list = Float64[]
    energy_list = Float64[]

    # hhg
    dU_data = get_derivative_two_order(pw.po_data_r, pw.delta_r)
    hhg_integral_t = zeros(ComplexF64, steps)
    integral_buffer = zeros(ComplexF64, pw.l_num)

    # optimization
    TDSE_LN_OPTIMIZE_THRESHOLD = 1e-15
    ids_mask = zeros(Int64, pw.shgrid.l_num ^ 2)   # 1 => selected.
    head_ptr = get_index_from_lm(abs(m), m, pw.shgrid.l_num)
    bundle_size = pw.shgrid.l_num - abs(m)
    optimized_count::Int64 = 0
    tmp::Float64 = 0.0

    for i in 1: steps
        # We only select the valuable ids
        # But we do not need to get optimized_count every time, just sometimes.
        if (i - 1) % 10 == 0
            ids_mask .= 0          # clear all first
            optimized_count = 0

            Threads.@threads for j = 1: bundle_size
                id = head_ptr + j - 1
                # tmp::Float64 = real(dot(crt_shwave[id], crt_shwave[id]))
                tmp = 0
                for k = 1: pw.Nr
                    tmp += real(conj(crt_shwave[id][k]) * crt_shwave[id][k])
                end

                if tmp > TDSE_LN_OPTIMIZE_THRESHOLD
                    ids_mask[id] = 1    # this id should be count in.
                end
                if id == head_ptr || id == head_ptr + 1
                    ids_mask[id] = 1    # always count in head_ptr
                end
            end

            # filtering
            for j = 1: bundle_size
                id = head_ptr + j - 1
                if ids_mask[id] == 0
                    optimized_count += 1
                end
            end

            # make sure that we can update the block at boundary + 1
            optimized_count = max(0, optimized_count - 1)
        end

        # Run it.
        fdshpl_length_gauge_one_step(crt_shwave, rt, pw.shgrid, pw.delta_t, Et_data[i], m; optimizing_count=optimized_count)
        
        # Printing Logging message
        if (i - 1) % 200 == 0
            en = get_energy_sh_mbunch(crt_shwave, rt, pw.shgrid, m)
            norm_value = sum(map(rvec->dot(rvec, rvec), crt_shwave))
            gs_ppls_value = calc_ground_state_population(crt_shwave, init_shwave, pw.shgrid)
            push!(norm_list, norm_value)
            push!(gs_population_list, gs_ppls_value)
            push!(energy_list, en)
            #println("step $(i-1) energy = $en, gs_population = $gs_ppls_value, norm = $norm_value")
            println("step $(i-1) energy = $en, bundle_size = $(bundle_size - optimized_count), norm = $norm_value")
        end

        if i < start_rcd_step || i > end_rcd_step
            continue
        end

        # HHG Part
        Threads.@threads for j = 1: bundle_size - optimized_count
            l = abs(m) + j - 1
            id = head_ptr + j - 1
            id1 = get_index_from_lm(l - 1, m, pw.shgrid.l_num)
            id2 = get_index_from_lm(l + 1, m, pw.shgrid.l_num)

            integral_buffer[l + 1] = 0      # clear it first
            # if id1 != -1    
            #     @fastmath integral_buffer[l + 1] += dot(crt_shwave[id], crt_shwave[id1] .* dU_data) * pw.delta_r * c_expr(l - 1, m)
            # end
            # if id2 != -1
            #     @fastmath integral_buffer[l + 1] += dot(crt_shwave[id], crt_shwave[id2] .* dU_data) * pw.delta_r * c_expr(l, m)
            # end
            if id1 != -1    # if id1 (l - 1, m) is in bound, then add 
                for k = 1: pw.Nr
                    integral_buffer[l + 1] += conj(crt_shwave[id][k]) * crt_shwave[id1][k] * dU_data[k] * c_expr(l - 1, m) * pw.delta_r
                end
            end
            if id2 != -1    # if id2 (l + 1, m) is in bound, then add
                for k = 1: pw.Nr
                    integral_buffer[l + 1] += conj(crt_shwave[id][k]) * crt_shwave[id2][k] * dU_data[k] * c_expr(l, m) * pw.delta_r
                end
            end
        end
        for j = 1: bundle_size - optimized_count
            l = abs(m) + j - 1
            hhg_integral_t[i] += integral_buffer[l + 1]
        end
    end

    return hhg_integral_t, energy_list
end