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

function fdshpl_one_step(crt_shwave, rt::tdse_sh_rt, shgrid, delta_t, At, At_next, m)
    bundle_size = shgrid.l_num - abs(m)
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

function fdshpl_length_gauge_one_step(crt_shwave, rt::tdse_sh_rt, shgrid, delta_t, Et, m)
    bundle_size = shgrid.l_num - abs(m)
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
            push!(energy_list, en)
            println("step $(i-1) energy = $en")
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