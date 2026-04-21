# ---------------------------
#    fd_sh functions below
# ---------------------------
c_expr(l, m) = sqrt(((l + 1) ^ 2 - m ^ 2) / ((2 * l + 1) * (2 * l + 3)))

function build_z_operator_bundle(m::Int64, l_num::Int64)
    bundle_size = l_num - abs(m)
    z_op = zeros(Float64, bundle_size, bundle_size)
    for j = 1:bundle_size
        l = abs(m) + j - 1
        if j > 1
            z_op[j, j - 1] = c_expr(l - 1, m)
        end
        if j < bundle_size
            z_op[j, j + 1] = c_expr(l, m)
        end
    end
    return z_op
end

function gauge_transform_v2l_linear_bundle!(bundle_wave_l, crt_shwave, bundle_ids::Vector{Int64},
                                            z_eigvecs, z_eigvals, A::Float64, r_values::Vector{Float64},
                                            coeff_v, coeff_eig)
    bundle_size = length(bundle_ids)
    @inbounds for j = 1:bundle_size
        id = bundle_ids[j]
        coeff_v[j, :] .= crt_shwave[id]
    end

    mul!(coeff_eig, adjoint(z_eigvecs), coeff_v)
    @inbounds for p = 1:bundle_size
        lam = z_eigvals[p]
        for k = 1:length(r_values)
            coeff_eig[p, k] *= exp(im * A * lam * r_values[k])
        end
    end
    mul!(coeff_v, z_eigvecs, coeff_eig)

    @inbounds for j = 1:bundle_size
        bundle_wave_l[j] .= coeff_v[j, :]
    end
end


@doc raw"""
    calc_ground_state_population(crt_shwave::shwave_t, init_shwave::shwave_t, shgrid::GridSH) -> ComplexF64

Calculate the ground state population 
"""
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

#  H_mix
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

    # apply H_atom
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
            #     @fastmath integral_buffer[l + 1] += dot(crt_shwave[id], crt_shwave[id1] .* dU_data) * c_expr(l - 1, m)
            # end
            # if id2 != -1    # if id2 (l + 1, m) is in bound, then add
            #     @fastmath integral_buffer[l + 1] += dot(crt_shwave[id], crt_shwave[id2] .* dU_data) * c_expr(l, m)
            # end
            if id1 != -1
                for k = 1: pw.Nr
                    integral_buffer[l + 1] += conj(crt_shwave[id][k]) * crt_shwave[id1][k] * dU_data[k] * c_expr(l - 1, m)
                end
            end
            if id2 != -1
                for k = 1: pw.Nr
                    integral_buffer[l + 1] += conj(crt_shwave[id][k]) * crt_shwave[id2][k] * dU_data[k] * c_expr(l, m)
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



function _prepare_compact_bound_projection(bound_states, head_ptr::Int64, bundle_size::Int64, Nr::Int64)
    compact_bundle_pos = zeros(Int64, length(bound_states))
    bound_norms = zeros(Float64, length(bound_states))
    for sid in eachindex(bound_states)
        state = bound_states[sid]
        if length(state.radial) != Nr
            throw(ArgumentError("bound state $sid has radial length $(length(state.radial)), expected $Nr"))
        end
        pos = state.id - head_ptr + 1
        if 1 <= pos <= bundle_size
            compact_bundle_pos[sid] = pos
            bound_norms[sid] = max(real(dot(state.radial, state.radial)), eps(Float64))
        else
            compact_bundle_pos[sid] = 0
            bound_norms[sid] = 1.0
        end
    end
    return compact_bundle_pos, bound_norms
end


function tdseln_sh_mainloop_velocity_gauge_hhg_analysis(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, At_data, steps::Int64, Ri_tsurf::Float64, bound_states; m::Int64 = 0, start_rcd_step::Int64 = 1, end_rcd_step::Int64 = 1000000, gauge_transform_for_decomposition::Bool = true)
    # hhg
    dU_data = get_derivative_two_order(pw.po_data_r, pw.delta_r)
    hhg_integral_t = zeros(ComplexF64, steps)
    hhg_integral_t_free = zeros(ComplexF64, steps)
    hhg_integral_t_bound = zeros(ComplexF64, steps)

    # wave decomposition buffers (m-bundle)
    head_ptr = get_index_from_lm(abs(m), m, pw.shgrid.l_num)
    bundle_size = pw.shgrid.l_num - abs(m)
    bundle_ids = [head_ptr + j - 1 for j = 1:bundle_size]
    bundle_wave_bound = [zeros(ComplexF64, pw.Nr) for _ = 1:bundle_size]
    bundle_wave_free = [zeros(ComplexF64, pw.Nr) for _ = 1:bundle_size]
    bundle_wave_length = [zeros(ComplexF64, pw.Nr) for _ = 1:bundle_size]
    use_compact_states = !isempty(bound_states) && (bound_states[1] isa compact_bound_state_t)
    compact_bundle_pos = zeros(Int64, length(bound_states))
    bound_norms = zeros(Float64, length(bound_states))
    bound_coeffs = zeros(ComplexF64, length(bound_states))

    # precompute m-bundle couplings for hhg integral
    id_left = zeros(Int64, bundle_size)
    id_right = zeros(Int64, bundle_size)
    c_left = zeros(Float64, bundle_size)
    c_right = zeros(Float64, bundle_size)
    for j = 1:bundle_size
        l = abs(m) + j - 1
        id_left[j] = get_index_from_lm(l - 1, m, pw.shgrid.l_num)
        id_right[j] = get_index_from_lm(l + 1, m, pw.shgrid.l_num)
        c_left[j] = (id_left[j] == -1) ? 0.0 : c_expr(l - 1, m)
        c_right[j] = (id_right[j] == -1) ? 0.0 : c_expr(l, m)
    end

    # gauge-transform buffers (V->L)
    r_values = get_linspace(pw.shgrid.rgrid)
    z_op = build_z_operator_bundle(m, pw.shgrid.l_num)
    z_eig = eigen(Hermitian(z_op))
    z_eigvecs = z_eig.vectors
    z_eigvals = z_eig.values
    coeff_v = zeros(ComplexF64, bundle_size, pw.Nr)
    coeff_eig = zeros(ComplexF64, bundle_size, pw.Nr)

    if use_compact_states
        compact_bundle_pos, bound_norms = _prepare_compact_bound_projection(bound_states, head_ptr, bundle_size, pw.Nr)
    else
        # precompute bound-state norms on propagated m-bundle
        for sid in eachindex(bound_states)
            state_norm = 0.0
            for j = 1:bundle_size
                id = bundle_ids[j]
                state_norm += real(dot(bound_states[sid][id], bound_states[sid][id]))
            end
            bound_norms[sid] = max(state_norm, eps(Float64))
        end
    end

    # optimization
    TDSE_LN_OPTIMIZE_THRESHOLD = 1e-15
    ids_mask = zeros(Int64, pw.shgrid.l_num ^ 2)   # 1 => selected.
    optimized_count::Int64 = 0

    for i in 1:steps
        # We do not need to get optimized_count every time, just sometimes.
        if (i - 1) % 20 == 0
            optimized_count = 0
            ids_mask .= 0

            Threads.@threads for j = 1:bundle_size
                id = head_ptr + j - 1
                tmp::Float64 = real(dot(crt_shwave[id], crt_shwave[id]))
                if tmp > TDSE_LN_OPTIMIZE_THRESHOLD
                    ids_mask[id] = 1
                end
                if id == head_ptr || id == head_ptr + 1
                    ids_mask[id] = 1
                end
            end

            for j = 1:bundle_size
                id = head_ptr + j - 1
                if ids_mask[id] == 0
                    optimized_count += 1
                end
            end

            optimized_count = max(0, optimized_count - 1)
        end

        At = At_data[i]
        At_next = At_data[min(i + 1, steps)]
        fdshpl_one_step(crt_shwave, rt, pw.shgrid, pw.delta_t, At, At_next, m; optimizing_count=optimized_count)

        if i < start_rcd_step || i > end_rcd_step
            continue
        end

        active_size = bundle_size - optimized_count

        # V->L gauge transform for decomposition if requested.
        if gauge_transform_for_decomposition
            gauge_transform_v2l_linear_bundle!(bundle_wave_length, crt_shwave, bundle_ids,
                                               z_eigvecs, z_eigvals, At, r_values,
                                               coeff_v, coeff_eig)
        end

        # Project out bound states on m-bundle in selected gauge.
        if use_compact_states
            for sid in eachindex(bound_states)
                j = compact_bundle_pos[sid]
                if j == 0
                    bound_coeffs[sid] = 0.0 + 0.0im
                    continue
                end
                psi_piece = gauge_transform_for_decomposition ? bundle_wave_length[j] : crt_shwave[bundle_ids[j]]
                bound_coeffs[sid] = dot(bound_states[sid].radial, psi_piece) / bound_norms[sid]
            end

            Threads.@threads for j = 1:bundle_size
                fill!(bundle_wave_bound[j], 0.0 + 0.0im)
            end
            for sid in eachindex(bound_states)
                coeff = bound_coeffs[sid]
                abs2(coeff) < 1e-30 && continue
                j = compact_bundle_pos[sid]
                j == 0 && continue
                bs_piece = bound_states[sid].radial
                piece = bundle_wave_bound[j]
                @inbounds for k = 1:pw.Nr
                    piece[k] += coeff * bs_piece[k]
                end
            end
            Threads.@threads for j = 1:bundle_size
                psi_piece = gauge_transform_for_decomposition ? bundle_wave_length[j] : crt_shwave[bundle_ids[j]]
                @inbounds for k = 1:pw.Nr
                    bundle_wave_free[j][k] = psi_piece[k] - bundle_wave_bound[j][k]
                end
            end
        else
            for sid in eachindex(bound_states)
                overlap = 0.0 + 0.0im
                for j = 1:bundle_size
                    id = bundle_ids[j]
                    psi_piece = gauge_transform_for_decomposition ? bundle_wave_length[j] : crt_shwave[id]
                    overlap += dot(bound_states[sid][id], psi_piece)
                end
                bound_coeffs[sid] = overlap / bound_norms[sid]
            end

            Threads.@threads for j = 1:bundle_size
                id = bundle_ids[j]
                psi_piece = gauge_transform_for_decomposition ? bundle_wave_length[j] : crt_shwave[id]
                fill!(bundle_wave_bound[j], 0.0 + 0.0im)
                for sid in eachindex(bound_states)
                    coeff = bound_coeffs[sid]
                    abs2(coeff) < 1e-30 && continue
                    bs_piece = bound_states[sid][id]
                    @inbounds for k = 1:pw.Nr
                        bundle_wave_bound[j][k] += coeff * bs_piece[k]
                    end
                end
                @inbounds for k = 1:pw.Nr
                    bundle_wave_free[j][k] = psi_piece[k] - bundle_wave_bound[j][k]
                end
            end
        end

        # Evaluate total / free / bound HHG in one pass.
        total_val = 0.0 + 0.0im
        free_val = 0.0 + 0.0im
        bound_val = 0.0 + 0.0im
        for j = 1:active_size
            id = bundle_ids[j]
            crt_piece = gauge_transform_for_decomposition ? bundle_wave_length[j] : crt_shwave[id]
            free_piece = bundle_wave_free[j]
            bound_piece = bundle_wave_bound[j]

            if id_left[j] != -1
                cc = c_left[j]
                left_idx = j - 1
                crt_piece_left = gauge_transform_for_decomposition ? bundle_wave_length[left_idx] : crt_shwave[id_left[j]]
                free_piece_left = bundle_wave_free[left_idx]
                bound_piece_left = bundle_wave_bound[left_idx]
                @inbounds for k = 1:pw.Nr
                    tmp = dU_data[k] * cc
                    total_val += conj(crt_piece[k]) * crt_piece_left[k] * tmp
                    free_val += conj(free_piece[k]) * free_piece_left[k] * tmp
                    bound_val += conj(bound_piece[k]) * bound_piece_left[k] * tmp
                end
            end

            if id_right[j] != -1
                cc = c_right[j]
                right_idx = j + 1
                crt_piece_right = gauge_transform_for_decomposition ? bundle_wave_length[right_idx] : crt_shwave[id_right[j]]
                free_piece_right = bundle_wave_free[right_idx]
                bound_piece_right = bundle_wave_bound[right_idx]
                @inbounds for k = 1:pw.Nr
                    tmp = dU_data[k] * cc
                    total_val += conj(crt_piece[k]) * crt_piece_right[k] * tmp
                    free_val += conj(free_piece[k]) * free_piece_right[k] * tmp
                    bound_val += conj(bound_piece[k]) * bound_piece_right[k] * tmp
                end
            end
        end
        hhg_integral_t[i] = total_val
        hhg_integral_t_free[i] = free_val
        hhg_integral_t_bound[i] = bound_val

        if (i - 1) % 200 == 0
            en = get_energy_sh_mbunch(crt_shwave, rt, pw.shgrid, m)
            norm_value = sum(map(rvec->dot(rvec, rvec), crt_shwave))
            bound_norm_value = sum(norm(bound_coeffs) .^ 2)
            free_norm_value = sum(map(rvec->dot(rvec, rvec), bundle_wave_free))
            norm_value = round(norm_value, digits=6)
            bound_norm_value = round(bound_norm_value, digits=6)
            free_norm_value = round(free_norm_value, digits=6)
            println("[TDSE] Runing TDSE-SH-linear (velocity gauge). step $(i-1) energy = $en, norm_value = $norm_value, bound_norm=$bound_norm_value, free_norm=$free_norm_value")
            println("     | block_size = $(bundle_size - optimized_count) ")
        end
    end
    return hhg_integral_t, hhg_integral_t_free, hhg_integral_t_bound
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
            #     @fastmath integral_buffer[l + 1] += dot(crt_shwave[id], crt_shwave[id1] .* dU_data) * c_expr(l - 1, m)
            # end
            # if id2 != -1
            #     @fastmath integral_buffer[l + 1] += dot(crt_shwave[id], crt_shwave[id2] .* dU_data) * c_expr(l, m)
            # end
            if id1 != -1    # if id1 (l - 1, m) is in bound, then add 
                for k = 1: pw.Nr
                    integral_buffer[l + 1] += conj(crt_shwave[id][k]) * crt_shwave[id1][k] * dU_data[k] * c_expr(l - 1, m)
                end
            end
            if id2 != -1    # if id2 (l + 1, m) is in bound, then add
                for k = 1: pw.Nr
                    integral_buffer[l + 1] += conj(crt_shwave[id][k]) * crt_shwave[id2][k] * dU_data[k] * c_expr(l, m)
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


function tdseln_sh_mainloop_length_gauge_hhg_analysis(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, Et_data, steps::Int64, Ri_tsurf::Float64, bound_states; m::Int64 = 0, start_rcd_step::Int64 = 1, end_rcd_step::Int64 = 1000000)
    # hhg
    dU_data = get_derivative_two_order(pw.po_data_r, pw.delta_r)
    hhg_integral_t = zeros(ComplexF64, steps)
    hhg_integral_t_free = zeros(ComplexF64, steps)
    hhg_integral_t_bound = zeros(ComplexF64, steps)

    # wave decomposition buffers (only m-bundle is required)
    bundle_wave_bound = [zeros(ComplexF64, pw.Nr) for _ = 1:pw.l_num]
    bundle_wave_free = [zeros(ComplexF64, pw.Nr) for _ = 1:pw.l_num]
    use_compact_states = !isempty(bound_states) && (bound_states[1] isa compact_bound_state_t)
    compact_bundle_pos = zeros(Int64, length(bound_states))
    bound_norms = zeros(Float64, length(bound_states))
    bound_coeffs = zeros(ComplexF64, length(bound_states))

    # optimization
    TDSE_LN_OPTIMIZE_THRESHOLD = 1e-15
    head_ptr = get_index_from_lm(abs(m), m, pw.shgrid.l_num)
    bundle_size = pw.shgrid.l_num - abs(m)
    ids_mask = zeros(Int64, pw.shgrid.l_num ^ 2)   # 1 => selected.
    optimized_count::Int64 = 0

    # precompute m-bundle ids and couplings for hhg integral
    bundle_ids = [head_ptr + j - 1 for j = 1:bundle_size]
    id_left = zeros(Int64, bundle_size)
    id_right = zeros(Int64, bundle_size)
    c_left = zeros(Float64, bundle_size)
    c_right = zeros(Float64, bundle_size)
    for j = 1:bundle_size
        l = abs(m) + j - 1
        id_left[j] = get_index_from_lm(l - 1, m, pw.shgrid.l_num)
        id_right[j] = get_index_from_lm(l + 1, m, pw.shgrid.l_num)
        c_left[j] = (id_left[j] == -1) ? 0.0 : c_expr(l - 1, m)
        c_right[j] = (id_right[j] == -1) ? 0.0 : c_expr(l, m)
    end

    if use_compact_states
        compact_bundle_pos, bound_norms = _prepare_compact_bound_projection(bound_states, head_ptr, bundle_size, pw.Nr)
    else
        # precompute bound-state norms only on the propagated m-bundle
        for sid in eachindex(bound_states)
            state_norm = 0.0
            for j = 1:bundle_size
                id = bundle_ids[j]
                state_norm += real(dot(bound_states[sid][id], bound_states[sid][id]))
            end
            bound_norms[sid] = max(state_norm, eps(Float64))
        end
    end

    for i = 1:steps
        # We do not need to get optimized_count every time, just sometimes.
        if (i - 1) % 20 == 0
            optimized_count = 0
            ids_mask .= 0

            Threads.@threads for j = 1:bundle_size
                id = head_ptr + j - 1
                tmp::Float64 = real(dot(crt_shwave[id], crt_shwave[id]))
                if tmp > TDSE_LN_OPTIMIZE_THRESHOLD
                    ids_mask[id] = 1
                end
                if id == head_ptr || id == head_ptr + 1
                    ids_mask[id] = 1
                end
            end

            for j = 1:bundle_size
                id = head_ptr + j - 1
                if ids_mask[id] == 0
                    optimized_count += 1
                end
            end

            optimized_count = max(0, optimized_count - 1)
        end

        fdshpl_length_gauge_one_step(crt_shwave, rt, pw.shgrid, pw.delta_t, Et_data[i], m; optimizing_count=optimized_count)

        # if (i - 1) % 200 == 0
        #     en = get_energy_sh_mbunch(crt_shwave, rt, pw.shgrid, m)
        #     norm_value = sum(map(rvec->dot(rvec, rvec), crt_shwave))
        #     println("[TDSE] Runing TDSE-SH-linear. step $(i-1) energy = $en, norm_value = $norm_value")
        #     println("     | block_size = $(bundle_size - optimized_count) ")
        # end

        if i < start_rcd_step || i > end_rcd_step
            continue
        end

        active_size = bundle_size - optimized_count

        # Project out bound states on m-bundle: wave_bound = sum_n |n><n|psi>, wave_free = psi - wave_bound
        if use_compact_states
            for sid in eachindex(bound_states)
                j = compact_bundle_pos[sid]
                if j == 0
                    bound_coeffs[sid] = 0.0 + 0.0im
                    continue
                end
                bound_coeffs[sid] = dot(bound_states[sid].radial, crt_shwave[bundle_ids[j]]) / bound_norms[sid]
            end

            Threads.@threads for j = 1:bundle_size
                fill!(bundle_wave_bound[j], 0.0 + 0.0im)
            end
            for sid in eachindex(bound_states)
                coeff = bound_coeffs[sid]
                abs2(coeff) < 1e-30 && continue
                j = compact_bundle_pos[sid]
                j == 0 && continue
                bs_piece = bound_states[sid].radial
                piece = bundle_wave_bound[j]
                @inbounds for k = 1: pw.Nr
                    piece[k] += coeff * bs_piece[k]
                end
            end
            Threads.@threads for j = 1:bundle_size
                id = bundle_ids[j]
                crt_piece = crt_shwave[id]
                @inbounds for k = 1:pw.Nr
                    bundle_wave_free[j][k] = crt_piece[k] - bundle_wave_bound[j][k]
                end
            end
        else
            for sid = eachindex(bound_states)
                overlap = 0.0 + 0.0im
                for j = 1:bundle_size
                    id = bundle_ids[j]
                    overlap += dot(bound_states[sid][id], crt_shwave[id])
                end
                bound_coeffs[sid] = overlap / bound_norms[sid]
            end

            Threads.@threads for j = 1:bundle_size
                id = bundle_ids[j]
                fill!(bundle_wave_bound[j], 0.0 + 0.0im)
                for sid = eachindex(bound_states)
                    coeff = bound_coeffs[sid]
                    abs2(coeff) < 1e-30 && continue
                    bs_piece = bound_states[sid][id]
                    @inbounds for k = 1: pw.Nr
                        bundle_wave_bound[j][k] += coeff * bs_piece[k]
                    end
                end
                crt_piece = crt_shwave[id]
                @inbounds for k = 1:pw.Nr
                    bundle_wave_free[j][k] = crt_piece[k] - bundle_wave_bound[j][k]
                end
            end
        end

        # Evaluate total / free / bound HHG in one pass
        total_val = 0.0 + 0.0im
        free_val = 0.0 + 0.0im
        bound_val = 0.0 + 0.0im
        for j = 1:active_size
            id = bundle_ids[j]
            crt_piece = crt_shwave[id]
            free_piece = bundle_wave_free[j]
            bound_piece = bundle_wave_bound[j]

            if id_left[j] != -1
                cc = c_left[j]
                left_idx = j - 1
                crt_piece_left = crt_shwave[id_left[j]]
                free_piece_left = bundle_wave_free[left_idx]
                bound_piece_left = bundle_wave_bound[left_idx]
                @inbounds for k = 1:pw.Nr
                    tmp = dU_data[k] * cc
                    total_val += conj(crt_piece[k]) * crt_piece_left[k] * tmp
                    free_val += conj(free_piece[k]) * free_piece_left[k] * tmp
                    bound_val += conj(bound_piece[k]) * bound_piece_left[k] * tmp
                end
            end

            if id_right[j] != -1
                cc = c_right[j]
                right_idx = j + 1
                crt_piece_right = crt_shwave[id_right[j]]
                free_piece_right = bundle_wave_free[right_idx]
                bound_piece_right = bundle_wave_bound[right_idx]
                @inbounds for k = 1:pw.Nr
                    tmp = dU_data[k] * cc
                    total_val += conj(crt_piece[k]) * crt_piece_right[k] * tmp
                    free_val += conj(free_piece[k]) * free_piece_right[k] * tmp
                    bound_val += conj(bound_piece[k]) * bound_piece_right[k] * tmp
                end
            end
        end
        hhg_integral_t[i] = total_val
        hhg_integral_t_free[i] = free_val
        hhg_integral_t_bound[i] = bound_val

        if (i - 1) % 200 == 0
            en = get_energy_sh_mbunch(crt_shwave, rt, pw.shgrid, m)
            norm_value = sum(map(rvec->dot(rvec, rvec), crt_shwave))
            bound_norm_value = sum(norm(bound_coeffs) .^ 2)
            free_norm_value = sum(map(rvec->dot(rvec, rvec), bundle_wave_free))
            norm_value = round(norm_value, digits=6)
            bound_norm_value = round(bound_norm_value, digits=6)
            free_norm_value = round(free_norm_value, digits=6)
            println("[TDSE] Runing TDSE-SH-linear (length gauge). step $(i-1) energy = $en, norm_value = $norm_value, bound_norm=$bound_norm_value, free_norm=$free_norm_value")
            println("     | block_size = $(bundle_size - optimized_count) ")
        end
    end

    return hhg_integral_t, hhg_integral_t_free, hhg_integral_t_bound
end


function tdseln_sh_mainloop_length_gauge_hhg_analysis_dipole(crt_shwave, pw::physics_world_sh_t, rt::tdse_sh_rt, Et_data, steps::Int64, Ri_tsurf::Float64, bound_states; m::Int64 = 0, start_rcd_step::Int64 = 1, end_rcd_step::Int64 = 1000000)
    # dipole moment in z direction, z(t) = <Psi|z|Psi>, where z = r * cos(theta)
    r_data = get_linspace(pw.shgrid.rgrid)
    dipole_t = zeros(ComplexF64, steps)
    dipole_t_free = zeros(ComplexF64, steps)
    dipole_t_bound = zeros(ComplexF64, steps)

    # wave decomposition buffers (only m-bundle is required)
    bundle_wave_bound = [zeros(ComplexF64, pw.Nr) for _ = 1:pw.l_num]
    bundle_wave_free = [zeros(ComplexF64, pw.Nr) for _ = 1:pw.l_num]
    use_compact_states = !isempty(bound_states) && (bound_states[1] isa compact_bound_state_t)
    compact_bundle_pos = zeros(Int64, length(bound_states))
    bound_norms = zeros(Float64, length(bound_states))
    bound_coeffs = zeros(ComplexF64, length(bound_states))

    # optimization
    TDSE_LN_OPTIMIZE_THRESHOLD = 1e-15
    head_ptr = get_index_from_lm(abs(m), m, pw.shgrid.l_num)
    bundle_size = pw.shgrid.l_num - abs(m)
    ids_mask = zeros(Int64, pw.shgrid.l_num ^ 2)   # 1 => selected.
    optimized_count::Int64 = 0

    # precompute m-bundle ids and couplings for dipole integral
    bundle_ids = [head_ptr + j - 1 for j = 1:bundle_size]
    id_left = zeros(Int64, bundle_size)
    id_right = zeros(Int64, bundle_size)
    c_left = zeros(Float64, bundle_size)
    c_right = zeros(Float64, bundle_size)
    for j = 1:bundle_size
        l = abs(m) + j - 1
        id_left[j] = get_index_from_lm(l - 1, m, pw.shgrid.l_num)
        id_right[j] = get_index_from_lm(l + 1, m, pw.shgrid.l_num)
        c_left[j] = (id_left[j] == -1) ? 0.0 : c_expr(l - 1, m)
        c_right[j] = (id_right[j] == -1) ? 0.0 : c_expr(l, m)
    end

    if use_compact_states
        compact_bundle_pos, bound_norms = _prepare_compact_bound_projection(bound_states, head_ptr, bundle_size, pw.Nr)
    else
        # precompute bound-state norms only on the propagated m-bundle
        for sid in eachindex(bound_states)
            state_norm = 0.0
            for j = 1:bundle_size
                id = bundle_ids[j]
                state_norm += real(dot(bound_states[sid][id], bound_states[sid][id]))
            end
            bound_norms[sid] = max(state_norm, eps(Float64))
        end
    end

    for i = 1:steps
        # We do not need to get optimized_count every time, just sometimes.
        if (i - 1) % 20 == 0
            optimized_count = 0
            ids_mask .= 0

            Threads.@threads for j = 1:bundle_size
                id = head_ptr + j - 1
                tmp::Float64 = real(dot(crt_shwave[id], crt_shwave[id]))
                if tmp > TDSE_LN_OPTIMIZE_THRESHOLD
                    ids_mask[id] = 1
                end
                if id == head_ptr || id == head_ptr + 1
                    ids_mask[id] = 1
                end
            end

            for j = 1:bundle_size
                id = head_ptr + j - 1
                if ids_mask[id] == 0
                    optimized_count += 1
                end
            end

            optimized_count = max(0, optimized_count - 1)
        end

        fdshpl_length_gauge_one_step(crt_shwave, rt, pw.shgrid, pw.delta_t, Et_data[i], m; optimizing_count=optimized_count)

        if i < start_rcd_step || i > end_rcd_step
            continue
        end

        active_size = bundle_size - optimized_count

        # Project out bound states on m-bundle: wave_bound = sum_n |n><n|psi>, wave_free = psi - wave_bound
        if use_compact_states
            for sid in eachindex(bound_states)
                j = compact_bundle_pos[sid]
                if j == 0
                    bound_coeffs[sid] = 0.0 + 0.0im
                    continue
                end
                bound_coeffs[sid] = dot(bound_states[sid].radial, crt_shwave[bundle_ids[j]]) / bound_norms[sid]
            end

            Threads.@threads for j = 1:bundle_size
                fill!(bundle_wave_bound[j], 0.0 + 0.0im)
            end
            for sid in eachindex(bound_states)
                coeff = bound_coeffs[sid]
                abs2(coeff) < 1e-30 && continue
                j = compact_bundle_pos[sid]
                j == 0 && continue
                bs_piece = bound_states[sid].radial
                piece = bundle_wave_bound[j]
                @inbounds for k = 1: pw.Nr
                    piece[k] += coeff * bs_piece[k]
                end
            end
            Threads.@threads for j = 1:bundle_size
                id = bundle_ids[j]
                crt_piece = crt_shwave[id]
                @inbounds for k = 1:pw.Nr
                    bundle_wave_free[j][k] = crt_piece[k] - bundle_wave_bound[j][k]
                end
            end
        else
            for sid = eachindex(bound_states)
                overlap = 0.0 + 0.0im
                for j = 1:bundle_size
                    id = bundle_ids[j]
                    overlap += dot(bound_states[sid][id], crt_shwave[id])
                end
                bound_coeffs[sid] = overlap / bound_norms[sid]
            end

            Threads.@threads for j = 1:bundle_size
                id = bundle_ids[j]
                fill!(bundle_wave_bound[j], 0.0 + 0.0im)
                for sid = eachindex(bound_states)
                    coeff = bound_coeffs[sid]
                    abs2(coeff) < 1e-30 && continue
                    bs_piece = bound_states[sid][id]
                    @inbounds for k = 1: pw.Nr
                        bundle_wave_bound[j][k] += coeff * bs_piece[k]
                    end
                end
                crt_piece = crt_shwave[id]
                @inbounds for k = 1:pw.Nr
                    bundle_wave_free[j][k] = crt_piece[k] - bundle_wave_bound[j][k]
                end
            end
        end

        # Evaluate total / free / bound dipole z(t) in one pass
        total_val = 0.0 + 0.0im
        free_val = 0.0 + 0.0im
        bound_val = 0.0 + 0.0im
        for j = 1:active_size
            id = bundle_ids[j]
            crt_piece = crt_shwave[id]
            free_piece = bundle_wave_free[j]
            bound_piece = bundle_wave_bound[j]

            if id_left[j] != -1
                cc = c_left[j]
                left_idx = j - 1
                crt_piece_left = crt_shwave[id_left[j]]
                free_piece_left = bundle_wave_free[left_idx]
                bound_piece_left = bundle_wave_bound[left_idx]
                @inbounds for k = 1:pw.Nr
                    tmp = r_data[k] * cc
                    total_val += conj(crt_piece[k]) * crt_piece_left[k] * tmp
                    free_val += conj(free_piece[k]) * free_piece_left[k] * tmp
                    bound_val += conj(bound_piece[k]) * bound_piece_left[k] * tmp
                end
            end

            if id_right[j] != -1
                cc = c_right[j]
                right_idx = j + 1
                crt_piece_right = crt_shwave[id_right[j]]
                free_piece_right = bundle_wave_free[right_idx]
                bound_piece_right = bundle_wave_bound[right_idx]
                @inbounds for k = 1:pw.Nr
                    tmp = r_data[k] * cc
                    total_val += conj(crt_piece[k]) * crt_piece_right[k] * tmp
                    free_val += conj(free_piece[k]) * free_piece_right[k] * tmp
                    bound_val += conj(bound_piece[k]) * bound_piece_right[k] * tmp
                end
            end
        end
        dipole_t[i] = total_val
        dipole_t_free[i] = free_val
        dipole_t_bound[i] = bound_val

        if (i - 1) % 200 == 0
            en = get_energy_sh_mbunch(crt_shwave, rt, pw.shgrid, m)
            norm_value = sum(map(rvec->dot(rvec, rvec), crt_shwave))
            bound_norm_value = sum(norm(bound_coeffs) .^ 2)
            free_norm_value = sum(map(rvec->dot(rvec, rvec), bundle_wave_free))
            norm_value = round(norm_value, digits=6)
            bound_norm_value = round(bound_norm_value, digits=6)
            free_norm_value = round(free_norm_value, digits=6)
            println("[TDSE] Runing TDSE-SH-linear (length gauge, dipole form). step $(i-1) energy = $en, norm_value = $norm_value, bound_norm=$bound_norm_value, free_norm=$free_norm_value")
            println("     | block_size = $(bundle_size - optimized_count) ")
        end
    end

    return dipole_t, dipole_t_free, dipole_t_bound
end
