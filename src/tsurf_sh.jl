
function hanning_window(t, T)
    if t < T / 2
        return 1.0
    else
        return (1 - cos(2 * pi * t / T)) / 2
    end
end

c_co(l, m) = sqrt(((l + 1) ^ 2 - m ^ 2) / ((2 * l + 1) * (2 * l + 3)))
b_co(l, m) = sqrt(((l + m - 1) * (l + m)) / (2 * (2 * l - 1) * (2 * l + 1)))
d_co(l, m) = sqrt(((l + m + 1) * (l + m + 2) * (l + 1)) / ((2 * l + 2) * (2 * l + 3) * (2 * l + 1)))

const TSURF_MODE_PL = 1
const TSURF_MODE_ELLI = 2

@kwdef struct tsurf_k_space_t
    k_r_range::Vector{Float64}
    k_theta_range::Vector{Float64}
    k_phi_range::Vector{Float64}

    k_collection::Vector{Tuple{Float64, Float64, Float64}}   # (kx, ky, kz)
    ijk_mapping::Vector{Tuple{Int64, Int64, Int64}}
end



# Based on [ V. Mosert, D. Bauer / Computer Physics Communications 207 (2016) 452–463 ]
# available for all situations.
function tsurf_sh(pw::physics_world_sh_t, phi_record, dphi_record, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, momentum_range, tsurf_mode)
    shgrid = pw.shgrid
    delta_t = pw.delta_t
    # lmap, mmap: mapping the index with l, m
    lmap, mmap = create_lmmap(shgrid.l_num)

    # 5 integrals used in formula
    integral_vals_0 = Dict{Tuple{Int64, Int64}, Vector{ComplexF64}}()
    integral_vals_1 = Dict{Tuple{Int64, Int64}, Vector{ComplexF64}}()
    integral_vals_2 = Dict{Tuple{Int64, Int64}, Vector{ComplexF64}}()
    integral_vals_3 = Dict{Tuple{Int64, Int64}, Vector{ComplexF64}}()
    integral_vals_4 = Dict{Tuple{Int64, Int64}, Vector{ComplexF64}}()

    # allocate memory for integrals
    print("[t-SURFF] Allocating memory ...  ")
    for id1 = 1: shgrid.l_num ^ 2
        for id2 = 1: shgrid.l_num ^ 2
            integral_vals_0[(id1, id2)] = zeros(ComplexF64, length(momentum_range))
            integral_vals_1[(id1, id2)] = zeros(ComplexF64, length(momentum_range))
            integral_vals_2[(id1, id2)] = zeros(ComplexF64, length(momentum_range))
            integral_vals_3[(id1, id2)] = zeros(ComplexF64, length(momentum_range))
            integral_vals_4[(id1, id2)] = zeros(ComplexF64, length(momentum_range))
        end
    end

    # reserve for out-of-range index (equal to zero)
    for id = 1: shgrid.l_num ^ 2
        integral_vals_0[(-1, id)] = zeros(ComplexF64, length(momentum_range))
        integral_vals_1[(-1, id)] = zeros(ComplexF64, length(momentum_range))
        integral_vals_2[(-1, id)] = zeros(ComplexF64, length(momentum_range))
        integral_vals_3[(-1, id)] = zeros(ComplexF64, length(momentum_range))
        integral_vals_4[(-1, id)] = zeros(ComplexF64, length(momentum_range))
    end
    println("Completed.")

    # definition of alpha and Fi (i=0,1,2,3)
    At_len = length(At_data_x)
    alpha_x = copy(At_data_x)
    alpha_y = copy(At_data_y)
    alpha_z = copy(At_data_z)

    # alpha = sum of At_data (obviously)
    alpha_x[1] = At_data_x[1]
    alpha_y[1] = At_data_y[1]
    alpha_z[1] = At_data_z[1]
    for i = 2: At_len
        alpha_x[i] = alpha_x[i - 1] + At_data_x[i]
        alpha_y[i] = alpha_y[i - 1] + At_data_y[i]
        alpha_z[i] = alpha_z[i - 1] + At_data_z[i]
    end
    alpha_x .*= delta_t
    alpha_y .*= delta_t
    alpha_z .*= delta_t

    alpha_x .+= 1e-50   # avoid zero (otherwise alpha_θ or alpha_ϕ will get NaN)
    alpha_y .+= 1e-50
    alpha_z .+= 1e-50

    # transform alpha_xyz to alpha_r, alpha_θ and alpha_ϕ
    alpha_r = zeros(Float64, At_len)
    alpha_theta = zeros(Float64, At_len)
    alpha_phi = zeros(Float64, At_len)
    for i = 1: At_len
        alpha_r[i], alpha_theta[i], alpha_phi[i] = xyz_to_sphere(alpha_x[i], alpha_y[i], alpha_z[i])
    end

    # Fi
    F0_val = [conj(At_data_x[i] + im * At_data_y[i]) for i = 1: At_len]
    F1_val = [(At_data_x[i] + im * At_data_y[i]) for i = 1: At_len]
    F2_val = copy(At_data_z)
    # F3_val = 1.0

    print("[t-SURFF] Preparing buffer ...  ")
    # used for pruning
    phi_is_empty_map = zeros(Bool, shgrid.l_num ^ 2)
    for id1 = 1: shgrid.l_num ^ 2
        if abs(sum(phi_record[id1])) < 1e-30
            phi_is_empty_map[id1] = true
        end
    end


    ### preparing buffer (to speed up) ###
    sh_bessel_buffer = [zeros(Float64, length(momentum_range), length(t_linspace)) for _ = 1: shgrid.l_num]
    exp_buffer = zeros(ComplexF64, length(momentum_range), length(t_linspace))
    window_buffer = zeros(Float64, length(t_linspace))

    # evaluate Ylm_alpha_buffer
    Ylm_alpha_buffer = [computeYlm(alpha_theta[i], alpha_phi[i], lmax=shgrid.l_num - 1) for i = 1: At_len]
    
    # get spherical_bessel_func(l2, k * alpha_r[j]) -> (l2, k, j)
    for l2 = 0: shgrid.l_num - 1
        for i in eachindex(momentum_range)
            @inbounds k = momentum_range[i]
            for j in eachindex(t_linspace)
                sh_bessel_buffer[l2 + 1][i, j] = spherical_bessel_func(l2, k * alpha_r[j])
            end
        end
    end

    # get exp(im * k^2 * t / 2) -> (k, t)
    for i in eachindex(momentum_range)
        @inbounds k = momentum_range[i]
        for j in eachindex(t_linspace)
            @inbounds t = t_linspace[j]
            exp_buffer[i, j] = exp(im * k^2 * t / 2)
        end
    end

    # get hanning_window(t, last(t_linspace))
    for j in eachindex(t_linspace)
        @inbounds t = t_linspace[j]
        window_buffer[j] = hanning_window(t, last(t_linspace))
    end
    println("Completed.")


    ### evaluating integrals ###
    println("[t-SURFF] Start evaluating intergrals.")
    for id1 = 1: shgrid.l_num ^ 2
        l1 = lmap[id1]
        m1 = mmap[id1]

        if tsurf_mode == TSURF_MODE_PL && m1 != 0   # for linear polarization, we can cut-off those parts.
            continue 
        end
        if phi_is_empty_map[id1] == true    # pruning, if phi was empty, the next steps would become useless.
            continue
        end

        for id2 = 1: shgrid.l_num ^ 2
            l2 = lmap[id2]
            m2 = mmap[id2]

            if tsurf_mode == TSURF_MODE_PL && m2 != 0   # the same.
                continue 
            end

            println("[t-SURFF] Evaluating intergral. Now l1 = $l1, m1 = $m1, l2 = $l2, m2 = $m2.")
            Threads.@threads for i in eachindex(momentum_range)
                # @inbounds k = momentum_range[i]
                for j in eachindex(t_linspace)
                    # @inbounds t = t_linspace[j]
                    @inbounds @fastmath tmp_val = window_buffer[j] * delta_t * exp_buffer[i, j] * sh_bessel_buffer[l2 + 1][i, j] * conj(Ylm_alpha_buffer[j][(l2, m2)])
                    @inbounds @fastmath integral_vals_0[(id1, id2)][i] += tmp_val * F0_val[j] * phi_record[id1][j]
                    @inbounds @fastmath integral_vals_1[(id1, id2)][i] += tmp_val * F1_val[j] * phi_record[id1][j]
                    @inbounds @fastmath integral_vals_2[(id1, id2)][i] += tmp_val * F2_val[j] * phi_record[id1][j]
                    @inbounds @fastmath integral_vals_3[(id1, id2)][i] += tmp_val * 1.0 * phi_record[id1][j]
            
                    @inbounds @fastmath integral_vals_4[(id1, id2)][i] += tmp_val * dphi_record[id1][j]
                end
            end
        end
    end


    ### evaluating A_tsurff(k) ###
    a_tsurff = [zeros(ComplexF64, length(momentum_range)) for i = 1: shgrid.l_num ^ 2]
    
    # preparing sh_bessel_buffer_2
    sh_bessel_buffer_2 = [zeros(Float64, length(momentum_range)) for l = 0: shgrid.l_num]
    for l1 = 0: shgrid.l_num    # important: l1 must be 0 ~ l_max + 1, because sh_bessel_buffer_2[l1 + 1] will be used.
        for i in eachindex(momentum_range)
            k = momentum_range[i]
            sh_bessel_buffer_2[l1 + 1][i] = spherical_bessel_func(l1, k * Ri_tsurf)
        end
    end

    println("[t-SURFF] Start evaluating A_tsurff.")
    for id = 1: shgrid.l_num ^ 2
        @inbounds l = lmap[id]
        @inbounds m = mmap[id]
        if tsurf_mode == TSURF_MODE_PL && m != 0   # for linear polarization, we can cut-off those parts.
            continue 
        end

        println("[t-SURFF] Evaluating A_tsurff... Now L = $l, M = $m")
        for id1 = 1: shgrid.l_num ^ 2
            @inbounds l1 = lmap[id1]
            @inbounds m1 = mmap[id1]
            if tsurf_mode == TSURF_MODE_PL && m1 != 0   # for linear polarization, we can cut-off those parts.
                continue 
            end

            for l2 = 0: shgrid.l_num - 1
                id2 = get_index_from_lm(l2, m - m1, shgrid.l_num)
                if id2 == -1
                    continue
                end

                @fastmath co1 = sqrt((2 * (2 * l1 + 1) * (2 * l2 + 1)) / (2 * l + 1))
                @fastmath co2 = CG_coefficient(l1, 0, l2, 0, l, 0) * CG_coefficient(l1, m1, l2, m - m1, l, m) * (-1.0im) ^ (l1 - l2 + 1.0)

                for i in eachindex(momentum_range)
                    @inbounds k = momentum_range[i]
                    @inbounds @fastmath co3 = sh_bessel_buffer_2[l1 + 1][i] * (integral_vals_4[(id1, id2)][i] - ((l1 + 1) / Ri_tsurf) * integral_vals_3[(id1, id2)][i])
                    @inbounds @fastmath co4 = k * integral_vals_3[(id1, id2)][i] * sh_bessel_buffer_2[l1 + 2][i]
                    @inbounds @fastmath co5 = im * sqrt(2.0) * sh_bessel_buffer_2[l1 + 1][i]
                    @inbounds @fastmath co6 = (b_co(l1, -m1) * integral_vals_1[(get_index_from_lm(l1 - 1, m1 + 1, shgrid.l_num), id2)][i] -
                        d_co(l1, m1) * integral_vals_1[(get_index_from_lm(l1 + 1, m1 + 1, shgrid.l_num), id2)][i] -
                        b_co(l1, m1) * integral_vals_0[(get_index_from_lm(l1 - 1, m1 - 1, shgrid.l_num), id2)][i] +
                        d_co(l1, -m1) * integral_vals_0[(get_index_from_lm(l1 + 1, m1 - 1, shgrid.l_num), id2)][i] +
                        sqrt(2.0) * (c_co(l1 - 1, m1) * integral_vals_2[(get_index_from_lm(l1 - 1, m1, shgrid.l_num), id2)][i] +
                            c_co(l1, m1) * integral_vals_2[(get_index_from_lm(l1 + 1, m1, shgrid.l_num), id2)][i]))

                    @inbounds @fastmath a_tsurff[id][i] += Ri_tsurf * co1 * co2 * (co3 + co4 + co5 * co6)
                end
            end
        end
    end

    println("[t-SURFF] t-SURFF procedure ended.")
    return a_tsurff
end


function tsurf_sh_vector(pw::physics_world_sh_t, phi_record, dphi_record, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_collection_xyz, tsurf_mode; cut_off_threhold::Float64=1e-6)
    shgrid = pw.shgrid
    delta_t = pw.delta_t
    # lmap, mmap: mapping the index with l, m
    lmap, mmap = create_lmmap(shgrid.l_num)

    # 5 integrals used in formula
    integral_vals_0 = Dict{Int64, Vector{ComplexF64}}()
    integral_vals_1 = Dict{Int64, Vector{ComplexF64}}()
    integral_vals_2 = Dict{Int64, Vector{ComplexF64}}()
    integral_vals_3 = Dict{Int64, Vector{ComplexF64}}()
    integral_vals_4 = Dict{Int64, Vector{ComplexF64}}()

    # allocate memory for integrals
    print("[t-SURFF] Allocating memory ...  ")
    k_len = length(k_collection_xyz)
    for id = 1: shgrid.l_num ^ 2
        integral_vals_0[id] = zeros(ComplexF64, k_len)
        integral_vals_1[id] = zeros(ComplexF64, k_len)
        integral_vals_2[id] = zeros(ComplexF64, k_len)
        integral_vals_3[id] = zeros(ComplexF64, k_len)
        integral_vals_4[id] = zeros(ComplexF64, k_len)
    end

    # reserve for out-of-range index (equal to zero)
    integral_vals_0[-1] = zeros(ComplexF64, k_len)
    integral_vals_1[-1] = zeros(ComplexF64, k_len)
    integral_vals_2[-1] = zeros(ComplexF64, k_len)
    integral_vals_3[-1] = zeros(ComplexF64, k_len)
    integral_vals_4[-1] = zeros(ComplexF64, k_len)
    println("Completed.")

    # definition of alpha and Fi (i=0,1,2,3)
    At_len = length(At_data_x)
    alpha_x = get_integral(At_data_x, delta_t)
    alpha_y = get_integral(At_data_y, delta_t)
    alpha_z = get_integral(At_data_z, delta_t)

    alpha_x .+= 1e-50   # avoid zero (otherwise alpha_θ or alpha_ϕ will get NaN)
    alpha_y .+= 1e-50
    alpha_z .+= 1e-50

    # transform alpha_xyz to alpha_r, alpha_θ and alpha_ϕ
    # alpha_r = zeros(Float64, At_len)
    # alpha_theta = zeros(Float64, At_len)
    # alpha_phi = zeros(Float64, At_len)
    # for i = 1: At_len
    #     alpha_r[i], alpha_theta[i], alpha_phi[i] = xyz_to_sphere(alpha_x[i], alpha_y[i], alpha_z[i])
    # end

    # Fi
    F0_val = [conj(At_data_x[i] + im * At_data_y[i]) for i = 1: At_len]
    F1_val = [(At_data_x[i] + im * At_data_y[i]) for i = 1: At_len]
    F2_val = copy(At_data_z)
    # F3_val = 1.0


    print("[t-SURFF] Preparing buffer ...  ")
    # used for pruning
    phi_is_empty_map = zeros(Bool, shgrid.l_num ^ 2)
    for id1 = 1: shgrid.l_num ^ 2
        if abs(sum(phi_record[id1])) < 1e-30
            phi_is_empty_map[id1] = true
        end
    end

    # pruning for t_linspace
    cut_off_position_t = 10^10      # A relative big number
    for id = 1: shgrid.l_num ^ 2
        tmp_pos::Int64 = 0
        for j in eachindex(t_linspace)
            if norm(phi_record[id][j]) > cut_off_threhold
                tmp_pos = j
                break
            end
        end
        if tmp_pos == 0
            tmp_pos = lastindex(t_linspace)
        end
        cut_off_position_t = min(cut_off_position_t, tmp_pos)
    end
    max_pos = lastindex(t_linspace)
    println("[t-SURFF] Pruning for t_linspace: cut_off_position_t=$cut_off_position_t, max_pos=$max_pos")
    t_linspace_pruned = t_linspace[cut_off_position_t: lastindex(t_linspace)]

    # get hanning_window(t, last(t_linspace))
    exp_buffer = zeros(ComplexF64, length(k_collection_xyz), length(t_linspace_pruned))
    window_buffer = zeros(Float64, length(t_linspace_pruned))
    for (i, k_vec) in enumerate(k_collection_xyz)
        k_abs = norm(k_vec)
        for (j, t) in enumerate(t_linspace_pruned)
            exp_buffer[i, j] = exp(im * t * k_abs ^ 2 / 2 + im * dot(k_vec, (alpha_x[j], alpha_y[j], alpha_z[j])))
        end
    end

    for j in eachindex(t_linspace_pruned)
        t = t_linspace_pruned[j]
        window_buffer[j] = hanning_window(t, last(t_linspace_pruned))
    end
    println("Completed.")


    ### evaluating integrals ###
    println("[t-SURFF] Start evaluating intergrals.")
    for id = 1: shgrid.l_num ^ 2
        l = lmap[id]
        m = mmap[id]
        
        if tsurf_mode == TSURF_MODE_PL && m != 0    # for linear polarization, we can cut-off those parts.
            continue
        end
        if phi_is_empty_map[id] == true    # pruning, if phi was empty, the next steps would become useless.
            continue
        end
        println("[t-SURFF] Evaluating intergral. Now l = $l, m = $m.")

        Threads.@threads for i in eachindex(k_collection_xyz)
            @inbounds @fastmath for j in eachindex(t_linspace_pruned)
                tmp_val = window_buffer[j] * delta_t * exp_buffer[i, j]
                integral_vals_0[id][i] += tmp_val * F0_val[j] * phi_record[id][j]
                integral_vals_1[id][i] += tmp_val * F1_val[j] * phi_record[id][j]
                integral_vals_2[id][i] += tmp_val * F2_val[j] * phi_record[id][j]
                integral_vals_3[id][i] += tmp_val * 1.0 * phi_record[id][j]
        
                integral_vals_4[id][i] += tmp_val * dphi_record[id][j]
            end
        end
    end

    ### evaluating A_tsurff(k_vec) ###
    a_tsurff_vec = [zeros(ComplexF64, length(k_collection_xyz)) for i = 1: shgrid.l_num ^ 2]

    # preparing sh_bessel_buffer_2
    sh_bessel_buffer_2 = [zeros(Float64, length(k_collection_xyz)) for l = 0: shgrid.l_num]
    for l = 0: shgrid.l_num    # important: l must be 0 ~ l_max + 1, because sh_bessel_buffer_2[l + 1] will be used.
        for (i, k_vec) in enumerate(k_collection_xyz)
            sh_bessel_buffer_2[l + 1][i] = spherical_bessel_func(l, norm(k_vec) * Ri_tsurf)
        end
    end

    println("[t-SURFF] Start evaluating A_tsurff.")
    for id = 1: shgrid.l_num ^ 2
        l = lmap[id]
        m = mmap[id]
        if tsurf_mode == TSURF_MODE_PL && m != 0   # for linear polarization, we can cut-off those parts.
            continue 
        end
        println("[t-SURFF] Evaluating A_tsurff. Now l = $l, m = $m.")

        id1 = get_index_from_lm(l - 1, m - 1, shgrid.l_num)
        id2 = get_index_from_lm(l + 1, m - 1, shgrid.l_num)
        id3 = get_index_from_lm(l - 1, m + 1, shgrid.l_num)
        id4 = get_index_from_lm(l + 1, m + 1, shgrid.l_num)
        id5 = get_index_from_lm(l - 1, m, shgrid.l_num)
        id6 = get_index_from_lm(l + 1, m, shgrid.l_num)
        for i in eachindex(k_collection_xyz)
            k_vec = k_collection_xyz[i]
            co0 = (Ri_tsurf * (-im) ^ (l + 1)) / (2 * pi) ^ 0.5
            co1 = sh_bessel_buffer_2[l + 1][i] * (integral_vals_4[id][i] - ((l + 1.0) / Ri_tsurf) * integral_vals_3[id][i])
            co2 = norm(k_vec) * sh_bessel_buffer_2[l + 2][i] * integral_vals_3[id][i]
            co3 = im * sqrt(2.0) * sh_bessel_buffer_2[l + 1][i]
            co4 = -(b_co(l, m) * integral_vals_0[id1][i] - d_co(l, -m) * integral_vals_0[id2][i])
            co5 = b_co(l, -m) * integral_vals_1[id3][i] - d_co(l, m) * integral_vals_1[id4][i]
            co6 = sqrt(2.0) * (c_co(l - 1, m) * integral_vals_2[id5][i] + c_co(l, m) * integral_vals_2[id6][i])
            
            a_tsurff_vec[id][i] = co0 * (co1 + co2 + co3 * (co4 + co5 + co6))
        end
    end

    println("[t-SURFF] t-SURFF procedure ended.")
    return a_tsurff_vec
end


function tsurf_get_energy_spectrum(a_tsurff_lm, momentum_range, pw::physics_world_sh_t)
    P_tsurff = zeros(Float64, length(a_tsurff_lm[1]))
    for (i, k) in enumerate(momentum_range)
        for id = 1: pw.l_num ^ 2
            P_tsurff[i] += k * norm(a_tsurff_lm[id][i]) ^ 2
        end
    end
    return P_tsurff
end


function tsurf_plot_energy_spectrum(a_tsurff_lm, momentum_range, pw::physics_world_sh_t; ylimit_min_log::Float64 = -15.0)
    P_tsurff = tsurf_get_energy_spectrum(a_tsurff_lm, momentum_range, pw)
    energy_range = momentum_range .^ 2 ./ 2
    plot(energy_range, P_tsurff,
        yscale=:log10,
        ylimits=(10 ^ ylimit_min_log, 0.0),
        xlabel="Energy of Photoelectron (a.u.)",
        ylabel="Diff. ioniz. possibility (a.u.)",
        labels="τ = 0",
        xlimits=(0,0.25),
        guidefont=Plots.font(14, "Times"),
        tickfont=Plots.font(14, "Times"),
        legendfont=Plots.font(14, "Times"),
        margin = 5 * Plots.mm)
end


function tsurf_plot_xy_momentum_spectrum(a_tsurff_lm, k_space::tsurf_k_space_t, pw::physics_world_sh_t;
    kr_flag::Bool=true, log_flag::Bool=false, min_threhold::Float64=0.0, k_min::Float64=0.0)

    Nk_r = length(k_space.k_r_range)
    Nk_phi = length(k_space.k_phi_range)
    k_r_range = k_space.k_r_range
    k_phi_range = k_space.k_phi_range
    k_theta_fixed = k_space.k_theta_range[1]

    if length(k_space.k_theta_range) != 1
        println("[t-SURFF] (Plot momentum spectrum) Warning! The input k_space is invalid. This function will get wrong result.")
    end

    a_tsurff_mat = zeros(ComplexF64, Nk_r, Nk_phi)
    result_mat = zeros(Float64, Nk_r, Nk_phi)
    for (i, kr) = enumerate(k_r_range)
        for (j, k_phi) in enumerate(k_phi_range)
            k_phi_next = exert_limit.(0, 2pi, -k_phi .- pi/2, 2pi)
            Ylm_vals = computeYlm(k_theta_fixed, k_phi_next, lmax=pw.l_num - 1)
            for id = 1: pw.l_num ^ 2
                l = pw.lmap[id]
                m = pw.mmap[id]
                a_tsurff_mat[i, j] += a_tsurff_lm[id][i] * Ylm_vals[(l, m)]
            end
        end
    end

    for (i, kr) = enumerate(k_r_range)
        for (j, k_phi) in enumerate(k_phi_range)
            result_mat[i, j] = (kr_flag == true ? kr : 1.0) * norm.(a_tsurff_mat[i, j]) .^ 2
            if result_mat[i, j] < min_threhold
                result_mat[i, j] = min_threhold
            end
            if kr < k_min
                result_mat[i, j] = min_threhold
            end
        end
    end

    # temporary
    result_mat .*= 2 * 10 ^ (-4)

    x_min = -0.75
    x_max = 0.75
    x_delta = 0.01
    x_range = x_min: x_delta: x_max
    x_proj_data = zeros(Float64, length(x_range))
    x_proj_times = zeros(Float64, length(x_range))

    # projection
    for (i, kr) = enumerate(k_r_range)
        for (j, k_phi) in enumerate(k_phi_range)
            actual_x = kr * cos(k_phi)
            x_id = Int64((actual_x - x_min) ÷ x_delta) + 1
            if x_id <= 0 || x_id > length(x_range)
                continue
            end
            # println(x_id)
            x_proj_data[x_id] += result_mat[i, j]
            x_proj_times[x_id] += 1
        end
    end

    gr()
    # colormap = cgrad([:white, :black, :red, :blue, :white])
    colormap = cgrad(:jet1, rev = false)
    if log_flag == true
        Plots.heatmap(k_phi_range, k_r_range,
            abs.(log10.(result_mat)); color = colormap,
            projection = :polar,
            right_margin = 8 * Plots.mm)
    else
        Plots.heatmap(k_phi_range, k_r_range, result_mat;
            color = colormap, projection = :polar,
            right_margin = 12 * Plots.mm,
            guidefont=Plots.font(12, "Times"),
            tickfont=Plots.font(12, "Times"),
            legendfont=Plots.font(12, "Times"),
            xlabel="Energy of Photoelectron (a.u.)",
            ylabel="Energy of Photoelectron (a.u.)",
            )
    end
    # return x_proj_data, x_proj_times
    # Plots.heatmap(k_phi_range, k_r_range, result_mat; color = colormap, right_margin = 8 * Plots.mm)
end


# function tsurf_get_average_momentum(a_tsurff_lm, k_space::tsurf_k_space_t, pw::physics_world_sh_t)
    
#     Nk_theta = length(k_space.k_theta_range)
#     Nk_phi = length(k_space.k_phi_range)

#     Ylm_vals_buffer = Matrix{Any}(undef, Nk_theta, Nk_phi)
#     for (j, k_theta) in enumerate(k_space.k_theta_range)
#         for (p, k_phi) in enumerate(k_space.k_phi_range)
#             Ylm_vals_buffer[j, p] = computeYlm(k_theta, k_phi, lmax=pw.l_num - 1)
#         end
#     end

#     expected_kx::Float64 = 0.0
#     expected_ky::Float64 = 0.0
#     expected_kz::Float64 = 0.0
#     P_sum::Float64 = 0.0
#     for (i, kr) in enumerate(k_space.k_r_range)
#         for (j, k_theta) in enumerate(k_space.k_theta_range)
#             for (p, k_phi) in enumerate(k_space.k_phi_range)
#                 tmp_result::ComplexF64 = 0.0
#                 for id = 1: pw.l_num ^ 2
#                     l = pw.lmap[id]
#                     m = pw.mmap[id]
#                     tmp_result += a_tsurff_lm[id][i] * Ylm_vals_buffer[j, p][(l, m)]
#                 end
#                 kx, ky, kz = sphere_to_xyz(kr, k_theta, k_phi)
#                 P_result::Float64 = norm(tmp_result) ^ 2
#                 expected_kx += P_result * kx
#                 expected_ky += P_result * ky
#                 expected_kz += P_result * kz
#                 P_sum += P_result
#             end
#         end
#     end
    
#     expected_kx /= P_sum
#     expected_ky /= P_sum
#     expected_kz /= P_sum
#     return expected_kx, expected_ky, expected_kz
# end


function tsurf_get_average_momentum_single_kr(a_tsurff_lm, k_space::tsurf_k_space_t, pw::physics_world_sh_t, Ylm_vals_buffer, kr_id_list, k_min)

    expected_kx::Float64 = 0.0
    expected_ky::Float64 = 0.0
    expected_kz::Float64 = 0.0
    P_sum::Float64 = 0.0
    for i in kr_id_list
        kr = k_space.k_r_range[i]
        if kr < k_min
            continue
        end
        for (j, k_theta) in enumerate(k_space.k_theta_range)
            for (p, k_phi) in enumerate(k_space.k_phi_range)
                tmp_result::ComplexF64 = 0.0
                for id = 1: pw.l_num ^ 2
                    l = pw.lmap[id]
                    m = pw.mmap[id]
                    tmp_result += a_tsurff_lm[id][i] * Ylm_vals_buffer[j, p][(l, m)]
                end
                kx, ky, kz = sphere_to_xyz(kr, k_theta, k_phi)
                P_result::Float64 = norm(tmp_result) ^ 2
                expected_kx += P_result * kx
                expected_ky += P_result * ky
                expected_kz += P_result * kz
                P_sum += P_result
            end
        end
    end
    return (expected_kx, expected_ky, expected_kz, P_sum)
end


function tsurf_get_average_momentum_parallel(a_tsurff_lm, k_space::tsurf_k_space_t, pw::physics_world_sh_t; k_min::Float64=0.01)
    
    Nk_theta = length(k_space.k_theta_range)
    Nk_phi = length(k_space.k_phi_range)

    Ylm_vals_buffer = Matrix{Any}(undef, Nk_theta, Nk_phi)
    for (j, k_theta) in enumerate(k_space.k_theta_range)
        for (p, k_phi) in enumerate(k_space.k_phi_range)
            Ylm_vals_buffer[j, p] = computeYlm(k_theta, k_phi, lmax=pw.l_num - 1)
        end
    end

    expected_kx = 0.0
    expected_ky = 0.0
    expected_kz = 0.0
    P_sum = 0.0

    kr_id_list_total = eachindex(k_space.k_r_range)
    kr_list_chunks = Iterators.partition(kr_id_list_total, length(kr_id_list_total) ÷ Threads.nthreads())
    tasks = map(kr_list_chunks) do chunk
        Threads.@spawn tsurf_get_average_momentum_single_kr(a_tsurff_lm, k_space, pw, Ylm_vals_buffer, chunk, k_min)
    end
    chunk_results = fetch.(tasks)
    
    for res in chunk_results
        expected_kx += res[1]
        expected_ky += res[2]
        expected_kz += res[3]
        P_sum += res[4]
    end

    expected_kx /= P_sum
    expected_ky /= P_sum
    expected_kz /= P_sum
    return expected_kx, expected_ky, expected_kz
end



# ================================

fixed_r(r) = [r]
fixed_theta(theta) = [theta]
fixed_phi(phi) = [phi]

# const ANGLE_START_MARGIN = 0.001    # avoid NaN
const ANGLE_START_MARGIN = 0.0
theta_linspace(N_theta) = ANGLE_START_MARGIN: pi / N_theta: pi
phi_linspace(N_phi) = ANGLE_START_MARGIN: 2 * pi / N_phi: 2 * pi

function create_k_space(k_r_range_input, k_theta_range_input, k_phi_range_input)
    k_r_range = [value for value in k_r_range_input]
    k_theta_range = [value for value in k_theta_range_input]
    k_phi_range = [value for value in k_phi_range_input]
    
    Nk_r = length(k_r_range)
    Nk_theta = length(k_theta_range)
    Nk_phi = length(k_phi_range)
    k_collection = [(0.0, 0.0, 0.0) for _ = 1: Nk_r * Nk_theta * Nk_phi]
    ijk_mapping = [(0, 0, 0) for _ = 1: Nk_r * Nk_theta * Nk_phi]

    p = 1
    for (i, kr) in enumerate(k_r_range)   # order: r, θ, ϕ
        for (j, k_theta) in enumerate(k_theta_range)
            for (k, k_phi) in enumerate(k_phi_range)
                kx, ky, kz = sphere_to_xyz(kr, k_theta, k_phi)
                k_collection[p] = (kx, ky, kz)
                ijk_mapping[p] = (i, j, k)
                p += 1
            end
        end
    end

    return tsurf_k_space_t(k_r_range, k_theta_range, k_phi_range, k_collection, ijk_mapping)
end


function tsurf_combine_lm_vec(a_tsurff_lm_vec, k_space::tsurf_k_space_t, pw::physics_world_sh_t)
    a_tsurff_vec = zeros(ComplexF64, length(k_space.k_collection))

    for (p, k_vec) in enumerate(k_space.k_collection)
        kr, k_theta, k_phi = xyz_to_sphere(k_vec...)
        Ylm_vals = computeYlm(k_theta, k_phi, lmax=pw.l_num - 1)
        for id = 1: pw.l_num ^ 2
            l = pw.lmap[id]
            m = pw.mmap[id]
            a_tsurff_vec[p] += a_tsurff_lm_vec[id][p] * Ylm_vals[(l, m)]
        end
    end
    return a_tsurff_vec
end


function tsurf_plot_xy_momentum_spectrum_vector(a_tsurff_vec, k_space::tsurf_k_space_t; kr_flag::Bool=false, log_flag::Bool=false, kr_min::Float64=0.0)

    Nk_r = length(k_space.k_r_range)
    Nk_phi = length(k_space.k_phi_range)
    k_r_range = k_space.k_r_range
    k_phi_range = k_space.k_phi_range

    pes_mat = zeros(Float64, Nk_r, Nk_phi)
    for (p, k_vec) in enumerate(k_space.k_collection)
        i, _, k = k_space.ijk_mapping[p]
        pes_mat[i, k] = (kr_flag == true ? norm(k_vec) : 1.0) * norm(a_tsurff_vec[p]) ^ 2 * (norm(k_vec) > kr_min)
    end

    gr()
    # colormap = cgrad([:white, :black, :red, :blue, :white])
    colormap = cgrad(:hot, rev=true)
    if log_flag == true
        Plots.heatmap(k_phi_range, k_r_range, abs.(log10.(pes_mat)); color = colormap, projection = :polar, right_margin = 10 * Plots.mm)
    else
        Plots.heatmap(k_phi_range, k_r_range, pes_mat; color = colormap, projection = :polar,
            right_margin = 12 * Plots.mm,
            guidefont=Plots.font(12, "Times"),
            tickfont=Plots.font(12, "Times"),
            legendfont=Plots.font(12, "Times"))
    end
end


function tsurf_plot_xz_momentum_spectrum_vector(a_tsurff_vec, k_space::tsurf_k_space_t; kr_flag::Bool = false, log_flag::Bool = false, min_threhold::Float64=0.0, kr_min::Float64=0.0)

    Nk_r = length(k_space.k_r_range)
    Nk_theta = length(k_space.k_theta_range)
    k_r_range = k_space.k_r_range
    k_theta_range = k_space.k_theta_range

    pes_mat = zeros(Float64, Nk_r, Nk_theta)
    for (p, k_vec) in enumerate(k_space.k_collection)
        i, j, _ = k_space.ijk_mapping[p]
        pes_mat[i, j] = (kr_flag == true ? norm(k_vec) : 1.0) * norm(a_tsurff_vec[p]) ^ 2 * (norm(k_vec) > kr_min)
        if pes_mat[i, j] < min_threhold
            pes_mat[i, j] = min_threhold
        end
    end

    gr()
    colormap = cgrad([:white, :black, :red, :blue, :white])
    # colormap = cgrad(:hot, rev = true)
    if log_flag == true
        colormap = cgrad([:white, :black, :red, :blue, :white], rev=false)
        fig = Plots.heatmap(k_theta_range, k_r_range, abs.(log10.(pes_mat)); color = colormap, projection = :polar, right_margin = 8 * Plots.mm)
        Plots.heatmap!(fig, k_theta_range .- pi, k_r_range, abs.(log10.(pes_mat)); color = colormap, projection = :polar, right_margin = 8 * Plots.mm)
        return fig
    else
        pes_mat_rev = copy(pes_mat)       # reverse the columns of pes_mat
        for i = 1: size(pes_mat)[1]
            for j = 1: size(pes_mat)[2]
                pes_mat_rev[i, j] = pes_mat[i, size(pes_mat)[2] - j + 1]
            end
        end
        fig = Plots.heatmap(k_theta_range, k_r_range, pes_mat; color = colormap, projection = :polar, right_margin = 8 * Plots.mm)
        Plots.heatmap!(fig, k_theta_range .- pi, k_r_range, pes_mat; color = colormap, projection = :polar, right_margin = 8 * Plots.mm)
        return fig
    end
end

function tsurf_get_average_momentum_vector_parallel_single(a_tsurff_vec, k_space::tsurf_k_space_t, k_col_id_list, k_min)
    expected_kx = 0.0
    expected_ky = 0.0
    expected_kz = 0.0
    P_sum = 0.0
    for p in k_col_id_list
        k_vec = k_space.k_collection[p]
        kr, _, _ = xyz_to_sphere(k_vec...)
        if kr < k_min
            continue
        end
        P_sum += norm(a_tsurff_vec[p]) ^ 2
        expected_kx += norm(a_tsurff_vec[p]) ^ 2 * k_vec[1]
        expected_ky += norm(a_tsurff_vec[p]) ^ 2 * k_vec[2]
        expected_kz += norm(a_tsurff_vec[p]) ^ 2 * k_vec[3]
    end
    return (expected_kx, expected_ky, expected_kz, P_sum)
end


function tsurf_get_average_momentum_vector_parallel(a_tsurff_vec, k_space::tsurf_k_space_t; k_min::Float64 = 0.0)
    expected_kx = 0.0
    expected_ky = 0.0
    expected_kz = 0.0
    P_sum = 0.0
    
    k_col_id_list = eachindex(k_space.k_collection)
    k_col_chunks = Iterators.partition(k_col_id_list, length(k_col_id_list) ÷ Threads.nthreads())
    tasks = map(k_col_chunks) do chunk
        Threads.@spawn tsurf_get_average_momentum_vector_parallel_single(a_tsurff_vec, k_space, chunk, k_min)
    end
    chunk_results = fetch.(tasks)
    
    for res in chunk_results
        expected_kx += res[1]
        expected_ky += res[2]
        expected_kz += res[3]
        P_sum += res[4]
    end

    expected_kx /= P_sum
    expected_ky /= P_sum
    expected_kz /= P_sum
    return expected_kx, expected_ky, expected_kz
end


function plot_line_order(arr, k_space::tsurf_k_space_t)

    Nk_r = length(k_space.k_r_range)
    Nk_phi = length(k_space.k_phi_range)
    k_r_range = k_space.k_r_range
    k_phi_range = k_space.k_phi_range

    result_mat = zeros(eltype(arr), Nk_r, Nk_phi)
    for (p, k_vec) in enumerate(k_space.k_collection)
        i, _, k = k_space.ijk_mapping[p]
        result_mat[i, k] = arr[p]
    end

    gr()
    # colormap = cgrad([:white, :black, :red, :blue, :white])
    colormap = cgrad(:hot, rev = true)
    Plots.heatmap(k_phi_range, k_r_range, result_mat; color = colormap, projection = :polar, right_margin = 8 * Plots.mm)
end




###################

# isurf

function isurf_rest_part(crt_shwave::shwave_t, k_linspace, tau_p, Ri_tsurf, pw::physics_world_sh_t, rt::tdse_sh_rt)
    
    δa_lm = [zeros(ComplexF64, length(k_linspace)) for i = 1: pw.l_num ^ 2]

    # prepare buffer
    sh_bessel_buffer = [zeros(Float64, length(k_linspace)) for l = 0: pw.l_num]
    for l1 = 0: pw.l_num    # important: l1 must be 0 ~ l_max + 1, because sh_bessel_buffer[l1 + 1] will be used.
        for i in eachindex(k_linspace)
            k = k_linspace[i]
            sh_bessel_buffer[l1 + 1][i] = spherical_bessel_func(l1, k * Ri_tsurf)
        end
    end

    for i in eachindex(k_linspace)
        k = k_linspace[i]
        Ek = k ^ 2 / 2
        println("[i-SURFF] i = $i, k = $k")

        copy_shwave(rt.phi, crt_shwave)
        Threads.@threads for id = 1: pw.l_num ^ 2
            l = rt.lmap[id]
            m = rt.mmap[id]

            if l == abs(m)
                @. rt.Htmp_list[id] = rt.M2_boost * Ek - rt.Hl_right_list_im_boost[l + 1]
                mul!(rt.phi_tmp[id], rt.M2_boost, rt.phi[id])
                rt.phi[id] .= rt.phi_tmp[id]
            else
                @. rt.Htmp_list[id] = rt.M2 * Ek - rt.Hl_right_list_im[l + 1]
                mul!(rt.phi_tmp[id], rt.M2, rt.phi[id])
                rt.phi[id] .= rt.phi_tmp[id]
            end

            trimat_elimination(rt.phi_tmp[id], rt.Htmp_list[id], rt.phi[id], rt.A_add_list[id], rt.B_add_list[id])
            rt.phi[id] .= rt.phi_tmp[id]

            rt.phi[id] .*= -exp(im * Ek * tau_p)
            
            R_id = grid_reduce(pw.shgrid.rgrid, Ri_tsurf)
            phi_r = rt.phi[id][R_id]
            dphi_r = four_order_difference(rt.phi[id], R_id, pw.delta_r)

            a::Float64 = (-k * Ri_tsurf * sh_bessel_buffer[l+2][i] + (l - 1.0) * sh_bessel_buffer[l+1][i])
            b::Float64 = (sh_bessel_buffer[l+1][i] * Ri_tsurf)
            δa_lm[id][i] = (a * phi_r + b * dphi_r) * ((im) ^ l / sqrt(2 * pi))
        end
    end

    return δa_lm
end


function isurf_sh(pw::physics_world_sh_t, rt::tdse_sh_rt, phi_record, dphi_record, shwave_tau, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_linspace, tsurf_mode)
    tau_p = last(t_linspace)
    a_tsurff = tsurf_sh(pw, phi_record, dphi_record, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_linspace, tsurf_mode)
    δa_lm = isurf_rest_part(shwave_tau, k_linspace, tau_p, Ri_tsurf, pw, rt)

    # add them together
    for id = 1: pw.l_num ^ 2
        for i in eachindex(k_linspace)
            a_tsurff[id][i] += δa_lm[id][i]
        end
    end

    return a_tsurff
end


function isurf_sh_vector(pw::physics_world_sh_t, rt::tdse_sh_rt, phi_record, dphi_record, shwave_tau, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space::tsurf_k_space_t, tsurf_mode)

    tau_p = last(t_linspace)
    a_tsurff_lm_vec = tsurf_sh_vector(pw, phi_record, dphi_record, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_space.k_collection, tsurf_mode)
    δa_lm = isurf_rest_part(shwave_tau, k_space.k_r_range, tau_p, Ri_tsurf, pw, rt)
    
    δa_vec = zeros(ComplexF64, length(k_space.k_collection))
    a_tsurff_vec = zeros(ComplexF64, length(k_space.k_collection))

    Nk_r = length(k_space.k_r_range)
    Nk_theta = length(k_space.k_theta_range)
    Nk_phi = length(k_space.k_phi_range)

    # evaluate buffer of Ylm
    Ylm_buffer = Matrix{Any}(undef, Nk_theta, Nk_phi)
    for (j, theta) in enumerate(k_space.k_theta_range)
        for (k, phi) in enumerate(k_space.k_phi_range)
            Ylm_buffer[j, k] = computeYlm(theta, phi, lmax=pw.l_num - 1)
        end
    end

    # get δa_vec from δa_lm
    for (p, k_vec) in enumerate(k_space.k_collection)
        i, j, k = k_space.ijk_mapping[p]    # kr, kθ, kϕ
        for id = 1: pw.l_num ^ 2
            l = pw.lmap[id]
            m = pw.mmap[id]
            δa_vec[p] += δa_lm[id][i] * Ylm_buffer[j, k][(l, m)]
            a_tsurff_vec[p] += a_tsurff_lm_vec[id][p] * Ylm_buffer[j, k][(l, m)]
        end
    end

    tsurf_result_vec = deepcopy(a_tsurff_vec)

    # add them together
    for p in eachindex(k_space.k_collection)
        a_tsurff_vec[p] += δa_vec[p]
    end

    return a_tsurff_vec
end