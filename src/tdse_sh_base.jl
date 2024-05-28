
@kwdef struct physics_world_sh_t
    Nr::Int64
    delta_r::Float64
    shgrid::GridSH

    delta_t::Float64
    itp_delta_t::ComplexF64
    
    Z::Float64
    po_data_r::Vector{Float64}
    po_data_r_im::Vector{ComplexF64}

    l_num::Int64
    lmap::Vector{Int64}
    mmap::Vector{Int64}
end

mutable struct tdse_sh_rt
    # l, m map, and par_strategy for elli
    lmap::Vector{Int64}
    mmap::Vector{Int64}
    par_strategy::Vector{Vector{Int64}}

    # basic matrices
    D2::SymTridiagonal{Float64, Vector{Float64}}
    M2::SymTridiagonal{Float64, Vector{Float64}}
    D1::Tridiagonal{Float64, Vector{Float64}}   # Note: D1, M1 have been modified !!
    M1::Tridiagonal{Float64, Vector{Float64}}
    D2_boost::SymTridiagonal{Float64, Vector{Float64}}
    M2_boost::SymTridiagonal{Float64, Vector{Float64}}
    M2_lu::LDLt{Float64, SymTridiagonal{Float64, Vector{Float64}}}
    M2_boost_lu::LDLt{Float64, SymTridiagonal{Float64, Vector{Float64}}}

    # Hamitonian
    Hl_right_list::Vector{Tridiagonal{Float64, Vector{Float64}}}
    Hl_right_list_boost::Vector{Tridiagonal{Float64, Vector{Float64}}}
    Hl_right_list_im::Vector{Tridiagonal{ComplexF64, Vector{ComplexF64}}}
    Hl_right_list_im_boost::Vector{Tridiagonal{ComplexF64, Vector{ComplexF64}}}

    # medium matrices
    W_pos::Vector{Tridiagonal{ComplexF64, Vector{ComplexF64}}}
    W_neg::Vector{Tridiagonal{ComplexF64, Vector{ComplexF64}}}
    W_pos_boost::Vector{Tridiagonal{ComplexF64, Vector{ComplexF64}}}
    W_neg_boost::Vector{Tridiagonal{ComplexF64, Vector{ComplexF64}}}

    W_pos_im::Vector{Tridiagonal{ComplexF64, Vector{ComplexF64}}}
    W_neg_im::Vector{Tridiagonal{ComplexF64, Vector{ComplexF64}}}
    W_pos_boost_im::Vector{Tridiagonal{ComplexF64, Vector{ComplexF64}}}
    W_neg_boost_im::Vector{Tridiagonal{ComplexF64, Vector{ComplexF64}}}

    B_pl::Matrix{Float64}
    B_elli::Matrix{ComplexF64}
    B_tilde_elli::Matrix{ComplexF64}

    # temp matrices
    Ylm_pos::Vector{Tridiagonal{Float64, Vector{Float64}}}
    Ylm_neg::Vector{Tridiagonal{Float64, Vector{Float64}}}
    Ylm_tilde_pos::Vector{Tridiagonal{Float64, Vector{Float64}}}
    Ylm_tilde_neg::Vector{Tridiagonal{Float64, Vector{Float64}}}
    Htmp_list::Vector{Tridiagonal{ComplexF64, Vector{ComplexF64}}}

    # temp shwave
    tmp_shwave::shwave_t
    tmp_shwave1::shwave_t
    tmp_shwave2::shwave_t
    phi::shwave_t
    phi_tmp::shwave_t

    # buffer 
    A_add_list_scalar::Vector{Vector{Float64}}
    A_add_list::Vector{Vector{ComplexF64}}
    B_add_list::Vector{Vector{ComplexF64}}
end


function get_energy_sh_mbunch(shwave::shwave_t, en_rt::tdse_sh_rt, shgrid, m)
    bundle_size::Int64 = shgrid.l_num - abs(m)
    head_ptr::Int64 = get_index_from_lm(abs(m), m, shgrid.l_num)
    en_list = zeros(Float64, bundle_size)
    en::Float64 = 0

    Threads.@threads for j in 1: bundle_size
        l::Int64 = abs(m) + j - 1
        id::Int64 = head_ptr + j - 1
        if j == 1
            mul!(en_rt.tmp_shwave1[id], en_rt.Hl_right_list_boost[l+1], shwave[id])
            ldiv!(en_rt.tmp_shwave2[id], en_rt.M2_boost_lu, en_rt.tmp_shwave1[id])
            en_list[j] = real(dot(shwave[id], en_rt.tmp_shwave2[id]))
        else
            mul!(en_rt.tmp_shwave1[id], en_rt.Hl_right_list[l+1], shwave[id])
            ldiv!(en_rt.tmp_shwave2[id], en_rt.M2_lu, en_rt.tmp_shwave1[id])
            en_list[j] = real(dot(shwave[id], en_rt.tmp_shwave2[id]))
        end
    end
    # reduction
    en = sum(en_list)
    return en
end


function get_energy_sh(shwave::shwave_t, en_rt::tdse_sh_rt, shgrid)
    l_num2 = shgrid.l_num * shgrid.l_num
    en_list = zeros(Float64, l_num2)

    Threads.@threads for id = 1: l_num2
        l = en_rt.lmap[id]
        m = en_rt.mmap[id]
        if abs(m) == l
            mul!(en_rt.tmp_shwave1[id], en_rt.Hl_right_list_boost[l + 1], shwave[id])
            ldiv!(en_rt.tmp_shwave2[id], en_rt.M2_boost_lu, en_rt.tmp_shwave1[id])
            en_list[id] = real(dot(shwave[id], en_rt.tmp_shwave2[id]))
        else
            mul!(en_rt.tmp_shwave1[id], en_rt.Hl_right_list[l + 1], shwave[id])
            ldiv!(en_rt.tmp_shwave2[id], en_rt.M2_lu, en_rt.tmp_shwave1[id])
            en_list[id] = real(dot(shwave[id], en_rt.tmp_shwave2[id]))
        end
    end
    # reduction
    return sum(en_list)
end


function get_energy_sh_so(shwave::shwave_t, en_rt::tdse_sh_rt, id)
    
    l = en_rt.lmap[id]
    m = en_rt.mmap[id]
    en::Float64 = 0

    if abs(m) == l
        mul!(en_rt.tmp_shwave1[id], en_rt.Hl_right_list_boost[l + 1], shwave[id])
        ldiv!(en_rt.tmp_shwave2[id], en_rt.M2_boost_lu, en_rt.tmp_shwave1[id])
        en = real(dot(shwave[id], en_rt.tmp_shwave2[id]))
    else
        mul!(en_rt.tmp_shwave1[id], en_rt.Hl_right_list[l + 1], shwave[id])
        ldiv!(en_rt.tmp_shwave2[id], en_rt.M2_lu, en_rt.tmp_shwave1[id])
        en = real(dot(shwave[id], en_rt.tmp_shwave2[id]))
    end
    return en
end



######## itp part ########

function fdsh_no_laser_one_step_so(crt_shwave, rt::tdse_sh_rt, id)
    # just apply inv(W+)W- only
    l = rt.lmap[id]
    m = rt.mmap[id]
    if l == abs(m)
        mul!(rt.tmp_shwave[id], rt.W_neg_boost[l + 1], crt_shwave[id])
        trimat_elimination(crt_shwave[id], rt.W_pos_boost[l + 1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
    else
        mul!(rt.tmp_shwave[id], rt.W_neg[l + 1], crt_shwave[id])
        trimat_elimination(crt_shwave[id], rt.W_pos[l + 1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
    end
end

function fdsh_no_laser_one_step_so_itp(crt_shwave, rt::tdse_sh_rt, id)
    # just apply inv(W+)W- only
    l = rt.lmap[id]
    m = rt.mmap[id]
    if l == abs(m)
        mul!(rt.tmp_shwave[id], rt.W_neg_boost_im[l + 1], crt_shwave[id])
        trimat_elimination(crt_shwave[id], rt.W_pos_boost_im[l + 1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
    else
        mul!(rt.tmp_shwave[id], rt.W_neg_im[l + 1], crt_shwave[id])
        trimat_elimination(crt_shwave[id], rt.W_pos_im[l + 1], rt.tmp_shwave[id], rt.A_add_list[id], rt.B_add_list[id])
    end
end

function get_k_occupation_list(k, l_num)
    occupation_list = Int64[]
    cnt = 0
    for n = 1: 100
        for l = 0: n - 1
            for m = -l: l
                if cnt >= k
                    return occupation_list
                end
                push!(occupation_list, get_index_from_lm(l, m, l_num))
                cnt += 1
            end
        end
    end
end
# occ_list = get_k_occupation_list(5, l_num)
# map(x -> (lmap[x], mmap[x]), occ_list)

function gram_schmidt_sh_so(wave_list, occ_list)
    self_dot_bf = [dot(wave_list[i][occ_list[i]], wave_list[i][occ_list[i]]) for i in eachindex(wave_list)]
    scalar::ComplexF64 = 0
    for i in eachindex(wave_list)
        for j = 1: i - 1
            if occ_list[j] == occ_list[i]
                scalar = dot(wave_list[j][occ_list[j]], wave_list[i][occ_list[i]]) / self_dot_bf[j]
                wave_list[i][occ_list[i]] -= wave_list[j][occ_list[j]] * scalar
            end
        end
    end
end


function itp_fdsh(pw::physics_world_sh_t, rt::tdse_sh_rt; k = 1, err = 1e-8)
    ori_wave_list = [create_empty_shwave_array_fixed(pw.shgrid) for i = 1: k]
    occ_list = get_k_occupation_list(k, pw.l_num)
    energy_diff = 10000.0     # A big enough Float64
    last_energy = 10000.0

    rs = get_linspace(pw.shgrid.rgrid)
    for i = 1: k
        @. ori_wave_list[i][occ_list[i]] = rs * exp(-rs * i)
    end

    loop_times = 0
    while energy_diff >= err
        for i in eachindex(ori_wave_list)
            fdsh_no_laser_one_step_so_itp(ori_wave_list[i], rt, occ_list[i])
            normalize!(ori_wave_list[i][occ_list[i]])
        end
        gram_schmidt_sh_so(ori_wave_list, occ_list)
        if loop_times % 10 == true
            crt_energy = get_energy_sh_so(ori_wave_list[k], rt, occ_list[k])
            energy_diff = abs(crt_energy - last_energy)
            last_energy = crt_energy
            println("[ITP]: $energy_diff")
        end
        loop_times += 1
    end
    return ori_wave_list
end


function create_physics_world_sh(Nr, l_num, delta_r, delta_t, po_func_r, Z; delta_t_im = delta_t, manually_converge_to_zero_flag::Bool = false, Rco::Float64 = 50.0)
    rgrid = Grid1D(count=Nr, delta=delta_r, shift=delta_r)
    shgrid = GridSH(rgrid=rgrid, l_num=l_num)
    lmap, mmap = create_lmmap(l_num)

    r_linspace = get_linspace(rgrid)
    po_data_r = po_func_r.(r_linspace)
    po_data_r_imb = zeros(ComplexF64, length(po_data_r))

    return physics_world_sh_t(Nr=Nr, delta_r=delta_r, shgrid=shgrid, delta_t=delta_t, itp_delta_t=-im * delta_t_im,
        Z=Z, po_data_r=deepcopy(po_data_r), po_data_r_im=deepcopy(po_data_r .+ po_data_r_imb), l_num=l_num, lmap=lmap, mmap=mmap)
end

function create_physics_world_sh(Nr, l_num, delta_r, delta_t, po_func_r, Z, imb_func; delta_t_im = delta_t, manually_converge_to_zero_flag::Bool = false, Rco::Float64 = 50.0)
    rgrid = Grid1D(count=Nr, delta=delta_r, shift=delta_r)
    shgrid = GridSH(rgrid=rgrid, l_num=l_num)
    lmap, mmap = create_lmmap(l_num)

    r_linspace = get_linspace(rgrid)
    po_data_r = po_func_r.(r_linspace)
    po_data_r_imb = imb_func.(r_linspace)

    return physics_world_sh_t(Nr=Nr, delta_r=delta_r, shgrid=shgrid, delta_t=delta_t, itp_delta_t=-im * delta_t_im,
        Z=Z, po_data_r=deepcopy(po_data_r), po_data_r_im=deepcopy(po_data_r .+ po_data_r_imb), l_num=l_num, lmap=lmap, mmap=mmap)
end


function create_par_strategy(l_num)
    # create iteratering strategy (for non-paralized algorithm)
    # iter_strategy = Int64[]
    # for l = 0: l_num - 2
    #     for m = -l: l
    #         push!(iter_strategy, get_index_from_lm(l, m, l_num))
    #     end
    # end

    # create par strategy (for paralized algorithm)
    par_strategy = [Int64[], Int64[], Int64[]]
    start_point = (0, 0)
    order_map = Dict()
    for i = 0: l_num - 2    # down_move
        for j = 0: l_num - 2    # up_move
            crt_point = map(+, start_point, (1, -1) .* i, (1, 1) .* j)
            if crt_point[1] < l_num - 1
                order = (j + (999 - i) % 3) % 3 + 1
                order_map[crt_point] = order
                push!(par_strategy[order], get_index_from_lm(crt_point[1], crt_point[2], l_num))
            end
        end
    end
    return par_strategy
end

function create_tdse_rt_sh(pw::physics_world_sh_t)
    shgrid = pw.shgrid
    rgrid = pw.shgrid.rgrid
    delta_t = pw.delta_t
    l_num = pw.l_num

    r_linspace = get_linspace(rgrid)
    lmap, mmap = create_lmmap(l_num)
    l_num2 = shgrid.l_num ^ 2

    # mapping potiential to actual data (OPTIONAL: add absorbing boundary)
    V_pure = Diagonal(pw.po_data_r)
    V = Diagonal(pw.po_data_r_im)

    V_apdix = Array{Diagonal{Float64}}(undef, shgrid.l_num)
    for l = 0: shgrid.l_num - 1
        V_apdix[l + 1] = Diagonal(@. (l * (l + 1.0) / 2.0) * (1.0 / (r_linspace ^ 2)))
    end

    # create Tridiagonal matrice, which are in common use in all fd algorithms
    # D2  M2  D2_boost  M2_boost
    D2 = SymTridiagonal(fill(-2, rgrid.count), fill(1, rgrid.count - 1)) * (1.0 / rgrid.delta ^ 2)
    M2 = SymTridiagonal(fill(10, rgrid.count), fill(1, rgrid.count - 1)) * (-1.0 / 6.0)
    D2_boost = deepcopy(D2)
    M2_boost = deepcopy(M2)
    foo = (-2.0 / (rgrid.delta * rgrid.delta)) * (1.0 - pw.Z * rgrid.delta / (12.0 - 10.0 * pw.Z * rgrid.delta))
    bar = -2.0 * (1.0 + rgrid.delta * rgrid.delta * foo / 12.0) 
    D2_boost[1, 1] = foo
    M2_boost[1, 1] = bar


    # D1  M1
    D1 = Tridiagonal(fill(-1, rgrid.count - 1), fill(0, rgrid.count), fill(1, rgrid.count - 1)) * (1.0 / (2.0 * rgrid.delta))
    D1[1, 1] = (sqrt(3.0) - 2.0) * (1.0 / (2.0 * rgrid.delta))
    D1[rgrid.count, rgrid.count] = -D1[1, 1];
    M1 = Tridiagonal(fill(1, rgrid.count - 1), fill(4, rgrid.count), fill(1, rgrid.count - 1)) * (1 / 6)
    M1[1, 1] = (4.0 + (sqrt(3.0) - 2.0)) * (1 / 6)
    M1[rgrid.count, rgrid.count] = M1[1, 1]


    # W_pos/neg(boost)
    W_neg = [(M2 - (D2 + M2 * (V + V_apdix[j])) * (0.5im * delta_t)) for j in 1: shgrid.l_num]
    W_pos = [(M2 + (D2 + M2 * (V + V_apdix[j])) * (0.5im * delta_t)) for j in 1: shgrid.l_num]
    W_neg_boost = [(M2_boost - (D2_boost + M2_boost * (V + V_apdix[j])) * (0.5im * delta_t)) for j in 1: shgrid.l_num]
    W_pos_boost = [(M2_boost + (D2_boost + M2_boost * (V + V_apdix[j])) * (0.5im * delta_t)) for j in 1: shgrid.l_num]
    # for itp
    W_neg_im = [(M2 - (D2 + M2 * (V_pure + V_apdix[j])) * (0.5im * pw.itp_delta_t)) for j in 1: shgrid.l_num]
    W_pos_im = [(M2 + (D2 + M2 * (V_pure + V_apdix[j])) * (0.5im * pw.itp_delta_t)) for j in 1: shgrid.l_num]
    W_neg_boost_im = [(M2_boost - (D2_boost + M2_boost * (V_pure + V_apdix[j])) * (0.5im * pw.itp_delta_t)) for j in 1: shgrid.l_num]
    W_pos_boost_im = [(M2_boost + (D2_boost + M2_boost * (V_pure + V_apdix[j])) * (0.5im * pw.itp_delta_t)) for j in 1: shgrid.l_num]


    # create runtime struct obj
    empty_trimat = Tridiagonal(zeros(rgrid.count - 1), zeros(rgrid.count), zeros(rgrid.count - 1))
    tmp_shwave = create_empty_shwave_array(shgrid)
    Ylm_pos = [deepcopy(empty_trimat) for i in range(1, l_num2)]
    Ylm_neg = [deepcopy(empty_trimat) for i in range(1, l_num2)]
    Ylm_pos_tilde = [deepcopy(empty_trimat) for i in range(1, l_num2)]  # use for elli
    Ylm_neg_tilde = [deepcopy(empty_trimat) for i in range(1, l_num2)]  # use for elli
    A_add_list_scalar = [zeros(Float64, rgrid.count) for i in range(1, l_num2)]
    A_add_list = [zeros(ComplexF64, rgrid.count) for i in range(1, l_num2)]
    B_add_list = [zeros(ComplexF64, rgrid.count) for i in range(1, l_num2)]

    B_pl = [1 1; -1 1] ./ sqrt(2)           # B_pl never changes. (in fdsh_pl)
    B_elli = zeros(ComplexF64, 2, 2)        # B_elli(tilde) will be updated in runtime. (for its η(t) dependency)
    B_tilde_elli = zeros(ComplexF64, 2, 2)

    # create lu object and Hermitians for getting energy. 
    # (length gauge)
    tmp_shwave1 = create_empty_shwave_array(shgrid)
    tmp_shwave2 = create_empty_shwave_array(shgrid)
    M2_lu = factorize(M2)
    M2_boost_lu = factorize(M2_boost)
    Hl_right_list = [(D2 + M2 * (V_pure + V_apdix[j])) for j in 1: shgrid.l_num]
    Hl_right_list_boost = [(D2_boost + M2_boost * (V_pure + V_apdix[j])) for j in 1: shgrid.l_num]
    Hl_right_list_im = [(D2 + M2 * (V + V_apdix[j])) for j in 1: shgrid.l_num]
    Hl_right_list_im_boost = [(D2_boost + M2_boost * (V + V_apdix[j])) for j in 1: shgrid.l_num]

    par_strategy = create_par_strategy(l_num)

    empty_trimat_cplx = Tridiagonal(zeros(ComplexF64, rgrid.count - 1), zeros(ComplexF64, rgrid.count), zeros(ComplexF64, rgrid.count - 1))
    Htmp_list = [deepcopy(empty_trimat_cplx) for i in range(1, l_num2)]
    phi = create_empty_shwave(shgrid)
    phi_tmp = create_empty_shwave(shgrid)

    rt = tdse_sh_rt(lmap, mmap, par_strategy,
        D2, M2, D1, M1, D2_boost, M2_boost, M2_lu, M2_boost_lu,
        Hl_right_list, Hl_right_list_boost,
        Hl_right_list_im, Hl_right_list_im_boost,
        W_pos, W_neg, W_pos_boost, W_neg_boost,
        W_pos_im, W_neg_im, W_pos_boost_im, W_neg_boost_im,
        B_pl, B_elli, B_tilde_elli,
        Ylm_pos, Ylm_neg, Ylm_pos_tilde, Ylm_neg_tilde,
        Htmp_list, tmp_shwave, tmp_shwave1, tmp_shwave2,
        phi, phi_tmp,
        A_add_list_scalar, A_add_list, B_add_list)

    return rt
end


function get_hhg_spectrum_xy(hhg_integral_t, Et_data_x, Et_data_y)
    hhg_xy_t = -hhg_integral_t .- (Et_data_x .+ im .* Et_data_y)
    hhg_len = length(hhg_xy_t)

    # two type of window function
    hhg_window_f(t) = sin(t / (hhg_len / 2) * pi) ^ 2 * (t < hhg_len/2 && t > 0)
    hhg_windows_data = hhg_window_f.(eachindex(hhg_xy_t) .- hhg_len/4)

    # hhg_window_f(t) = sin(t / hhg_len * pi) ^ 2
    # hhg_windows_data = hhg_window_f.(eachindex(hhg_xy_t))
    # plot(hhg_windows_data)

    hhg_spectrum_x = fft(real.(hhg_xy_t) .* hhg_windows_data)
    hhg_spectrum_y = fft(imag.(hhg_xy_t) .* hhg_windows_data)
    hhg_spectrum = norm.(hhg_spectrum_x) .^ 2 + norm.(hhg_spectrum_y) .^ 2

    return hhg_spectrum
end