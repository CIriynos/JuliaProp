
@kwdef struct physics_world_1d_t
    Nx::Int64
    delta_x::Float64
    xgrid::Grid1D

    delta_t::Float64
    itp_delta_t::ComplexF64
    
    po_data::Vector{Float64}
    po_data_im::Vector{ComplexF64}
end

function create_physics_world_1d(Nx, delta_x, delta_t, po_func; delta_t_im=delta_t)
    xgrid = Grid1D(count=Nx, delta=delta_x, shift=-Nx * delta_x / 2)
    x_linspace = get_linspace(xgrid)
    po_data = po_func.(x_linspace)
    pw1d = physics_world_1d_t(Nx, delta_x, xgrid, delta_t, -im * delta_t_im, po_data, copy(po_data .+ 0.0im))
    return pw1d
end

function create_physics_world_1d(Nx, delta_x, delta_t, po_func, imb_func; delta_t_im=delta_t)
    xgrid = Grid1D(count=Nx, delta=delta_x, shift=-Nx * delta_x / 2)
    x_linspace = get_linspace(xgrid)
    po_data = po_func.(x_linspace)
    imb_data = imb_func.(x_linspace)
    pw1d = physics_world_1d_t(Nx, delta_x, xgrid, delta_t, -im * delta_t_im, po_data, copy(po_data .+ imb_data))
    return pw1d
end



# ================= tdse 1d ===================

@kwdef mutable struct tdse_rt_1d_t
    # D2::SymTridiagonal{Float64, Vector{Float64}}
    # M2::SymTridiagonal{Float64, Vector{Float64}}
    # A_pos::Tridiagonal{ComplexF64, Vector{ComplexF64}}
    # A_neg::Tridiagonal{ComplexF64, Vector{ComplexF64}}
    # A_pos_itp::Tridiagonal{ComplexF64, Vector{ComplexF64}}
    # A_neg_itp::Tridiagonal{ComplexF64, Vector{ComplexF64}}

    D2_penta::Matrix{Float64}
    D1_penta::Matrix{Float64}
    I_penta::Matrix{Float64}
    A_pos_penta::Matrix{ComplexF64}
    A_neg_penta::Matrix{ComplexF64}
    H_penta::Matrix{ComplexF64}
    A_pos_penta_itp::Matrix{ComplexF64}
    A_neg_penta_itp::Matrix{ComplexF64}

    # mutable vars
    half_wave::wave_t
    tmp_wave::wave_t
    A_add::Vector{ComplexF64}
    B_add::Vector{ComplexF64}

    A_buffer::pentamat_t
    B_buffer::Vector{ComplexF64}
    tmp_penta_1::pentamat_t
    tmp_penta_2::pentamat_t
end

function create_tdse_rt_1d(pw::physics_world_1d_t)
    # 4-order approx
    D2_penta = create_pentamat([-1, 16, -30, 16, -1], pw.Nx) * (1.0 / (12.0 * pw.delta_x ^ 2))
    D1_penta = create_pentamat([1, -8, 0, 8, -1], pw.Nx) * (1.0 / (12.0 * pw.delta_x))
    I_penta = create_identity_pentamat(pw.Nx)
    V_penta = create_diag_pentamat(pw.po_data, pw.Nx)
    V_penta_im = create_diag_pentamat(pw.po_data_im, pw.Nx)
    A_pos_penta = I_penta + 0.25im * pw.delta_t * D2_penta - 0.5im * pw.delta_t * V_penta_im
    A_neg_penta = I_penta - 0.25im * pw.delta_t * D2_penta + 0.5im * pw.delta_t * V_penta_im
    H_penta = -0.5 * D2_penta + V_penta

    # itp version
    A_pos_penta_itp = I_penta + 0.25im * pw.itp_delta_t * D2_penta - 0.5im * pw.itp_delta_t * V_penta
    A_neg_penta_itp = I_penta - 0.25im * pw.itp_delta_t * D2_penta + 0.5im * pw.itp_delta_t * V_penta

    # mutable vars
    half_wave = zeros(ComplexF64, pw.Nx)
    tmp_wave = zeros(ComplexF64, pw.Nx)
    A_add = zeros(ComplexF64, pw.Nx)
    B_add = zeros(ComplexF64, pw.Nx)
    
    A_buffer = similar(D2_penta)
    B_buffer = zeros(ComplexF64, pw.Nx)
    tmp_penta_1 = similar(A_pos_penta)
    tmp_penta_2 = similar(A_neg_penta)

    # return tdse_rt_1d_t(D2=D2, M2=M2, A_pos=A_pos, A_neg=A_neg, A_pos_itp=A_pos_itp, A_neg_itp=A_neg_itp,
    #     D2_penta=D2_penta, D1_penta=D1_penta, I_penta=I_penta, A_pos_penta=A_pos_penta, A_neg_penta=A_neg_penta,
    #     H_penta=H_penta, A_pos_penta_itp=A_pos_penta_itp, A_neg_penta_itp=A_neg_penta_itp,
    #     half_wave=half_wave, tmp_wave=tmp_wave, A_add=A_add, B_add=B_add, A_buffer=A_buffer, B_buffer=B_buffer,
    #     tmp_penta_1=tmp_penta_1, tmp_penta_2=tmp_penta_2)

    return tdse_rt_1d_t(D2_penta=D2_penta, D1_penta=D1_penta, I_penta=I_penta, A_pos_penta=A_pos_penta, A_neg_penta=A_neg_penta,
        H_penta=H_penta, A_pos_penta_itp=A_pos_penta_itp, A_neg_penta_itp=A_neg_penta_itp,
        half_wave=half_wave, tmp_wave=tmp_wave, A_add=A_add, B_add=B_add, A_buffer=A_buffer, B_buffer=B_buffer,
        tmp_penta_1=tmp_penta_1, tmp_penta_2=tmp_penta_2)
end


# ================= functions ===================

function gauss_package_1d(x_linspace, omega_x, k0, x0)
    return @. (1.0 / (2π) ^ 0.25) * exp(1im * k0 * x_linspace) * exp(-((x_linspace - x0) / (2 * omega_x)) ^ 2)
end

function get_energy_1d(crt_wave::wave_t, rt::tdse_rt_1d_t)
    penta_mul(rt.tmp_wave, rt.H_penta, crt_wave)
    return real(dot(crt_wave, rt.tmp_wave))
end

function get_energy_1d_laser(crt_wave::wave_t, At::Float64, rt::tdse_rt_1d_t)
    tmp_mat = rt.H_penta .- im .* At .* rt.D1_penta #.+ rt.I_penta .* (0.5 * At^2)
    penta_mul(rt.tmp_wave, tmp_mat, crt_wave)
    return real(dot(crt_wave, rt.tmp_wave))
end

function tdse_fd1d_mainloop(crt_wave::wave_t, rt::tdse_rt_1d_t, steps)
    for _ = 1: steps
        mul!(rt.half_wave, rt.A_pos, crt_wave)
        trimat_elimination(crt_wave, rt.A_neg, rt.half_wave, rt.A_add, rt.B_add)
    end
end

function tdse_fd1d_mainloop_penta(crt_wave::wave_t, rt::tdse_rt_1d_t, steps)
    for _ = 1: steps
        penta_mul(rt.half_wave, rt.A_pos_penta, crt_wave)
        pentamat_elimination(crt_wave, rt.A_neg_penta, rt.half_wave, rt.A_buffer, rt.B_buffer)
    end
end

function itp_fd1d(seed_wave::wave_t, rt::tdse_rt_1d_t; min_error::Float64 = 1e-8, max_steps::Int64 = 100000000)
    last_energy = 0.0
    energy_diff = 0.0
    crt_wave = copy(seed_wave)
    for _ = 1: max_steps
        crt_energy = get_energy_1d(crt_wave, rt)
        energy_diff = crt_energy - last_energy
        last_energy = crt_energy
        println("[ITP_1d]: energy_diff = $energy_diff")
        
        penta_mul(rt.half_wave, rt.A_pos_penta_itp, crt_wave)
        pentamat_elimination(crt_wave, rt.A_neg_penta_itp, rt.half_wave, rt.A_buffer, rt.B_buffer)
        normalize!(crt_wave)

        if abs(energy_diff) < min_error
            break
        end
    end
    return crt_wave
end


function tdse_laser_fd1d_mainloop_penta(crt_wave::wave_t, rt::tdse_rt_1d_t, pw::physics_world_1d_t, Ats, steps, X; record_steps::Int64 = 200)
    println("[TDSE_1d]: tdse process starts. 1d with laser in dipole approximation.")
    energy_list = []
    smooth_record = []
    X_pos_vals = zeros(ComplexF64, steps)
    X_neg_vals = zeros(ComplexF64, steps)
    X_pos_dvals = zeros(ComplexF64, steps)
    X_neg_dvals = zeros(ComplexF64, steps)
    X_pos_id = grid_reduce(pw.xgrid, X)
    X_neg_id = grid_reduce(pw.xgrid, -X)

    mid_len = 5
    mask = [!(i in pw.Nx ÷ 2 - mid_len: pw.Nx ÷ 2 + mid_len + 1) for i in 1: pw.Nx]
    dU_dx = get_derivative_two_order(pw.po_data, pw.delta_x) #.* mask
    hhg_integral = zeros(ComplexF64, steps)
    x_linspace = get_linspace(pw.xgrid)

    d_po_func(x) = x * (x^2 + 1) ^ (-1.5)
    dU_dx_2 = d_po_func.(x_linspace)

    @inbounds for i = 1: steps
        @fastmath @. rt.tmp_penta_1 = rt.A_pos_penta - 0.5 * pw.delta_t * Ats[i] * rt.D1_penta
        @fastmath @. rt.tmp_penta_2 = rt.A_neg_penta + 0.5 * pw.delta_t * Ats[i] * rt.D1_penta

        penta_mul(rt.half_wave, rt.tmp_penta_1, crt_wave)
        pentamat_elimination(crt_wave, rt.tmp_penta_2, rt.half_wave, rt.A_buffer, rt.B_buffer)

        X_pos_vals[i] = crt_wave[X_pos_id]
        X_neg_vals[i] = crt_wave[X_neg_id]
        X_pos_dvals[i] = four_order_difference(crt_wave, X_pos_id, pw.delta_x)
        X_neg_dvals[i] = four_order_difference(crt_wave, X_neg_id, pw.delta_x)
        
        # normalize!(crt_wave)
        if i % record_steps == 0
            crt_wave_L = gauge_transform_V2L(crt_wave, Ats[1:i], pw.delta_t, x_linspace)
            en = get_energy_1d(crt_wave_L, rt)
            nm = dot(crt_wave, crt_wave)
            sm = get_smoothness_1(crt_wave, pw.delta_x)
            push!(energy_list, en)
            push!(smooth_record, sm)
            println("[FD_1d]:step $i, energy = $en, norm = $nm")
        end

        # HHG
        # hhg_integral[i] = dot(crt_wave, crt_wave .* dU_dx)
        # hhg_integral[i] = numerical_integral(dU_dx .* norm.(crt_wave) .^ 2, 1.0)
        for k = 1: pw.Nx
            hhg_integral[i] += dU_dx[k] * norm.(crt_wave[k]) .^ 2 * pw.delta_x
        end 
    end
    println("[TDSE_1d]: tdse process end.")
    return [X_pos_vals, X_neg_vals, X_pos_dvals, X_neg_dvals], hhg_integral, energy_list, smooth_record
    # return energy_list
end


function exert_windows_operator_1d_penta(phi::wave_t, gamma::Float64, Ev::Float64, n::Int64, rt::tdse_rt_1d_t)
    phi .*= gamma ^ (2 ^ n)
    @inbounds @fastmath for k = reverse(1: 2 ^ (n - 1))
        alpha_pos = -Ev + exp(im * (2k - 1) * pi / 2 ^ n) * gamma
        alpha_neg = -Ev - exp(im * (2k - 1) * pi / 2 ^ n) * gamma
        rt.tmp_penta_1 .= rt.H_penta .+ alpha_pos .* rt.I_penta
        pentamat_elimination(rt.tmp_wave, rt.tmp_penta_1, phi, rt.A_buffer, rt.B_buffer)
        phi .= rt.tmp_wave

        rt.tmp_penta_2 .= rt.H_penta .+ alpha_neg .* rt.I_penta
        pentamat_elimination(rt.tmp_wave, rt.tmp_penta_2, phi, rt.A_buffer, rt.B_buffer)
        phi .= rt.tmp_wave
    end
end

function windows_operator_method_1d(crt_wave, gamma, n, Ev_list, rt::tdse_rt_1d_t, pw::physics_world_1d_t)
    Plist_pos = zeros(Float64, length(Ev_list))
    Plist_neg = zeros(Float64, length(Ev_list))
    Plist = zeros(Float64, length(Ev_list))
    phi = copy(crt_wave)
    zero_id = grid_reduce(pw.xgrid, 0.0)

    for (i, Ev) in enumerate(Ev_list)
        phi .= crt_wave
        exert_windows_operator_1d_penta(phi, gamma, Ev, n, rt)
        p_pos = real(dot(phi[zero_id: length(crt_wave)], phi[zero_id: length(crt_wave)]))
        p_neg = real(dot(phi[1: zero_id], phi[1: zero_id]))
        p = real(dot(phi, phi))

        Plist_pos[i] = p_pos
        Plist_neg[i] = p_neg
        Plist[i] = p
    end

    Plist_total = [reverse(Plist_neg); Plist_pos[2: length(Plist_pos)]]
    return Plist_total, Plist
end

# function exert_windows_operator_1d_penta_laser(At::Float64, phi::wave_t, gamma::Float64, Ev::Float64, n::Int64, rt::tdse_rt_1d_t)
#     phi .*= gamma ^ (2 ^ n)
#     @inbounds @fastmath for k = reverse(1: 2 ^ (n - 1))
#         alpha_pos = -Ev + exp(im * (2k - 1) * pi / 2 ^ n) * gamma
#         alpha_neg = -Ev - exp(im * (2k - 1) * pi / 2 ^ n) * gamma
#         rt.tmp_penta_1 .= rt.H_penta .- (im * At) .* rt.D1_penta + 0.5 * At^2 * rt.I_penta .+ alpha_pos .* rt.I_penta
#         pentamat_elimination(rt.tmp_wave, rt.tmp_penta_1, phi, rt.A_buffer, rt.B_buffer)
#         phi .= rt.tmp_wave

#         rt.tmp_penta_2 .= rt.H_penta .- (im * At) .* rt.D1_penta + 0.5 * At^2 * rt.I_penta .+ alpha_neg .* rt.I_penta
#         pentamat_elimination(rt.tmp_wave, rt.tmp_penta_2, phi, rt.A_buffer, rt.B_buffer)
#         phi .= rt.tmp_wave
#     end
# end

# function windows_operator_method_1d_laser(crt_wave, At, gamma, n, Ev_list, rt::tdse_rt_1d_t, pw::physics_world_1d_t)
#     Plist_pos = zeros(Float64, length(Ev_list))
#     Plist_neg = zeros(Float64, length(Ev_list))
#     Plist = zeros(Float64, length(Ev_list))
#     phi = copy(crt_wave)
#     zero_id = grid_reduce(pw.xgrid, 0.0)

#     for (i, Ev) in enumerate(Ev_list)
#         phi .= crt_wave
#         exert_windows_operator_1d_penta_laser(At, phi, gamma, Ev, n, rt)
#         p_pos = real(dot(phi[zero_id: length(crt_wave)], phi[zero_id: length(crt_wave)]))
#         p_neg = real(dot(phi[1: zero_id], phi[1: zero_id]))
#         p = real(dot(phi, phi))

#         Plist_pos[i] = p_pos
#         Plist_neg[i] = p_neg
#         Plist[i] = p
#     end

#     Plist_total = [reverse(Plist_neg); Plist_pos[2: length(Plist_pos)]]
#     return Plist_total, Plist
# end


function tsurf_1d(pw::physics_world_1d_t, k_linspace, t_linspace, At_data, X, Xi_data)
    h(t) = 0.5 * (1 - cos(2 * pi * t / last(t_linspace)))

    X_pos_vals = Xi_data[1]
    X_neg_vals = Xi_data[2]
    X_pos_dvals = Xi_data[3]
    X_neg_dvals = Xi_data[4]
    b1k = zeros(ComplexF64, length(k_linspace))
    b2k = zeros(ComplexF64, length(k_linspace))

    alpha = similar(At_data)
    for i = eachindex(At_data)
        if i == 1
            alpha[i] = At_data[i]
        else
            alpha[i] = alpha[i - 1] + At_data[i]
        end
    end
    alpha .*= pw.delta_t

    for (i, k) in enumerate(k_linspace)
        for (j, t) in enumerate(t_linspace)
            b1k[i] += h(t) * (pw.delta_t / sqrt(2 * pi)) * exp(im * t * k ^ 2 / 2) * exp(-im * k * (X - alpha[j])) * ((0.5 * k + At_data[j]) * X_pos_vals[j] - 0.5im * X_pos_dvals[j])
            b2k[i] += h(t) * (-pw.delta_t / sqrt(2 * pi)) * exp(im * t * k ^ 2 / 2) * exp(-im * k * (-X - alpha[j])) * ((0.5 * k + At_data[j]) * X_neg_vals[j] - 0.5im * X_neg_dvals[j])
        end
    end

    Pk = norm.(b1k .+ b2k) .^ 2
    return Pk
end


# gauge transform
function gauge_transform_V2L(wave_V, Ats, delta_t, x_linspace)
    tmp = get_integral(Ats .^ 2, delta_t)
    wave_L = wave_V .* exp.(im * last(Ats) .* x_linspace .- 0.5im * last(tmp))
    return wave_L
end