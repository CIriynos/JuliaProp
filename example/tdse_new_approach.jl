using LinearAlgebra
using Plots
include("util.jl")

# set x-space and time grid
Nx = 32000
delta_x = 0.2
delta_t = 0.05
itp_delta_t = 0.1

# set wave package and potiential
init_wave_func(x, omega_x, k0, x0) = @. (1.0 / (2π) ^ 0.25) * exp(1im * k0 * x) * exp(-((x - x0) / (2 * omega_x)) ^ 2)
# po_func(x) = -1 / sqrt(x * x + 1.0)
function po_func(x)
    a = -1.00004998875156e-08
    b = 4.00029994000813e-06
    c = -0.000400059989501376
    d = 3.99940007502620e-06
    if abs(x) < 100
        return -1 / sqrt(x * x + 1.0)
    elseif abs(x) < 200
        return a * abs(x) ^ 3 + b * abs(x) ^ 2 + c * abs(x) + d
    else
        return 0.0
    end
end
imag_po_func(s) = 0.0
# imag_po_func(s) = -10000im * (((s - 3000 > 0) ? (s - 3000) : ((600 - s) > 0 ? (600 - s) : 0)) / Nx) ^ 3
#################################


# ================= All global vars ===================
# create objects
xgrid = Grid1D(count=Nx, delta=delta_x, shift=-Nx * delta_x / 2)
x_linspace = get_linspace(xgrid)
init_wave = init_wave_func(x_linspace, 1, 5, 0)
normalize!(init_wave)
po_data = po_func.(x_linspace)
po_data_imbound = po_data + imag_po_func.(1: Nx)
plot(x_linspace, imag.(po_data_imbound))


# create basic trimat
# 2-order approx
# D2 = SymTridiagonal(fill(-2, xgrid.count), fill(1, xgrid.count - 1)) * (1 / xgrid.delta ^ 2)
# M2 = SymTridiagonal(I + (xgrid.delta ^ 2 / 12) * D2)
# tmp = Tridiagonal(M2 * diagm(po_data))
# A_pos = M2 - (1im * delta_t / 2) * (-0.5 * D2 + tmp)
# A_neg = M2 + (1im * delta_t / 2) * (-0.5 * D2 + tmp)
# A_neg_lu = factorize(A_neg)
# A_pos = I + 0.25im * delta_t * D2
# A_neg = I - 0.25im * delta_t * D2

# 4-order approx
D2_penta = create_pentamat([-1, 16, -30, 16, -1], xgrid.count) * (1.0 / (12.0 * xgrid.delta ^ 2))
D1_penta = create_pentamat([1, -8, 0, 8, -1], xgrid.count) * (1.0 / (12.0 * xgrid.delta))
I_penta = create_identity_pentamat(xgrid.count)
V_penta = create_diag_pentamat(po_data, xgrid.count)
V_penta_imbound = create_diag_pentamat(po_data_imbound, xgrid.count)
A_pos_penta = I_penta + 0.25im * delta_t * D2_penta - 0.5im * delta_t * V_penta_imbound
A_neg_penta = I_penta - 0.25im * delta_t * D2_penta + 0.5im * delta_t * V_penta_imbound
H_penta = -0.5 * D2_penta + V_penta

# itp version
delta_t_im = -im * itp_delta_t
A_pos_penta_itp = I_penta + 0.25im * delta_t_im * D2_penta - 0.5im * delta_t_im * V_penta
A_neg_penta_itp = I_penta - 0.25im * delta_t_im * D2_penta + 0.5im * delta_t_im * V_penta


# runtime variable
crt_wave = copy(init_wave)
half_wave = copy(init_wave)
tmp_wave = similar(init_wave)
A_add = similar(init_wave)
B_add = similar(init_wave)
A_buffer = similar(D2_penta)
B_buffer = similar(crt_wave)
tmp_ptmat_1 = similar(A_pos_penta)
tmp_ptmat_2 = similar(A_pos_penta)


# ================= functions ===================

function get_energy_1d_penta(crt_wave, H, tmp_wave)
    penta_mul(tmp_wave, H, crt_wave)
    return real(dot(crt_wave, tmp_wave))
end

function fd1d_mainloop_handmade(crt_wave, half_wave, A_pos, A_neg, A_add, B_add, steps)
    for _ = 1: steps
        mul!(half_wave, A_pos, crt_wave)
        trimat_elimination(crt_wave, A_neg, half_wave, A_add, B_add)
    end
end

function fd1d_mainloop_penta(crt_wave, half_wave, A_pos, A_neg, A_buffer, B_buffer, steps)
    for _ = 1: steps
        penta_mul(half_wave, A_pos, crt_wave)
        pentamat_elimination(crt_wave, A_neg, half_wave, A_buffer, B_buffer)
    end
end

function fd1d_mainloop_penta_itp(crt_wave, half_wave, A_pos, A_neg, H, tmp_wave, A_buffer, B_buffer, steps)
    last_energy = 0
    energy_diff = 0
    for _ = 1: steps
        crt_energy = get_energy_1d_penta(crt_wave, H, tmp_wave)
        energy_diff = crt_energy - last_energy
        last_energy = crt_energy
        println("[ITP_1d]: energy_diff = $energy_diff")
        
        penta_mul(half_wave, A_pos, crt_wave)
        pentamat_elimination(crt_wave, A_neg, half_wave, A_buffer, B_buffer)
        normalize!(crt_wave)
    end
    return energy_diff
end

function fd1d_mainloop_laser_penta(crt_wave, half_wave, A_pos, A_neg, D1p, Ats, H_penta, A_pos_new, A_neg_new, A_buffer, B_buffer, steps)
    println("[TDSE_1d]: tdse process starts. 1d with laser in dipole approximation.")
    energy_list = []
    @inbounds @fastmath for i = 1: steps
        @. A_pos_new = A_pos - 0.5 * delta_t * Ats[i] * D1p
        @. A_neg_new = A_neg + 0.5 * delta_t * Ats[i] * D1p

        penta_mul(half_wave, A_pos_new, crt_wave)
        pentamat_elimination(crt_wave, A_neg_new, half_wave, A_buffer, B_buffer)

        if i % 200 == 0
            en = get_energy_1d_penta(crt_wave, H_penta, half_wave)
            push!(energy_list, en)
            println("[FD_1d]: step $i, energy = $en")
        end
    end
    return energy_list
end

function fd1d_mainloop_laser_penta_record(crt_wave, half_wave, A_pos, A_neg, D1p, Ats, H_penta, A_pos_new, A_neg_new, A_buffer, B_buffer, steps, X_pos_id, X_neg_id, delta_x)
    println("[TDSE_1d]: tdse process starts. 1d with laser in dipole approximation. \nThe value at ±X and its derivative will be reserved for t-surff process.")
    X_pos_vals = zeros(ComplexF64, steps)
    X_neg_vals = zeros(ComplexF64, steps)
    X_pos_dvals = zeros(ComplexF64, steps)
    X_neg_dvals = zeros(ComplexF64, steps)
    for i = 1: steps
        @inbounds @fastmath @. A_pos_new = A_pos - 0.5 * delta_t * Ats[i] * D1p
        @inbounds @fastmath @. A_neg_new = A_neg + 0.5 * delta_t * Ats[i] * D1p

        penta_mul(half_wave, A_pos_new, crt_wave)
        pentamat_elimination(crt_wave, A_neg_new, half_wave, A_buffer, B_buffer)

        X_pos_vals[i] = crt_wave[X_pos_id]
        X_neg_vals[i] = crt_wave[X_neg_id]
        X_pos_dvals[i] = four_order_difference(crt_wave, X_pos_id, delta_x)
        X_neg_dvals[i] = four_order_difference(crt_wave, X_neg_id, delta_x)
    end
    println("[TDSE_1d]: tdse process end.")
    return X_pos_vals, X_neg_vals, X_pos_dvals, X_neg_dvals
end

function fd1d_mainloop(crt_wave, half_wave, A_pos, A_neg_lu, steps)
    for _ = 1: steps
        mul!(half_wave, A_pos, crt_wave)
        ldiv!(crt_wave, A_neg_lu, half_wave)
    end
end


function exert_windows_operator_1d_penta(phi, H0_penta, gamma, Ev, n::Int64, I_penta, H_tmp_penta, tmp_wave, A_buffer, B_buffer)
    phi .*= gamma ^ (2 ^ n)
    for k = reverse(1: 2^(n - 1))
        alpha_pos = -Ev + exp(im * (2k - 1) * pi / 2 ^ n) * gamma
        alpha_neg = -Ev - exp(im * (2k - 1) * pi / 2 ^ n) * gamma
        H_tmp_penta .= H0_penta .+ alpha_pos .* I_penta
        pentamat_elimination(tmp_wave, H_tmp_penta, phi, A_buffer, B_buffer)
        phi .= tmp_wave

        H_tmp_penta .= H0_penta .+ alpha_neg .* I_penta
        pentamat_elimination(tmp_wave, H_tmp_penta, phi, A_buffer, B_buffer)
        phi .= tmp_wave
    end
end


# ================= test here ===================

# @time fd1d_mainloop(crt_wave, half_wave, A_pos, A_neg_lu, total_time_steps)
# @time fd1d_mainloop_handmade(crt_wave, half_wave, A_pos, A_neg, A_add, B_add, total_time_steps)
# @time fd1d_mainloop_penta(crt_wave, half_wave, A_pos_penta, A_neg_penta, A_buffer, B_buffer, total_time_steps)


# 1d itp -> get base state
fd1d_mainloop_penta_itp(crt_wave, half_wave, A_pos_penta_itp, A_neg_penta_itp, H_penta, tmp_wave, A_buffer, B_buffer, 100)
init_wave .= crt_wave;



# propagate (tsurff)
# A0 = 1.0
# omega = 0.057
# Nc = 6
# steps = Int64((2 * Nc * pi / omega) ÷ delta_t)
# t_linspace = [delta_t * i for i in range(1, steps)]
# At_data = @. A0 * sin(omega * t_linspace / (2 * Nc)) ^ 2 * sin(omega * t_linspace)

# append!(At_data, zeros(steps))
# steps = steps + steps
# t_linspace = [delta_t * i for i in range(1, steps)]

# X = 200.0
# X_pos_id = grid_reduce(xgrid, X)
# X_neg_id = grid_reduce(xgrid, -X)
# X_pos_vals, X_neg_vals, X_pos_dvals, X_neg_dvals = fd1d_mainloop_laser_penta_record(crt_wave, half_wave, A_pos_penta, A_neg_penta, D1_penta, At_data, H_penta, tmp_ptmat_1, tmp_ptmat_2, A_buffer, B_buffer, steps, X_pos_id, X_neg_id, delta_x)

# tsurff 1d
# energy_delta = 0.01
# energy_linspace = 0: energy_delta: 3.0
# k_linspace = sign.(energy_linspace) .* sqrt.(2 * abs.(energy_linspace))

# b1k = zeros(ComplexF64, length(k_linspace))
# b2k = zeros(ComplexF64, length(k_linspace))

# alpha = similar(At_data)
# for i = eachindex(At_data)
#     if i == 1
#         alpha[i] = At_data[i]
#     else
#         alpha[i] = alpha[i - 1] + At_data[i]
#     end
# end
# alpha .*= delta_t

# τ = t_linspace[lastindex(t_linspace)]
# h(t) = 0.5 * (1 - cos(2 * pi * t / τ))

# for (i, k) in enumerate(k_linspace)
#     for (j, t) in enumerate(t_linspace)
#         b1k[i] += h(t) * (delta_t / sqrt(2 * pi)) * exp(im * t * k ^ 2 / 2) * exp(-im * k * (X - alpha[j])) * ((0.5 * k + At_data[j]) * X_pos_vals[j] - 0.5im * X_pos_dvals[j])
#         b2k[i] += h(t) * (-delta_t / sqrt(2 * pi)) * exp(im * t * k ^ 2 / 2) * exp(-im * k * (-X - alpha[j])) * ((0.5 * k + At_data[j]) * X_neg_vals[j] - 0.5im * X_neg_dvals[j])
#     end
# end
# Pk = norm.(b1k .+ b2k) .^ 2

# plot(energy_linspace, log10.((Pk)), ylimits=(-15, 0))




# propagate (windows operator)
crt_wave .= init_wave
A0 = 1.0
omega = 0.057
Nc = 6
steps = Int64((2 * Nc * pi / omega) ÷ delta_t)
t_linspace = [delta_t * i for i in range(1, steps)]
At_data = @. A0 * sin(omega * t_linspace / (2 * Nc)) ^ 2 * sin(omega * t_linspace)

energy_list = fd1d_mainloop_laser_penta(crt_wave, half_wave, A_pos_penta, A_neg_penta, D1_penta, At_data, H_penta, tmp_ptmat_1, tmp_ptmat_2, A_buffer, B_buffer, steps)
# plot(energy_list)

# get energy spectrum using windows-operator method
gamma_wd = 0.005
n_wd = 3
Emin_wd = 0.0
Emax_wd = 3.0
phi = copy(crt_wave)
Ev_list = Emin_wd: 2 * gamma_wd: Emax_wd
Plist_pos = Float64[]
Plist_neg = Float64[]

zero_id = grid_reduce(xgrid, 0.0)

for Ev in Ev_list
    phi .= crt_wave
    exert_windows_operator_1d_penta(phi, H_penta, gamma_wd, Ev, n_wd, I_penta, tmp_ptmat_1, tmp_wave, A_buffer, B_buffer)
    p_pos = real(dot(phi[zero_id: length(crt_wave)], phi[zero_id: length(crt_wave)]))
    p_neg = real(dot(phi[1: zero_id], phi[1: zero_id]))
    push!(Plist_pos, p_pos)
    push!(Plist_neg, p_neg)
end

Plist = Float64[]
append!(Plist, reverse(Plist_neg))
append!(Plist, Plist_pos[2: length(Plist_pos)])
plot(log10.(Plist), ylimits=(-15, 0))


# Pk_normal = normalize(Pk)
# plot(energy_linspace, [Plist_pos * 100 Pk], yscale=:log10, label=["windows operator" "t-SURFF"], xlabel="Energy (a.u.)", ylabel="Probability", title="1-D Energy Spectrum with Laser")