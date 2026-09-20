import Pkg
Pkg.activate(".")

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW
using SparseArrays
using SpecialFunctions
using Printf
const dot_t = Tuple{Float64, Float64, Float64}

# store the dipole results
_write_complex(h, name::String, x) = begin
    h[name * "/real"] = real.(x)
    h[name * "/imag"] = imag.(x)
end

_read_complex(h, name::String) = read(h[name * "/real"]) .+ im .* read(h[name * "/imag"])


# Basic Parameters
Nr =            10000            # number of radial grid points
Δr =            0.2             # radial grid step size
l_num =         10               # number of angular momentum components
Δt =            0.2            # time step size
Z =             1.0             # nuclear charge
po_func(r) =    -1 / r * exp(- r * r / (10.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
# po_func(r) =    -1 / r * exp(- r * r / (20.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
rmax =          Nr * Δr     
absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function

# define laser field
@expo E_fs =          0.15                    # peak electric field of the fs pulse
@expo ω_fs =          0.057 * 1               # angular frequency of the fs pulse
@expo nc =            6                       # number of optical cycles in the fs pulse
Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, 0, ellipticity=0.0, phase1=0.5pi)        # create the light pulse from the given parameters (+ ellipticity)
E_field(t) = Ex_fs(t)
Tp = 2 * nc * pi / ω_fs

# create pw, rt, and pre-calculated 
pw_ = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func);
rt_ = create_tdse_rt_sh(pw_, m_zero_flag=true);
rs = get_linspace(pw_.shgrid.rgrid)

# get ek_list
max_k = 2
# max_k = 6
ek_list = []
for k_ = 1: max_k
    init_wave = create_empty_mzero_shwave(pw_.shgrid)
    @. init_wave[k_] = rs * exp(-rs * k_)
    itp_fdsh_single(pw_, rt_, init_wave, k_, err=1e-15, log_info=false)
    ek = get_energy_sh_so(init_wave, rt_, k_)

    # ek = -0.5 / (k ^ 2)
    push!(ek_list, ek)
    println("Energy of the state with k_=$k_: ", ek)

    init_wave = nothing 
    GC.gc(true)
    ccall(:malloc_trim, Cint, (Csize_t,), 0)
end
# ek_list = [ -0.4964336949923912, -0.1133886677324868, -0.034607349557600184, -0.00314449576064707]  # ek_list for -1 / r * exp(- r * r / (20.0 ^ 2))
eigen_max_n = length(ek_list)
N_alpha = eigen_max_n * (eigen_max_n + 1) * (2 * eigen_max_n + 1) ÷ 6
alpha_list = Int64[]

# mapping eigen label (n, l) with α (suitable for all eigenstates)
function get_eigen_label(n, l, m=0)
    id = get_index_from_lm(l, m, n)
    return id + (n - 1) * (n) * (2*n - 1) ÷ 6
end

# calculate eigen_states for m = 0 special case
struct eigen_state_t
    n::Int64
    l::Int64
    Ip::Float64
    data::Vector{ComplexF64}
end

eigen_states = Vector{eigen_state_t}(undef, N_alpha)
for (n, ek) in enumerate(ek_list)
    pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func, delta_t_im = 2 / (-ek))
    rt = create_tdse_rt_sh(pw, m_zero_flag=true);
    for l in 0: min(n-1, l_num-1)
        m = 0
        id = get_index_from_lm(l, m, l_num)
        init_wave = create_empty_shwave(pw.shgrid)
        @. init_wave[id] = rs * exp(-rs * n)
        itp_fdsh_single(pw, rt, init_wave, id, log_info=false, mininum_loop_times=2000)
        en = get_energy_sh(init_wave, rt, pw.shgrid)
        alpha = get_eigen_label(n, l, m)
        eigen_states[alpha] = eigen_state_t(n, l, -en, copy(init_wave[id]))
        push!(alpha_list, alpha)
        println("Energy of the state with n=$n, l=$l, m=$m: ", en)
    end
end


# spherical bessel function j_l(x) & spherical harmonic function Y_l
function spherical_besselj_l(l::Int, x::Float64)
    l < 0 && return 0.0
    if abs(x) < 1e-12
        return l == 0 ? 1.0 : 0.0
    end
    return sqrt(pi / (2x)) * besselj(l + 0.5, x)
end

@inline function legendreP_l(l::Int, x::Float64)
    l == 0 && return 1.0
    l == 1 && return x
    p0 = 1.0
    p1 = x
    @inbounds for n in 2:l
        p = ((2n - 1) * x * p1 - (n - 1) * p0) / n
        p0, p1 = p1, p
    end
    return p1
end

@inline function Y_L0(L::Int, costh::Float64)
    return sqrt((2L + 1) / (4π)) * legendreP_l(L, clamp(costh, -1.0, 1.0))
end


# define the pgrid, and create RL_mat buffer for future use
pgrid_pmax = 5.0
Np = 2500 * 4
Δp = pgrid_pmax / Np
N_theta = 180 * 1
Δtheta = π / N_theta
p_grid = [(i - 1 + 0.5) * Δp for i = 1: Np]                 # use mid point grid
theta_grid = [(i - 1 + 0.5) * Δtheta for i = 1: N_theta]    # use mid point grid
RL_left = [zeros(ComplexF64, Np) for i in 1: N_alpha]    # R_nl^{l+1}(p)
RL_right = [zeros(ComplexF64, Np) for i in 1: N_alpha]   # R_nl^{l-1}(p)
RL_fix = [zeros(ComplexF64, Np) for i in 1: N_alpha]   # R^fix_nl^{l}(p)
Y_l0_buffer = [zeros(Float64, N_theta) for i in 1: eigen_max_n + 1]
spherical_besselj_table = [zeros(Float64, Nr) for i in 1: eigen_max_n + 1]
# RI = 20.0
# r_mask = [rs .> RI for i in 1: N_alpha]

############################# (Pre-calculation part)

# println("RL_left/right calculation starts.")
# # calculate the RL_left and RL_right
# for (i, p) in enumerate(p_grid)
#     if i % 100 == 0
#         println("RL_left/right: i = $i / $(length(p_grid))")
#     end
#     for l = 0: eigen_max_n
#         spherical_besselj_table[l + 1] .= spherical_besselj_l.(l, p .* rs)
#     end
#     for α in alpha_list
#         l = eigen_states[α].l
#         for (j, r) in enumerate(rs)
#             RL_left[α][i] += r ^ 2 * eigen_states[α].data[j] * spherical_besselj_table[(l + 1) + 1][j] * sqrt(pw.shgrid.rgrid.delta) #* r_mask[α][j]
#         end
#         if l != 0
#             for (j, r) in enumerate(rs)
#                 RL_right[α][i] += r ^ 2 * eigen_states[α].data[j] * spherical_besselj_table[(l - 1) + 1][j] * sqrt(pw.shgrid.rgrid.delta) #* r_mask[α][j]
#             end
#         end

#         for (j, r) in enumerate(rs)
#             RL_fix[α][i] += r * eigen_states[α].data[j] * spherical_besselj_table[l + 1][j] * sqrt(pw.shgrid.rgrid.delta) #* r_mask[α][j]
#         end
#     end
# end

# println("RL_left/right calculation finished.")

# # pre-calculate d^z_{α1, α2} (m = 0)
# dipole_z_bb = zeros(ComplexF64, N_alpha, N_alpha)
# for α1 in alpha_list
#     for α2 in alpha_list
#         l1 = eigen_states[α1].l
#         l2 = eigen_states[α2].l
#         if abs(l1 - l2) != 1
#             continue
#         end
#         for (j, r) in enumerate(rs)
#             # dipole_z_bb[α1, α2] += eigen_states[α1].data[j] * r * eigen_states[α2].data[j]
#             dipole_z_bb[α1, α2] += conj(eigen_states[α1].data[j]) * r * eigen_states[α2].data[j]
#         end
#         if l1 - l2 == -1
#             dipole_z_bb[α1, α2] *= (l1 + 1) / sqrt((2 * l1 + 1) * (2 * l1 + 3))
#         elseif l1 - l2 == 1
#             dipole_z_bb[α1, α2] *= (l1) / sqrt((2 * l1 - 1) * (2 * l1 + 1))
#         end
#     end
# end
# println("dipole_z_bb calculation finished.")

# # h5open("./data/2025_7_14.h5", "w") do h
# # h5open("./data/2025_7_14_denser.h5", "w") do h
# # h5open("./data/2025_7_14_denser_4times.h5", "w") do h
# h5open("./data/2025_7_14_denser_4times_less_theta.h5", "w") do h
#     _write_complex(h, "RL_left", hcat(RL_left...))
#     _write_complex(h, "RL_right", hcat(RL_right...))
#     _write_complex(h, "RL_fix", hcat(RL_fix...))
#     _write_complex(h, "dipole_z_bb", dipole_z_bb)
# end

#######################

# read the dipole results
data_name = "2025_7_14_denser_4times_less_theta"
RL_left = h5open("./data/$data_name.h5", "r") do h
    RL_left_mat = _read_complex(h, "RL_left")
    [RL_left_mat[:, i] for i in 1: size(RL_left_mat)[2]]
end

RL_right = h5open("./data/$data_name.h5", "r") do h
    RL_right_mat = _read_complex(h, "RL_right")
    [RL_right_mat[:, i] for i in 1: size(RL_right_mat)[2]]
end

RL_fix = h5open("./data/$data_name.h5", "r") do h
    RL_fix_mat = _read_complex(h, "RL_fix")
    [RL_fix_mat[:, i] for i in 1: size(RL_fix_mat)[2]]
end

dipole_z_bb = h5open("./data/$data_name.h5", "r") do h
    _read_complex(h, "dipole_z_bb")
end

# pre-calculate spherical harmonic functions
for l = 0: eigen_max_n
    for (j, theta) in enumerate(theta_grid)
        Y_l0_buffer[l + 1][j] = Y_L0(l, cos(theta))
    end
end

# calculate a coarse-grained d_dipole_cb_coarse
coarse_step = 1
coarse_p_grid = p_grid[1: coarse_step: end]
coarse_theta_grid = theta_grid[1: coarse_step: end]
coarse_dipole_z_cb = [zeros(ComplexF64, length(coarse_p_grid), length(coarse_theta_grid)) for i in 1: N_alpha]
orth_part = [zeros(ComplexF64, length(coarse_p_grid), length(coarse_theta_grid)) for i in 1: N_alpha]
for (i, p) in enumerate(coarse_p_grid)
    for (j, theta) in enumerate(coarse_theta_grid)
        p_id = floor(Int64, (p - 0.0) / Δp) + 1
        theta_id = floor(Int64, (theta - 0.0) / Δtheta) + 1
        # for each \alpha, we get the d^z_{p_j(t), nl} -> dipole_z_cb
        for α in alpha_list
            l = eigen_states[α].l
            C1 = (l + 1) / sqrt((2 * l + 1) * (2 * l + 3))
            dipole_z_cb = sqrt(2 / pi) * (C1 * (-im) ^ (l + 1) * Y_l0_buffer[(l + 1) + 1][theta_id] * RL_left[α][p_id])
            if l != 0
                C2 = (l) / sqrt((2 * l - 1) * (2 * l + 1))
                dipole_z_cb += sqrt(2 / pi) * (C2 * (-im) ^ (l - 1) * Y_l0_buffer[(l - 1) + 1][theta_id] * RL_right[α][p_id])
            end
            coarse_dipole_z_cb[α][i, j] = dipole_z_cb
        end

        # we check the orthogonality of <p|α>
        for α in alpha_list
            l = eigen_states[α].l
            orth_part[α][i, j] = sqrt(2 / pi) * (-im) ^ l * Y_l0_buffer[l + 1][theta_id] * RL_fix[α][p_id]
        end
    end
end

println("coarse_dipole_z_cb calculation finished.")

# α = 1
# destin_mat = copy(orth_part[α])

# res = 0.0
# for i in 1: length(p_grid), j in 1: length(theta_grid)
#     kappa_index = (i - 1) * length(theta_grid) + j
#     p = p_grid[i]
#     theta = theta_grid[j]
#     w = (2 * pi * p^2) * sin(theta) * Δp * Δtheta
#     res += norm.(orth_part[α][i, j]) .^ 2.0 * w
# end

# heatmap([coarse_theta_grid; π .+ coarse_theta_grid], coarse_p_grid,
#     ([norm.(destin_mat) norm.(destin_mat)[:, end:-1:1]]), projection=:polar, color=:cork)

coarse_dipole_z_cb_fixed = deepcopy(coarse_dipole_z_cb)

# correction of dipole_z_cb
for (i, p) in enumerate(coarse_p_grid)
    for (j, theta) in enumerate(coarse_theta_grid)
        p_id = floor(Int64, (p - 0.0) / Δp) + 1
        theta_id = floor(Int64, (theta - 0.0) / Δtheta) + 1
        for α in alpha_list
            for β in alpha_list
                if dipole_z_bb[β, α] == 0
                    continue
                end
                l = eigen_states[β].l
                dipole_z_cb_correction = sqrt(2 / pi) * (-im) ^ l * Y_l0_buffer[l + 1][theta_id] * RL_fix[β][p_id]
                coarse_dipole_z_cb_fixed[α][i, j] -= dipole_z_cb_correction * dipole_z_bb[β, α]
            end
        end
        # @printf "Finish %d, %d\n" i j
    end
end

# # display the coarse dipole matrix
# α = 2
# destin_mat = coarse_dipole_z_cb[α]
# heatmap([coarse_theta_grid; π .+ coarse_theta_grid], coarse_p_grid,
#     ([norm.(destin_mat) norm.(destin_mat)[:, end:-1:1]]), projection=:polar, color=:cork)

    
# # check the dipole_z_bb matrix
# n_list = [e.n for e in eigen_states[alpha_list]]
# l_list = [e.l for e in eigen_states[alpha_list]]
# label_list = ["($(n_list[i]), $(l_list[i]))" for i in 1: length(n_list)]
# heatmap(label_list, label_list, norm.(dipole_z_bb[alpha_list, alpha_list]))


# create kappa grid (Spherical)
# kappa_N_p = Np ÷ 2      # keep the same with the pgrid is OK.
kappa_p_max = 2.0
kappa_p_min = Δp
kappa_delta_p = Δp
kappa_N_p = floor(Int64, (kappa_p_max - kappa_p_min) / kappa_delta_p)
kappa_N_theta = N_theta
kappa_delta_theta = π / kappa_N_theta
kappa_p_subgrid = [kappa_p_min + (i - 0.5) * kappa_delta_p for i in 1: kappa_N_p]
kappa_theta_subgrid = [(q - 0.5) * kappa_delta_theta for q in 1: kappa_N_theta]
N_kappa = length(kappa_p_subgrid) * length(kappa_theta_subgrid)
kappa_grid_p = zeros(Float64, N_kappa)
kappa_grid_theta = zeros(Float64, N_kappa)
kappa_grid_x = zeros(Float64, N_kappa)
kappa_grid_y = zeros(Float64, N_kappa)
kappa_grid_z = zeros(Float64, N_kappa)

kkk = 1
for κ_p in kappa_p_subgrid, κ_theta in kappa_theta_subgrid
    kappa_grid_p[kkk] = κ_p
    kappa_grid_theta[kkk] = κ_theta
    kappa_grid_x[kkk], kappa_grid_y[kkk], kappa_grid_z[kkk] = sphere_to_xyz(κ_p, κ_theta, 0.0)
    global kkk += 1
end


# create time grid and auxiliaries for characteristic curves
Nt = Int64(floor(Tp / Δt))
ts = [i * Δt for i in range(0, Nt - 1)]
Et_data = E_field.(ts)
eta_data = get_integral(Et_data, Δt)
eta_1_data = get_integral(eta_data, Δt)
eta_2_data = get_integral(eta_data .^ 2, Δt)

u_bound = zeros(ComplexF64, N_alpha)
u_free = zeros(ComplexF64, N_kappa)
p_curves_x = zeros(Float64, N_kappa)
p_curves_y = zeros(Float64, N_kappa)
p_curves_z = zeros(Float64, N_kappa)
p_curves_p = zeros(Float64, N_kappa)
p_curves_theta = zeros(Float64, N_kappa)
chi_curves = zeros(Float64, N_kappa)


#############################################################
# SFA calculation

struct sfa_buffer_t
    N_alpha::Int64
    alpha_list::Vector{Int64}
    eigen_states::Vector{eigen_state_t}
    kappa_grid_p::Vector{Float64}
    kappa_grid_theta::Vector{Float64}

    N_kappa::Int64
    kappa_delta_p::Float64
    kappa_delta_theta::Float64    # kappa grid

    p_grid_delta::Float64
    theta_delta::Float64
    Et_data::Vector{Float64}
    eta_data::Vector{Float64}
    eta_1_data::Vector{Float64}
    eta_2_data::Vector{Float64}           # Et_data & its derivative
    coarse_dipole_z_cb::Vector{Matrix{ComplexF64}}
    dipole_z_bb::Matrix{ComplexF64}                                    # precompute results
    p_curves_p::Vector{Float64}
    p_curves_theta::Vector{Float64}
    chi_curves::Vector{Float64}                          # runtime buffer
    chunk_buffer::Vector{ComplexF64}
    chunk_sum_buffer::Vector{ComplexF64}
end

function sfa_rhs(
    u::Vector{ComplexF64},
    res_u::Vector{ComplexF64},    # res: the destination arrays
    t::Float64, it::Int64,        # time
    bf::sfa_buffer_t              # sfa buffer
    )

    t0::Float64 = 0.0
    fill!(res_u, 0.0 + 0.0im)

    # calculate the curves
    @. bf.p_curves_p = sqrt(bf.kappa_grid_p ^ 2 + 2 * bf.eta_data[it] * bf.kappa_grid_p * cos(bf.kappa_grid_theta) + bf.eta_data[it] ^ 2)
    @. bf.p_curves_theta = acos((bf.kappa_grid_p * cos(bf.kappa_grid_theta) + bf.eta_data[it]) / (bf.p_curves_p + 1e-8))
    @. bf.chi_curves = (bf.kappa_grid_p ^ 2) * (t - t0) / 2 + bf.kappa_grid_p * cos(bf.kappa_grid_theta) * bf.eta_1_data[it] + bf.eta_2_data[it] / 2  

    # update part 1
    # @time begin
    # for i in 1: N_kappa

    Threads.@threads for i in 1: bf.N_kappa
        p_id = floor(Int64, (bf.p_curves_p[i] - 0.0) / bf.p_grid_delta) + 1
        theta_id = floor(Int64, (bf.p_curves_theta[i] - 0.0) / bf.theta_delta) + 1
        @inbounds res_u[bf.N_alpha + i] = 0.0
        w = 2 * pi * bf.kappa_grid_p[i] ^ 2 * sin(bf.kappa_grid_theta[i]) * bf.kappa_delta_p * bf.kappa_delta_theta
        for α in bf.alpha_list
            @inbounds res_u[bf.N_alpha + i] += bf.coarse_dipole_z_cb[α][p_id, theta_id] * u[α] * exp(im * bf.eigen_states[α].Ip * t)
        end
        @inbounds res_u[bf.N_alpha + i] *= im * bf.Et_data[it] * exp(im * bf.chi_curves[i]) * sqrt(w)
    end
    # end

    # @time begin
    # update part 3
    # for α in alpha_list

    nt = Threads.nthreads()
    chunk_size = cld(bf.N_kappa, nt)
    for α in bf.alpha_list
        res_u[α] = 0.0
        Threads.@threads for j in 1: nt
            start_idx = (j - 1) * chunk_size + 1
            end_idx = min(j * chunk_size, bf.N_kappa)
            for i in start_idx: end_idx
                p_id = floor(Int64, (bf.p_curves_p[i] - 0.0) / bf.p_grid_delta) + 1
                theta_id = floor(Int64, (bf.p_curves_theta[i] - 0.0) / bf.theta_delta) + 1
                w = 2 * pi * bf.kappa_grid_p[i] ^ 2 * sin(bf.kappa_grid_theta[i]) * bf.kappa_delta_p * bf.kappa_delta_theta
                bf.chunk_buffer[i] = im * bf.Et_data[it] * sqrt(w) * conj(bf.coarse_dipole_z_cb[α][p_id, theta_id]) * exp(-im * bf.eigen_states[α].Ip * t) * u[bf.N_alpha + i] * exp(-im * bf.chi_curves[i])
            end
            bf.chunk_sum_buffer[j] = sum(@views bf.chunk_buffer[start_idx: end_idx])
        end
        res_u[α] = sum(bf.chunk_sum_buffer[1: nt])
    end
    
    for α in bf.alpha_list
        for β in bf.alpha_list
            @inbounds res_u[α] += im * bf.Et_data[it] * bf.dipole_z_bb[α, β] * exp(im * (bf.eigen_states[β].Ip - bf.eigen_states[α].Ip) * t) * u[β]
        end
    end
end


function rk4_one_step(dt_half::Float64,
    u::Vector{ComplexF64},
    k1::Vector{ComplexF64},    # res: the destination arrays
    k2::Vector{ComplexF64},
    k3::Vector{ComplexF64},
    k4::Vector{ComplexF64},
    buffer::Vector{ComplexF64},
    
    t::Float64, it::Int64,
    bf::sfa_buffer_t
    )
    h = 2 * dt_half

    sfa_rhs(u, k1, t, it, bf)
    
    @. buffer = u + (h / 2) * k1
    sfa_rhs(buffer, k2, t + dt_half, it + 1, bf)

    @. buffer = u + (h / 2) * k2
    sfa_rhs(buffer, k3, t + dt_half, it + 1, bf)
    
    @. buffer = u + (h) * k3
    sfa_rhs(buffer, k4, t + h, it + 2, bf)

    @. u += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end



###########################

# init state
u = zeros(ComplexF64, N_alpha + N_kappa)
u_buffer = [zeros(ComplexF64, N_alpha + N_kappa) for i = 1: 5]
u[1] = 1.0

chunk_buffer = zeros(ComplexF64, N_kappa)
chunk_sum_buffer = zeros(ComplexF64, N_kappa)
alpha_list_selected = alpha_list[1: end]

sfa_buffer = sfa_buffer_t(N_alpha, alpha_list_selected,
    eigen_states, kappa_grid_p, kappa_grid_theta, N_kappa, kappa_delta_p, kappa_delta_theta,
    Δp, Δtheta, Et_data, eta_data, eta_1_data, eta_2_data, coarse_dipole_z_cb_fixed, dipole_z_bb, p_curves_p, p_curves_theta,
    chi_curves, chunk_buffer, chunk_sum_buffer)

# RK4 propagation
bf = sfa_buffer
t0 = 0.0
nt = Threads.nthreads()
chunk_size = cld(bf.N_kappa, nt)
mainloop_ts = 1: 2: (length(ts) - 2)

D_BB = zeros(Float64, length(mainloop_ts))
D_BC = zeros(Float64, length(mainloop_ts))
D_CC = zeros(Float64, length(mainloop_ts))
D_total = zeros(Float64, length(mainloop_ts))

P_bound_sum = zeros(Float64, length(mainloop_ts))
P_1s = zeros(Float64, length(mainloop_ts))
P_2s = zeros(Float64, length(mainloop_ts))
P_2p = zeros(Float64, length(mainloop_ts))
P_free_sum = zeros(Float64, length(mainloop_ts))
P_total = zeros(Float64, length(mainloop_ts))

# pre-calculation for dipole
kappa_w = @. 2 * pi * bf.kappa_grid_p ^ 2 * sin(bf.kappa_grid_theta) * bf.kappa_delta_p * bf.kappa_delta_theta
kappa_sqrt_w = sqrt.(kappa_w)
kappa_inv_sqrt_w = 1.0 ./ kappa_sqrt_w
kappa_cos = cos.(bf.kappa_grid_theta)
kappa_sin = sin.(bf.kappa_grid_theta)
kappa_p_cos = bf.kappa_grid_p .* kappa_cos
kappa_sin_over_p = kappa_sin ./ bf.kappa_grid_p

p_id_buffer = zeros(Int64, bf.N_kappa)
theta_id_buffer = zeros(Int64, bf.N_kappa)

kp_r_buffer = zeros(Int64, bf.N_kappa)
km_r_buffer = zeros(Int64, bf.N_kappa)
kp_t_buffer = zeros(Int64, bf.N_kappa)
km_t_buffer = zeros(Int64, bf.N_kappa)
inv_Δκ_buffer = zeros(Float64, bf.N_kappa)
inv_Δθ_buffer = zeros(Float64, bf.N_kappa)

for ip in 1: kappa_N_p
    ip_p = min(ip + 1, kappa_N_p)
    ip_m = max(ip - 1, 1)

    for iq in 1: kappa_N_theta
        iq_p = min(iq + 1, kappa_N_theta)
        iq_m = max(iq - 1, 1)

        k = (ip - 1) * kappa_N_theta + iq

        kp_r_buffer[k] = (ip_p - 1) * kappa_N_theta + iq
        km_r_buffer[k] = (ip_m - 1) * kappa_N_theta + iq
        kp_t_buffer[k] = (ip - 1) * kappa_N_theta + iq_p
        km_t_buffer[k] = (ip - 1) * kappa_N_theta + iq_m

        inv_Δκ_buffer[k] = 1.0 / ((ip_p - ip_m) * bf.kappa_delta_p)
        inv_Δθ_buffer[k] = 1.0 / ((iq_p - iq_m) * bf.kappa_delta_theta)
    end
end

for (i, it) in enumerate(mainloop_ts)
    t = ts[it]

    rk4_one_step(Δt, u, u_buffer[1], u_buffer[2], u_buffer[3], u_buffer[4], u_buffer[5], t, it, bf);

    # after rk4_one_step, u corresponds to it + 2
    it2 = it + 2
    t = ts[it2]

    # calculate the curves; keep your original method
    @. bf.p_curves_p = sqrt(bf.kappa_grid_p ^ 2 + 2 * bf.eta_data[it2] * bf.kappa_grid_p * cos(bf.kappa_grid_theta) + bf.eta_data[it2] ^ 2)
    @. bf.p_curves_theta = acos((bf.kappa_grid_p * cos(bf.kappa_grid_theta) + bf.eta_data[it2]) / (bf.p_curves_p + 1e-8))
    @. bf.chi_curves = (bf.kappa_grid_p ^ 2) * (t - t0) / 2 + bf.kappa_grid_p * cos(bf.kappa_grid_theta) * bf.eta_1_data[it2] + bf.eta_2_data[it2] / 2  

    # pre-calculate p_id, theta_id, and sqrt(W) * C * exp(-iχ)
    Threads.@threads for k in 1: bf.N_kappa
        @inbounds begin
            p_id_buffer[k] = floor(Int64, bf.p_curves_p[k] / bf.p_grid_delta) + 1
            theta_id_buffer[k] = floor(Int64, bf.p_curves_theta[k] / bf.theta_delta) + 1
            bf.chunk_buffer[k] = kappa_sqrt_w[k] * u[bf.N_alpha + k] * cis(-bf.chi_curves[k])
        end
    end

    # # absorption of bound states (only for test)
    # for α in bf.alpha_list
    #     l = eigen_states[α].l
    #     if l == 3
    #         u[α] *= 0.95
    #     elseif l == 4
    #         u[α] *= 0.8
    #     elseif l == 5
    #         u[α] *= 0.5
    #     elseif l == 6
    #         u[α] *= 0.1
    #     end
    # end

    # D_BB
    for α in bf.alpha_list
        for β in bf.alpha_list
            @inbounds D_BB[i] += real(conj(u[α]) * u[β] *
                                      cis((bf.eigen_states[β].Ip - bf.eigen_states[α].Ip) * t) *
                                      bf.dipole_z_bb[α, β])
        end 
    end

    # D_BC
    for α in bf.alpha_list
        bf.chunk_sum_buffer[1: nt] .= 0.0 + 0.0im

        dipole_mat = bf.coarse_dipole_z_cb[α]
        Np_dipole = size(dipole_mat, 1)
        Nθ_dipole = size(dipole_mat, 2)

        Threads.@threads for j in 1: nt
            s = 0.0 + 0.0im
            start_idx = (j - 1) * chunk_size + 1
            end_idx = min(j * chunk_size, bf.N_kappa)

            for k in start_idx: end_idx
                @inbounds begin
                    p_id = p_id_buffer[k]

                    if 1 <= p_id <= Np_dipole
                        theta_id = min(max(theta_id_buffer[k], 1), Nθ_dipole)
                        s += conj(dipole_mat[p_id, theta_id]) * bf.chunk_buffer[k]
                    end
                end
            end

            bf.chunk_sum_buffer[j] = s
        end

        D_BC[i] += 2 * real(conj(u[α]) * cis(-bf.eigen_states[α].Ip * t) * sum(@views bf.chunk_sum_buffer[1: nt]))
    end

    # D_CC
    bf.chunk_sum_buffer[1:nt] .= 0.0 + 0.0im

    Threads.@threads for j in 1:nt
        s = 0.0 + 0.0im
        start_idx = (j - 1) * chunk_size + 1
        end_idx = min(j * chunk_size, bf.N_kappa)

        for k in start_idx:end_idx
            @inbounds begin
                kp_r = kp_r_buffer[k]
                km_r = km_r_buffer[k]
                kp_t = kp_t_buffer[k]
                km_t = km_t_buffer[k]

                Ck   = u[bf.N_alpha + k]    * kappa_inv_sqrt_w[k]
                Cr_p = u[bf.N_alpha + kp_r] * kappa_inv_sqrt_w[kp_r]
                Cr_m = u[bf.N_alpha + km_r] * kappa_inv_sqrt_w[km_r]
                Ct_p = u[bf.N_alpha + kp_t] * kappa_inv_sqrt_w[kp_t]
                Ct_m = u[bf.N_alpha + km_t] * kappa_inv_sqrt_w[km_t]

                dC_dκ = (Cr_p - Cr_m) * inv_Δκ_buffer[k]
                dC_dθ = (Ct_p - Ct_m) * inv_Δθ_buffer[k]

                dC_dκz = kappa_cos[k] * dC_dκ - kappa_sin_over_p[k] * dC_dθ

                dχ_dκz = kappa_p_cos[k] * (t - t0) + bf.eta_1_data[it2]

                s += kappa_w[k] * (im * conj(Ck) * dC_dκz + abs2(Ck) * dχ_dκz)
            end
        end
        bf.chunk_sum_buffer[j] = s
    end
    D_CC[i] = real(sum(@views bf.chunk_sum_buffer[1:nt]))
    
    # get total dipole
    D_total[i] = D_BB[i] + D_BC[i] + D_CC[i]

    # get population
    P_bound_sum[i] = sum(norm.(u[1: N_alpha]) .^ 2)
    P_1s[i] = norm(u[1]) ^ 2
    P_2s[i] = norm(u[2]) ^ 2
    P_2p[i] = norm(u[3]) ^ 2
    P_free_sum[i] = sum(norm.(u[N_alpha + 1: end]) .^ 2)
    P_total[i] = P_bound_sum[i] + P_free_sum[i]

    if it % 100 == 1
        println("it = $it")
        @printf "Total = %0.4f, Free = %0.4f\n"  P_total[i] P_free_sum[i]
        @printf "D_BB = %+0.6e, D_BC = %+0.6e, D_CC = %+0.6e, D_total = %+0.6e\n" D_BB[i] D_BC[i] D_CC[i] D_total[i]
    end
end


# save everything
example_name = "2026_7_14_$(E_fs)_$(ω_fs)_$(nc)_short_range_10"
h5open("./data/$example_name.h5", "w") do h
    write(h, "D_total", D_total)
    write(h, "D_CC", D_CC)
    write(h, "D_BB", D_BB)
    write(h, "D_BC", D_BC)
    write(h, "u", u)
    write(h, "P_free_sum", P_free_sum)
    write(h, "P_bound_sum", P_bound_sum)
    write(h, "P_total", P_total)
end

# example_name = "2026_7_14_$(E_fs)_$(ω_fs)_$(nc)_short_range_10"
# D_total = retrieve_mat(example_name, "D_total")
# D_CC = retrieve_mat(example_name, "D_CC")
# D_BB = retrieve_mat(example_name, "D_BB")
# D_BC = retrieve_mat(example_name, "D_BC")

# p1 = plot(ts[mainloop_ts], [D_total, D_BB, D_BC, D_CC], label=["D_total" "D_BB" "D_BC" "D_CC"])

# p4 = plot(ts[mainloop_ts], P_free_sum, yscale=:log10, ylimits=(1e-8, 1e0))
# plot(ts[mainloop_ts], P_free_sum, ylimits=(1e-8, 1e0))


# # get harmonic spectrum, including data, and k axis (frequency axis)
# # n_cut_off_estim = floor((-en + 3.17 * (E_fs ^ 2.0 / (4.0 * (ω_fs ^ 2.0)))) / ω_fs) * 1
# n_cut_off_estim = 20

# hg1, ks = get_hg_spectrum(ts[mainloop_ts], D_total, ω_fs * (n_cut_off_estim + 20))
# hg1_free, _ = get_hg_spectrum(ts[mainloop_ts], D_CC, ω_fs * (n_cut_off_estim + 20))
# hg1_bound, _ = get_hg_spectrum(ts[mainloop_ts], D_BB, ω_fs * (n_cut_off_estim + 20))
# hg1_cross, _ = get_hg_spectrum(ts[mainloop_ts], D_BC, ω_fs * (n_cut_off_estim + 20))

# # r
# p3 = plot(ks ./ ω_fs, [(ks .^ 3) .* hg1, (ks .^ 3) .* hg1_free, (ks .^ 3) .* hg1_bound, (ks .^ 3) .* hg1_cross],
#     yscale=:log10, xaxis=1:20, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-8, 1e3),
#     label=["Total" "Free" "Bound" "Cross"], xlabel="Harmonic Order", ylabel="HG Intensity", title="HHG Spectrum (E0=$E_fs, ω=$ω_fs)")


#####################

# # get a 2D slice of u_free in x-z plane (ϕ = 0) (Spherical)
# u_free_xz = zeros(ComplexF64, length(kappa_p_subgrid), length(kappa_theta_subgrid))
# for i in 1: length(kappa_p_subgrid), j in 1: length(kappa_theta_subgrid)
#     kappa_index = (i - 1) * length(kappa_theta_subgrid) + j
#     kappa_p = kappa_p_subgrid[i]
#     kappa_theta = kappa_theta_subgrid[j]
#     w = (2 * pi * kappa_p^2) * sin(kappa_theta) * kappa_delta_p * kappa_delta_theta
#     u_free_xz[i, j] = norm.(u[N_alpha + kappa_index]) .^ 2.0 / w
# end
# # colormap = cgrad([:white, :black, :red, :blue, :white])
# colormap = cgrad([:white, palette(:jet1, 6)...], rev = true)
# display_data = clamp.(abs.(log10.(norm.(u_free_xz) ./ maximum(norm.(u_free_xz)))), 0, 3.0)
# # display_data = (display_data ./ 5.0) .^ 0.5 * 5.0
# heatmap([kappa_theta_subgrid; π .+ kappa_theta_subgrid], kappa_p_subgrid, [display_data display_data[:, end:-1:1]], projection=:polar, color=colormap)

# colormap_2 = cgrad(["#710000", "#C60001", "#FB0001", "#FE3A01", "#FF7A00", "#FEAB01", "#FEAB01", "#FEAB01", "#D5FF17", "#91FF60", "#4AFFAE", "#02FFFC", "#00A3FE", "#003DFF", "#2120FF", "#8C8CFE", :white], rev=true)
# p2 = heatmap([kappa_theta_subgrid .+ π/2; kappa_theta_subgrid .+ 3π/2], kappa_p_subgrid[1: end ÷ 2], [norm.(u_free_xz) norm.(u_free_xz)[:, end:-1:1]][1: end ÷ 2, :], projection=:polar, color=colormap_2)



# plot(p_grid, [abs.(RL_right[1]) abs.(RL_right[2]) abs.(RL_right[3])], xlabel="p")