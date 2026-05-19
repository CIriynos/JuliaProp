import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW
using SparseArrays
using SpecialFunctions
const dot_t = Tuple{Float64, Float64, Float64}

# Basic Parameters
Nr =            1000            # number of radial grid points
Δr =            0.1             # radial grid step size
l_num =         5               # number of angular momentum components
Δt =            0.05            # time step size
Z =             1.0             # nuclear charge
# po_func(r) =    -1 / r        # potential function
po_func(r) =    -1 / r * exp(- r * r / (20.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
rmax =          Nr * Δr     
absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function

# define laser field
E_fs =          0.05                # peak electric field of the fs pulse
ω_fs =          0.057 * 1           # angular frequency of the fs pulse
nc =            6                   # number of optical cycles in the fs pulse
Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, 0, ellipticity=0.0, phase1=0.0)        # create the light pulse from the given parameters (+ ellipticity)
E_field(t) = Ex_fs(t)
Tp = 2 * nc * pi / ω_fs

# create pw, rt, and pre-calculated 
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func);
rt = create_tdse_rt_sh(pw, m_zero_flag=true);
rs = get_linspace(pw.shgrid.rgrid)
ek_list = [ -0.4964336949923912, -0.1133886677324868, -0.034607349557600184, -0.00314449576064707]  # ek_list for -1 / r * exp(- r * r / (20.0 ^ 2))
eigen_max_n = length(ek_list)
N_alpha = eigen_max_n * (eigen_max_n + 1) * (2 * eigen_max_n + 1) ÷ 6
alpha_list = []

# mapping eigen label (n, l) with α (suitable for all eigenstates)
function get_eigen_label(n, l, m=0)
    id = get_index_from_lm(l, m, n)
    return id + (n - 1) * (n) * (2*n - 1) ÷ 6
end

# calculate eigen_states for m = 0 special case
eigen_states = Vector{Dict{String, Any}}(undef, N_alpha)
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
        eigen_states[alpha] = Dict("n"=>n, "l"=>l, "Ip"=>(-en), "data"=>copy(init_wave[id]))
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
pgrid_pmax = 2.5
Np = 2000
Δp = pgrid_pmax / Np
N_theta = 180
Δtheta = π / N_theta
p_grid = [(i - 1) * Δp for i = 1: Np]
theta_grid = [(i - 1) * Δtheta for i = 1: N_theta]
RL_left = [zeros(ComplexF64, Np) for i in 1: N_alpha]    # R_nl^{l+1}(p)
RL_right = [zeros(ComplexF64, Np) for i in 1: N_alpha]   # R_nl^{l-1}(p)
Y_l0_buffer = [zeros(Float64, N_theta) for i in 1: eigen_max_n + 1]
spherical_besselj_table = [zeros(Float64, Nr) for i in 1: eigen_max_n + 1]

############################# (Pre-calculation part)

# calculate the RL_left and RL_right
for (i, p) in enumerate(p_grid)
    for l = 0: eigen_max_n
        spherical_besselj_table[l + 1] .= spherical_besselj_l.(l, p .* rs)
    end
    for α in alpha_list
        l = eigen_states[α]["l"]
        for (j, r) in enumerate(rs)
            RL_left[α][i] += r ^ 2 * eigen_states[α]["data"][j] * spherical_besselj_table[(l + 1) + 1][j] * sqrt(pw.shgrid.rgrid.delta)
        end
        if l != 0
            for (j, r) in enumerate(rs)
                RL_right[α][i] += r ^ 2 * eigen_states[α]["data"][j] * spherical_besselj_table[(l - 1) + 1][j] * sqrt(pw.shgrid.rgrid.delta)
            end
        end
    end
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
for (i, p) in enumerate(coarse_p_grid)
    for (j, theta) in enumerate(coarse_theta_grid)
        p_id = floor(Int64, (p - 0.0) / Δp) + 1
        theta_id = floor(Int64, (theta - 0.0) / Δtheta) + 1
        # for each \alpha, we get the d^z_{p_j(t), nl} -> dipole_z_cb
        for α in alpha_list
            l = eigen_states[α]["l"]
            C1 = (l + 1) / sqrt((2 * l + 1) * (2 * l + 3))
            dipole_z_cb = sqrt(2 / pi) * (C1 * (-im) ^ (l + 1) * Y_l0_buffer[(l + 1) + 1][theta_id] * RL_left[α][p_id])
            if l != 0
                C2 = (l) / sqrt((2 * l - 1) * (2 * l + 1))
                dipole_z_cb += sqrt(2 / pi) * (C2 * (-im) ^ (l - 1) * Y_l0_buffer[(l - 1) + 1][theta_id] * RL_right[α][p_id])
            end
            coarse_dipole_z_cb[α][i, j] = dipole_z_cb
        end
    end
end


# display the coarse dipole matrix
destin_mat = coarse_dipole_z_cb[2]
heatmap([coarse_theta_grid; π .+ coarse_theta_grid], coarse_p_grid,
    [norm.(destin_mat) norm.(destin_mat)], projection=:polar)


# pre-calculate d^z_{α1, α2} (m = 0)
dipole_z_bb = zeros(ComplexF64, N_alpha, N_alpha)
for α1 in alpha_list
    for α2 in alpha_list
        l1 = eigen_states[α1]["l"]
        l2 = eigen_states[α2]["l"]
        if abs(l1 - l2) != 1
            continue
        end
        for (j, r) in enumerate(rs)
            dipole_z_bb[α1, α2] += eigen_states[α1]["data"][j] * r * eigen_states[α2]["data"][j]
        end
        if l1 - l2 == -1
            dipole_z_bb[α1, α2] *= (l1 + 1) / sqrt((2 * l1 + 1) * (2 * l1 + 3))
        elseif l1 - l2 == 1
            dipole_z_bb[α1, α2] *= (l1) / sqrt((2 * l1 - 1) * (2 * l1 + 1))
        end
    end
end

# # store the dipole results
_write_complex(h, name::String, x) = begin
    h[name * "/real"] = real.(x)
    h[name * "/imag"] = imag.(x)
end

_read_complex(h, name::String) = read(h[name * "/real"]) .+ im .* read(h[name * "/imag"])

h5open("./data/2025_5_19.h5", "w") do h
    _write_complex(h, "RL_left", hcat(RL_left...))
    _write_complex(h, "RL_right", hcat(RL_right...))
    _write_complex(h, "dipole_z_bb", dipole_z_bb)
end

#######################

# read the dipole results
RL_left = h5open("./data/2025_5_19.h5", "r") do h
    RL_left_mat = _read_complex(h, "RL_left")
    [RL_left_mat[:, i] for i in 1: size(RL_left_mat)[2]]
end

RL_right = h5open("./data/2025_5_19.h5", "r") do h
    RL_right_mat = _read_complex(h, "RL_right")
    [RL_right_mat[:, i] for i in 1: size(RL_right_mat)[2]]
end

# check the dipole_z_bb matrix
n_list = [e["n"] for e in eigen_states[alpha_list]]
l_list = [e["l"] for e in eigen_states[alpha_list]]
label_list = ["($(n_list[i]), $(l_list[i]))" for i in 1: length(n_list)]
heatmap(label_list, label_list, norm.(dipole_z_bb[alpha_list, alpha_list]))


# create kappa grid
kappa_delta = 0.02
kappa_max = 1.0
kappa_min = -1.0
kappa_x_subgrid = kappa_min: kappa_delta: kappa_max
kappa_y_subgrid = -1.0: kappa_delta: 1.0
kappa_z_subgrid = -1.0: kappa_delta: 1.0
N_kappa = length(kappa_x_subgrid) * length(kappa_y_subgrid) * length(kappa_z_subgrid)
kappa_grid_x = zeros(Float64, N_kappa)
kappa_grid_y = zeros(Float64, N_kappa)
kappa_grid_z = zeros(Float64, N_kappa)
k = 1
for κ_x in kappa_x_subgrid, κ_y in kappa_y_subgrid, κ_z in kappa_z_subgrid
    kappa_grid_x[k] = κ_x
    kappa_grid_y[k] = κ_y
    kappa_grid_z[k] = κ_z
    k += 1
end


# create time grid and auxiliaries for characteristic curves
Δt = 0.2
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
chi_curves = zeros(Float64, N_kappa)

plot(cos.(eta_2_data * 0.5))


# pre-judgement
function pre_judgement(
    p_curves_x, p_curves_y, p_curves_z, chi_curves,
    kappa_grid_x, kappa_grid_y, kappa_grid_z, N_kappa,
    Δp, Δtheta, coarse_dipole_z_cb,
    Et_data, eta_data,
    kappa_included_threshold = 1e-3
)
    is_kappa_included = zeros(Int64, N_kappa)
    coupling_amplitude = zeros(ComplexF64, N_kappa) 
    dt = ts[2] - ts[1]

    for (it, t) in enumerate(ts)
        @. p_curves_x = kappa_grid_x
        @. p_curves_y = kappa_grid_y
        @. p_curves_z = kappa_grid_z + eta_data[it]
        @. chi_curves = (kappa_grid_x ^ 2 + kappa_grid_y ^ 2 + kappa_grid_z ^ 2) * (t - t0) / 2 + kappa_grid_z * eta_1_data[it] + eta_2_data[it] / 2
        
        p_abs::Float64 = 0.0
        theta::Float64 = 0.0
        phi::Float64 = 0.0
        p_id::Int64 = 0
        theta_id::Int64 = 0
        for i in 1: N_kappa
            if is_kappa_included[i] == 1
                continue
            end
            # interp to a certain position on (p_abs, θ, ϕ) grid (namely pgrid aforehead)
            p_abs, theta, phi = xyz_to_sphere(p_curves_x[i], p_curves_y[i], p_curves_z[i])
            p_id = floor(Int64, (p_abs - 0.0) / Δp) + 1
            theta_id = floor(Int64, (theta - 0.0) / Δtheta) + 1

            if theta_id == N_theta + 1
                theta_id = N_theta
            end

            α = 1       # let alpha = 1 (because it has the largest dipole-coupling area in p space)
            amp = abs(Et_data[it] * coarse_dipole_z_cb[α][p_id, theta_id])
            coupling_amplitude[i] += dt * im * Et_data[it] * coarse_dipole_z_cb[α][p_id, theta_id] * exp(im * eigen_states[α]["Ip"] * t) * exp(im * chi_curves[i])

            if amp > kappa_included_threshold
                is_kappa_included[i] = 1
            end
        end

        if it % 100 == 0
            println("step $it")
        end
    end

    return is_kappa_included, coupling_amplitude
end

is_kappa_included, coupling_amplitude = pre_judgement(p_curves_x, p_curves_y, p_curves_z, kappa_grid_x, kappa_grid_y, kappa_grid_z, N_kappa, Δp, Δtheta, coarse_dipole_z_cb, Et_data, eta_data)




# on fixing !!
function sfa_rhs(
    u_bound, u_free, res_u_bound, res_u_free,           # res: the destination arrays
    t, it, t0,
    alpha_list, eigen_states,
    kappa_grid_x, kappa_grid_y, kappa_grid_z, N_kappa,  # kappa grid
    kappa_active_list,                                  # maintain an active list, capacity increases only, empty initially
    p_grid_delta, theta_delta,                          # pgrid
    Et_data, eta_data, eta_1_data, eta_2_data,          # Et_data & its derivative
    RL_left, RL_right, dipole_z_bb, Y_l0_buffer,        # precompute results
    p_curves_x, p_curves_y, p_curves_z, chi_curves,     # runtime buffer
    dipole_z_cb_buffer
    )

    # calculate wj
    kappa_grid_x_delta = kappa_grid_x[2] - kappa_grid_x[1]  # we assmue an uniform grid
    kappa_grid_y_delta = kappa_grid_y[2] - kappa_grid_y[1]
    kappa_grid_z_delta = kappa_grid_z[2] - kappa_grid_z[1]
    wj = kappa_grid_x_delta * kappa_grid_y_delta * kappa_grid_z_delta

    # calculate the curves
    @. p_curves_x = kappa_grid_x
    @. p_curves_y = kappa_grid_y
    @. p_curves_z = kappa_grid_z + eta_data[it]
    @. chi_curves = (kappa_grid_x ^ 2 + kappa_grid_y ^ 2 + kappa_grid_z ^ 2) * (t - t0) / 2 + kappa_grid_z * eta_1_data[it] + eta_2_data[it] / 2
    
    for i in 1: N_kappa
        # interp to a certain position on (p_abs, θ, ϕ) grid (namely pgrid aforehead)
        p_abs, theta, phi = xyz_to_sphere(p_curves_x[i], p_curves_y[i], p_curves_z[i])
        p_id = floor(Int64, (p_abs - 0.0) / p_grid_delta) + 1
        theta_id = floor(Int64, (theta - 0.0) / theta_delta) + 1

        # for each \alpha, we get the d^z_{p_j(t), nl} -> dipole_z_cb
        for α in alpha_list
            l = eigen_states[α]["l"]
            C1 = (l + 1) / sqrt((2 * l + 1) * (2 * l + 3))
            C2 = (l) / sqrt((2 * l - 1) * (2 * l + 1))
            dipole_z_cb = sqrt(2 / pi) * (C1 * (-im) ^ (l + 1) * Y_l0_buffer[(l + 1) + 1][theta_id] * RL_left[α][p_id])
            if l != 0
                dipole_z_cb += sqrt(2 / pi) * (C2 * (-im) ^ (l - 1) * Y_l0_buffer[(l - 1) + 1][theta_id] * RL_right[α][p_id])
            end
            dipole_z_cb_buffer[i, α] = dipole_z_cb
        end
    end

    # now we are able to calculate RHS part directly
    for i in 1: N_kappa
        res_u_free[i] = 0.0
        for α in alpha_list
            res_u_free[i] += dipole_z_cb_buffer[i, α] * u_bound[α] * exp(im * eigen_states[α]["Ip"] * t)
        end
        res_u_free[i] *= im * Et_data[it] * exp(im * chi_curves[i])
    end

    for α in alpha_list
        res_u_bound[α] = 0.0
        for i in 1: N_kappa
            res_u_bound[α] += wj * conj(dipole_z_cb_buffer[i, α]) * exp(-im * eigen_states[α]["Ip"] * t) * u_free[i]
        end
        res_u_bound[α] *= im * Et_data[it] * exp(-im * chi_curves[i])
    end

    for α in alpha_list
        for β in alpha_list
            res_u_bound[α] += im * Et_data[it] * dipole_z_bb[α, β] * exp(im * (eigen_states[β]["Ip"] - eigen_states[α]["Ip"]) * t) * u_bound[β]
        end
    end
end