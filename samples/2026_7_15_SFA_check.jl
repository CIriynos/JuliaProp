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
E_fs =          0.05                    # peak electric field of the fs pulse
ω_fs =          0.057 * 0.5               # angular frequency of the fs pulse
nc =            6                       # number of optical cycles in the fs pulse
Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, 0, ellipticity=0.0, phase1=0.5pi)        # create the light pulse from the given parameters (+ ellipticity)
E_field(t) = Ex_fs(t)
Tp = 2 * nc * pi / ω_fs


N_alpha = 5


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

k = 1
for κ_p in kappa_p_subgrid, κ_theta in kappa_theta_subgrid
    kappa_grid_p[k] = κ_p
    kappa_grid_theta[k] = κ_theta
    kappa_grid_x[k], kappa_grid_y[k], kappa_grid_z[k] = sphere_to_xyz(κ_p, κ_theta, 0.0)
    k += 1
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

mainloop_ts = 1: 2: (length(ts) - 2)

# # save everything
# example_name = "2026_7_14_$(E_fs)_$(ω_fs)_$(nc)_short_range_10"
# h5open("./data/$example_name.h5", "w") do h
#     write(h, "D_total", D_total)
#     write(h, "D_CC", D_CC)
#     write(h, "D_BB", D_BB)
#     write(h, "D_BC", D_BC)
# end

example_name = "2026_7_14_$(E_fs)_$(ω_fs)_$(nc)_short_range_10"
D_total = retrieve_mat(example_name, "D_total")
D_CC = retrieve_mat(example_name, "D_CC")
D_BB = retrieve_mat(example_name, "D_BB")
D_BC = retrieve_mat(example_name, "D_BC")
u = retrieve_mat(example_name, "u")
P_free_sum = retrieve_mat(example_name, "P_free_sum")
P_bound_sum = retrieve_mat(example_name, "P_bound_sum")
P_total = retrieve_mat(example_name, "P_total")


p1 = plot(ts[mainloop_ts], [D_total, D_BB, D_BC, D_CC], label=["D_total" "D_BB" "D_BC" "D_CC"])

p4 = plot(ts[mainloop_ts], P_free_sum, yscale=:log10, ylimits=(1e-8, 1e0))
plot(ts[mainloop_ts], P_free_sum, ylimits=(1e-8, 1e0))


# get harmonic spectrum, including data, and k axis (frequency axis)
# n_cut_off_estim = floor((-en + 3.17 * (E_fs ^ 2.0 / (4.0 * (ω_fs ^ 2.0)))) / ω_fs) * 1
n_cut_off_estim = 20

hg1, ks = get_hg_spectrum(ts[mainloop_ts], D_total, ω_fs * (n_cut_off_estim + 20))
hg1_free, _ = get_hg_spectrum(ts[mainloop_ts], D_CC, ω_fs * (n_cut_off_estim + 20))
hg1_bound, _ = get_hg_spectrum(ts[mainloop_ts], D_BB, ω_fs * (n_cut_off_estim + 20))
hg1_cross, _ = get_hg_spectrum(ts[mainloop_ts], D_BC, ω_fs * (n_cut_off_estim + 20))

# r
p3 = plot(ks ./ ω_fs, [(ks .^ 3) .* hg1, (ks .^ 3) .* hg1_free, (ks .^ 3) .* hg1_bound, (ks .^ 3) .* hg1_cross],
    yscale=:log10, xaxis=1:20, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-8, 1e3),
    label=["Total" "Free" "Bound" "Cross"], xlabel="Harmonic Order", ylabel="HG Intensity", title="HHG Spectrum (E0=$E_fs, ω=$ω_fs)")


#####################

# u = h5open("./data/2025_5_19.h5", "r") do h
#     _read_complex(h, "u")
# end

# get a 2D slice of u_free in x-z plane (ϕ = 0) (Spherical)
u_free_xz = zeros(ComplexF64, length(kappa_p_subgrid), length(kappa_theta_subgrid))
for i in 1: length(kappa_p_subgrid), j in 1: length(kappa_theta_subgrid)
    kappa_index = (i - 1) * length(kappa_theta_subgrid) + j
    kappa_p = kappa_p_subgrid[i]
    kappa_theta = kappa_theta_subgrid[j]
    w = (2 * pi * kappa_p^2) * sin(kappa_theta) * kappa_delta_p * kappa_delta_theta
    u_free_xz[i, j] = norm.(u[N_alpha + kappa_index]) .^ 2.0 / w
end
# colormap = cgrad([:white, :black, :red, :blue, :white])
colormap = cgrad([:white, palette(:jet1, 6)...], rev = true)
display_data = clamp.(abs.(log10.(norm.(u_free_xz) ./ maximum(norm.(u_free_xz)))), 0, 3.0)
# display_data = (display_data ./ 5.0) .^ 0.5 * 5.0
heatmap([kappa_theta_subgrid; π .+ kappa_theta_subgrid], kappa_p_subgrid, [display_data display_data[:, end:-1:1]], projection=:polar, color=colormap)

colormap_2 = cgrad(["#710000", "#C60001", "#FB0001", "#FE3A01", "#FF7A00", "#FEAB01", "#FEAB01", "#FEAB01", "#D5FF17", "#91FF60", "#4AFFAE", "#02FFFC", "#00A3FE", "#003DFF", "#2120FF", "#8C8CFE", :white], rev=true)
p2 = heatmap([kappa_theta_subgrid .+ π/2; kappa_theta_subgrid .+ 3π/2], kappa_p_subgrid[1: end ÷ 2], [norm.(u_free_xz) norm.(u_free_xz)[:, end:-1:1]][1: end ÷ 2, :], projection=:polar, color=colormap_2)

p3