import Pkg
Pkg.activate(".")
# using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

e_fs_list = 0.01: 0.0025: 0.1
# e_fs_list = 0.01: 0.005: 0.1
# e_dc_list = [0.0, 0.001, 0.002]
e_dc_list = [0.000]
ω_fs = 0.057 * 0.5
nc = 12 / 2

base_total_data = []
base_bound_data = []
base_free_data = []

shg_total_data = []
shg_bound_data = []
shg_free_data = []

thg_total_data = []
thg_bound_data = []
thg_free_data = []

thz_total_data = []
thz_bound_data = []
thz_free_data = []

free_norm_value_data = []
norm_value_data = []

hg_figure_collection = []

ks = []
thz_id = 1
base_id = 1
shg_id = 1
thg_id = 1

for e_fs in e_fs_list
for e_dc in e_dc_list

# Basic Parameters
@expo grid_ratio = 1
@expo Nr =            20000 * grid_ratio           # number of radial grid points
@expo Δr =            0.2 / grid_ratio             # radial grid step size
@expo l_num =         200                  # number of angular momentum components
@expo Δt =            0.05 / grid_ratio            # time step size
@expo Z =             1.0                  # nuclear charge
# @expo po_func(r) =    -1 / r              # potential function
po_func(r) =    -1 / r * exp(- r * r / (20.0 ^ 2))   # a short-range potential function, which is used to test the ITP method for getting the initial wavefunction in a short-range potential
@expo rmax =          Nr * Δr     
@expo absorb_func     = absorb_boundary_r(rmax, rmax * 0.8)  # create absorbing boundary function
@expo Ri_tsurf        = rmax * 0.7        # radius for t-surf method

# Define the Laser pulse
E_fs =          e_fs                # peak electric field of the fs pulse
E_thz =         0.00005 * 0         # peak electric field of the THz pulse
E_dc =          e_dc                # static electric field
# ω_fs =          0.057 * 1.0           # angular frequency of the fs pulse
ω_thz =         ω_fs / 30           # angular frequency of the THz pulse
# nc =            12                  # number of optical cycles in the fs pulse
tau_fs =        0                   # delay of the fs pulse
tau_list = get_1c_thz_delay_list_ok(ω_fs, tau_fs, nc, ω_thz)
                                    # A helper function to get the delays of the THz pulse, 
                                    # which ensures consecutively scanning through the whole duration of the fs pulse.
tau_thz = tau_list[1]               # select one delay from the delay list

# create the Laser pulse data
Ex_fs, Ey_fs, Ez_fs, tmax = light_pulse(ω_fs, E_fs, nc, tau_fs, ellipticity=0.0, phase1=0.0)        # create the light pulse from the given parameters (+ ellipticity)
Ex_thz, = light_pulse(ω_thz, E_thz, 1, tau_thz, pulse_shape="sin2", phase1=0.5pi)       # create the THz pulse from the given parameters
E_applied(t) = (Ex_thz(t) + E_dc) * flap_top_windows_f(t, 0, tmax, 1/2)                 # define the applied electric field (THz + DC) with a flattop window
At_datas, Et_datas, ts, steps = create_tdata(tmax, 0, Δt, t -> Ex_fs(t) + E_applied(t), Ey_fs, no_light, appendix_steps=1)
                                                                                        # create the vector potential and electric field data for propagation


# Retrieve Data
example_name = "2026_4_26_hg_massive_$(E_fs)_$(E_dc)_$(ω_fs)_$(nc)_short_range"
hhg_integral_t = retrieve_mat(example_name, "hhg_integral_t")
hhg_integral_t_free = retrieve_mat(example_name, "hhg_integral_t_free")
hhg_integral_t_bound = retrieve_mat(example_name, "hhg_integral_t_bound")
hhg_integral_t_bound = retrieve_mat(example_name, "hhg_integral_t_bound")
free_norm_value = retrieve_mat(example_name, "free_norm_value")
norm_value = retrieve_mat(example_name, "norm_value")


# get harmonic spectrum, including data, and k axis (frequency axis)
n_display = 50

hg1, hg1_phase, ks = get_hg_spectrum_from_dipole(ts, hhg_integral_t, ω_fs * (n_display))
hg1_free, hg1_free_phase, _ = get_hg_spectrum_from_dipole(ts, hhg_integral_t_free, ω_fs * (n_display))
hg1_bound, hg1_bound_phase, _ = get_hg_spectrum_from_dipole(ts, hhg_integral_t_bound, ω_fs * (n_display))
hg1_cross, hg1_cross_phase, _ = get_hg_spectrum_from_dipole(ts, hhg_integral_t .- hhg_integral_t_free .- hhg_integral_t_bound, ω_fs * (n_display))

thz_id = 1
base_id = Int64((ω_fs ÷ (ks[2] - ks[1]))) + 0
shg_id = Int64(((2 * ω_fs) ÷ (ks[2] - ks[1]))) + 0
thg_id = Int64(((3 * ω_fs) ÷ (ks[2] - ks[1]))) + 0

# r
p = plot(ks ./ ω_fs, [hg1, hg1_free, hg1_bound],
    yscale=:log10, xaxis=1:20, yaxis=[1e-4, 1e-2, 1e0, 1e2, 1e4], ylimit=(1e-8, 1e3),
    label=["Total" "Free" "Bound"], xlabel="Harmonic Order", ylabel="HG Intensity", title="HHG Spectrum (E0=$E_fs, ω=$ω_fs)")

# indicator of base, shg, thg
plot!(p, [base_id * (ks[2] - ks[1]) ./ ω_fs, base_id * (ks[2] - ks[1]) ./ ω_fs], [1e-8, 1e3], color=:grey, line=(:dot, 0.8))
plot!(p, [shg_id * (ks[2] - ks[1]) ./ ω_fs, shg_id * (ks[2] - ks[1]) ./ ω_fs], [1e-8, 1e3], color=:grey, line=(:dot, 0.8))
plot!(p, [thg_id * (ks[2] - ks[1]) ./ ω_fs, thg_id * (ks[2] - ks[1]) ./ ω_fs], [1e-8, 1e3], color=:grey, line=(:dot, 0.8))

hg1_cplx = hg1 .* exp.(im * (hg1_phase .+ 0))
hg1_free_cplx = hg1_free .* exp.(im * (hg1_free_phase .+ 0))
hg1_bound_cplx = hg1_bound .* exp.(im * (hg1_bound_phase .+ 0))

ave_limit = 0

push!(base_total_data, sum(hg1_cplx[base_id - ave_limit: base_id + ave_limit]) / (2 * ave_limit + 1))
push!(base_free_data, sum(hg1_free_cplx[base_id - ave_limit: base_id + ave_limit]) / (2 * ave_limit + 1))
push!(base_bound_data, sum(hg1_bound_cplx[base_id - ave_limit: base_id + ave_limit]) / (2 * ave_limit + 1))

push!(shg_total_data, sum(hg1_cplx[shg_id - ave_limit: shg_id + ave_limit]) / (2 * ave_limit + 1))
push!(shg_free_data, sum(hg1_free_cplx[shg_id - ave_limit: shg_id + ave_limit]) / (2 * ave_limit + 1))
push!(shg_bound_data, sum(hg1_bound_cplx[shg_id - ave_limit: shg_id + ave_limit]) / (2 * ave_limit + 1))

push!(thg_total_data, sum(hg1_cplx[thg_id - ave_limit: thg_id + ave_limit]) / (2 * ave_limit + 1))
push!(thg_free_data, sum(hg1_free_cplx[thg_id - ave_limit: thg_id + ave_limit]) / (2 * ave_limit + 1))
push!(thg_bound_data, sum(hg1_bound_cplx[thg_id - ave_limit: thg_id + ave_limit]) / (2 * ave_limit + 1))

push!(thz_total_data, sum(hg1_cplx[thz_id: thz_id + ave_limit]) / (1 * ave_limit + 1))
push!(thz_free_data, sum(hg1_free_cplx[thz_id: thz_id + ave_limit]) / (1 * ave_limit + 1))
push!(thz_bound_data, sum(hg1_bound_cplx[thz_id: thz_id + ave_limit]) / (1 * ave_limit + 1))

push!(free_norm_value_data, free_norm_value)
push!(norm_value_data, norm_value)
push!(hg_figure_collection, p)

ks = ks

end
end

function convert_eletric_field_strength_to_light_intensity(E)
    return 3.509e16 * (E ^ 2)
end

intensity_list = convert_eletric_field_strength_to_light_intensity.(e_fs_list)


# Plotting Reflective index
N0 = 3.98e-6    # ambient air
# chi_eff = @. (4 * pi) * N0 * sqrt(base_total_data * (1.0 / (ks[base_id] ^ 3))) / e_fs_list * exp(im * angle.(base_total_data))
chi_eff = @. (1) * N0 * sqrt(base_total_data * (1.0 / (ks[base_id] ^ 3))) / e_fs_list * exp(im * angle.(base_total_data))
n_index = @. sqrt(1 + chi_eff)

chi_eff_bound = @. (1) * N0 * sqrt(base_bound_data * (1.0 / (ks[base_id] ^ 3))) / e_fs_list * exp(im * angle.(base_bound_data))
n_index_bound = @. sqrt(1 + chi_eff_bound)

chi_eff_free = @. (1) * N0 * sqrt(base_free_data * (1.0 / (ks[base_id] ^ 3))) / e_fs_list * exp(im * angle.(base_free_data))
n_index_free = @. sqrt(1 + chi_eff_free)

offset = 1e-4
n3 = real.(n_index_free) .+ (real.(n_index)[1] - real.(n_index_free)[1])
n4 = real.(n_index) .- (real.(n_index_bound) .+ 1.13e-4) .+ n3[1]
p_n = plot(intensity_list, [real.(n_index) real.(n_index_bound) .+ 1.13e-4  n3  n4], ylimit=(real.(n_index)[1] - offset * 2, real.(n_index)[1] + offset),
    # xscale=:log10, xminorticks=true, 
    marker=:x, legend=:bottomright)
plot!(twinx(), intensity_list, real.(free_norm_value_data), fillrange=1e-20, fillalpha = 0.2, fillstyle = :/, yscale=:log10, ylimit=(1e-6, 1e1))


kk = (real.(n_index)[5] - real.(n_index)[1]) / (intensity_list[5] - intensity_list[1])
n_index_linear = @. real.(n_index)[1] + kk * (intensity_list - intensity_list[1])

p1 = plot(intensity_list, [norm.(base_total_data) norm.(base_free_data) norm.(base_bound_data)], 
    xscale=:log10, yscale=:log10, xminorticks=true, yminorticks = true,
    # palette = :bam10,
    ylimit=(1e-6, 1e2),
    label=["1th harmonic total" "1th harmonic free" "1th harmonic bound"],
    marker=:dot, legend=:bottomright)
plot!(p1, intensity_list, real.(free_norm_value_data), fillrange=1e-20, fillalpha = 0.2, fillstyle = :/, label="ionization rate")
offset = 1e-4
plot!(twinx(), intensity_list, [real.(n_index) n_index_linear], ylimit=(real.(n_index)[1] - offset, real.(n_index)[1] + offset),
    marker=:x, xscale=:log10, label=["" "linear Kerr model"], legend=:bottomleft)


p1_phase = plot(intensity_list, [angle.(base_total_data) angle.(base_free_data) angle.(base_bound_data)], 
    xscale=:log10,
    label=["1th harmonic total" "1th harmonic free" "1th harmonic bound"],
    ylimit=(-pi - 0.1, pi + 0.1), yticks = ([-pi, -pi / 2, 0, pi / 2, pi], ["-π", "-π/2", "0", "π/2", "π"]),
    marker=:dot, legend=:bottomright)

p2 = plot(intensity_list, [norm.(shg_total_data) norm.(shg_free_data) norm.(shg_bound_data)], 
    yscale=:log10, xscale=:log10, 
    # palette = :hawaii10,
    ylimit=(1e-8, 1e2),
    label=["2nd harmonic total" "2nd harmonic free" "2nd harmonic bound"],
    marker=:dot, legend=:bottomright)
plot!(p2, intensity_list, real.(free_norm_value_data), fillrange=1e-20, fillalpha = 0.2, fillstyle = :/)

p2_phase = plot(intensity_list, [angle.(shg_total_data) angle.(shg_free_data) angle.(shg_bound_data)], 
    xscale=:log10,
    ylimit=(-pi - 0.1, pi + 0.1), yticks = ([-pi, -pi / 2, 0, pi / 2, pi], ["-π", "-π/2", "0", "π/2", "π"]),
    label=["2nd harmonic total" "2nd harmonic free" "2nd harmonic bound"],
    marker=:dot, legend=:bottomright)

p3 = plot(intensity_list, [norm.(thg_total_data) norm.(thg_free_data) norm.(thg_bound_data)], 
    yscale=:log10, xscale=:log10,
    # palette = :berlin10,
    ylimit=(1e-8, 1e2),
    label=["3rd harmonic total" "3rd harmonic free" "3rd harmonic bound"],
    marker=:dot, legend=:bottomright)
plot!(p3, intensity_list, real.(free_norm_value_data), fillrange=1e-20, fillalpha = 0.2, fillstyle = :/)

p3_phase = plot(intensity_list, [angle.(thg_total_data) angle.(thg_free_data) angle.(thg_bound_data)], 
    xscale=:log10,
    label=["3rd harmonic total" "3rd harmonic free" "3rd harmonic bound"],
    ylimit=(-pi - 0.1, pi + 0.1), yticks = ([-pi, -pi / 2, 0, pi / 2, pi], ["-π", "-π/2", "0", "π/2", "π"]),
    marker=:dot, legend=:bottomright)

p4 = plot(intensity_list, [norm.(thz_total_data) norm.(thz_free_data) norm.(thz_bound_data)], 
    xscale=:log10, yscale=:log10, xminorticks=true, yminorticks = true,
    # palette = :bam10,
    ylimit=(1e-10, 1e2),
    label=["THz total" "THz free" "THz bound"],
    marker=:dot, legend=:bottomright)
plot!(p4, intensity_list, real.(free_norm_value_data), fillrange=1e-20, fillalpha = 0.2, fillstyle = :/)

p4_phase = plot(intensity_list, [angle.(thz_total_data) angle.(thz_free_data) angle.(thz_bound_data)], 
    xscale=:log10,
    label=["THz total" "THz free" "THz bound"],
    ylimit=(-pi - 0.1, pi + 0.1), yticks = ([-pi, -pi / 2, 0, pi / 2, pi], ["-π", "-π/2", "0", "π/2", "π"]),
    marker=:dot, legend=:bottomright)

plot(p1, p2, p3)

plot(intensity_list, norm_value_data)

hg_figure_collection[28]