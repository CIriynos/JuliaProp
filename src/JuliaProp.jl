module JuliaProp

using Plots
using LinearAlgebra
using BenchmarkTools
using LoopVectorization
using DelimitedFiles
using WignerSymbols
using SpecialFunctions
using SphericalHarmonics
using HDF5
using FFTW
using Statistics

export Grid1D, GridSH
export xyz_to_sphere, sphere_to_xyz, CG_coefficient, spherical_bessel_func
export get_SH_integral
export get_linspace, grid_index, grid_reduce
export wave_t, shwave_t, create_empty_shwave, copy_shwave
export trimat_elimination
export pentamat_t, create_empty_pentamat, create_pentamat, create_identity_pentamat, create_diag_pentamat, pentamat_to_mat, penta_mul, pentamat_elimination
export four_order_difference, two_order_difference
export get_derivative_two_order, get_integral
export get_m_from_mm, create_lmmap, get_index_from_lm
export save_object, open_object, save_shwave, open_shwave
export computeYlm
export numerical_integral

export create_physics_world_1d
export create_tdse_rt_1d
export gauss_package_1d
export get_energy_1d, get_energy_1d_laser
export itp_fd1d
export tdse_fd1d_mainloop
export tdse_fd1d_mainloop_penta
export tdse_laser_fd1d_mainloop_penta
export windows_operator_method_1d, windows_operator_method_1d_laser
export tsurf_1d
export gauge_transform_V2L

export get_energy_sh_mbunch
export get_energy_sh_so
export get_energy_sh
export fdsh_no_laser_one_step_so
export gram_schmidt_sh_so
export itp_fdsh
export create_physics_world_sh
export create_tdse_rt_sh

export tdseln_sh_mainloop
export tdseln_sh_mainloop_record
export tdseln_sh_mainloop_record_optimized
export tdseln_sh_mainloop_record_optimized_hhg
export tdseln_sh_mainloop_length_gauge
export tdseln_sh_mainloop_length_gauge_hhg
export tdse_elli_sh_mainloop_record
export tdse_elli_sh_mainloop_record_xy
export tdse_elli_sh_mainloop_record_xy_hhg
export tdse_elli_sh_mainloop_record_xy_optimized
export tdse_elli_sh_mainloop_record_xy_hhg_optimized
export tdse_elli_sh_mainloop_record_xy_hhg_long_prop

export window_operator_method_sh

export TSURF_MODE_PL, TSURF_MODE_ELLI
export WOM_MODE_PL, WOM_MODE_ELLI
export tsurf_sh, tsurf_sh_vector
export tsurf_get_energy_spectrum
export tsurf_plot_energy_spectrum
export tsurf_plot_xy_momentum_spectrum
export tsurf_plot_xy_momentum_spectrum_vector
export tsurf_plot_xz_momentum_spectrum_vector
export tsurf_get_average_momentum_parallel
export tsurf_get_average_momentum_vector_parallel
export isurf_sh, isurf_rest_part, isurf_sh_vector
export create_k_space, theta_linspace, phi_linspace
export fixed_r, fixed_theta, fixed_phi
export tsurf_combine_lm_vec
export plot_line_order
export tsurf_get_average_momentum

export coulomb_potiential_zero_fixed
export coulomb_potiential_zero_fixed_plus
export coulomb_potiential_helium_zero_fixed_plus
export absorb_boundary_r
export create_linspace
export store_mat, store_obj, retrieve_mat, retrieve_obj
export get_smoothness_1, get_smoothness_2

export coulomb_potiential_zero_fixed_COS
export get_hhg_spectrum_xy
export flap_top_windows_f

include("util.jl")

# util.jl -> tdse_1d.jl
include("tdse_1d.jl")

# util.jl -> tdse_sh_base.jl ---> tdse_sh_ln.jl
#                             ├-> tdse_sh_elli.jl
#                             └-> tsurf_sh.jl | wom_sh.jl
include("tdse_sh_base.jl")

include("tdse_sh_ln.jl")
include("tdse_sh_elli.jl")
include("tsurf_sh.jl")
include("wom_sh.jl")

include("prefabricated.jl")

greet() = println("helloworld.")


end # module JuliaProp
