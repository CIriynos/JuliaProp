using JuliaProp
using Plots
using LinearAlgebra

# set basic parameters
Nx = 25000
Δx = 0.2
Δt = 0.05

# define potiential function
# Note: In JuliaProp, we no longer use OOP (like Class XX) to represent a potiential.
#       Instead, we use a Function to realize that goal, which is the direct translation of math formula in paper.
#       In Julia, function can be used as parameters in other function, just like the function-pointer in C++.
#
#       The underlying function define a coulomb-like potiential. You can create arbitrary potiential as you wish.
function po_func(x)
    Rco = 100.0
    if abs(x) < Rco
        return -1.0 / sqrt(x * x + 1.0)
    elseif abs(x) < 2 * Rco
        return -(2*Rco - abs(x)) / Rco^2
    else
        return 0.0
    end
end

# create physics_world and runtime
# Note: "physics world" is a collection of arguments that crucial for computational physics,
#       like how we create the numerical grid, the time resolution when propagating, the shape of potiential, etc...
#
#       A "runtime" is all nessasary (or temporary) variables we need when propagating wavefunction,
#       evaluating spectrum or getting state energy. A runtime ("rt" in short) will only correspond to one physics world.
#       But one physics world ("pw" in short) can create several kinds of runtime, depending on what algorithm we apply.
world = create_physics_world_1d(Nx, Δx, Δt, po_func)
runtime = create_tdse_rt_1d(world);


# In the same way, lets create the physics world for t-SURFF
Nx_tsurf = 3600
imag_area_width = 60.0
imag_area_boundary = 300.0
imb_func(x) = @. -100.0im * ((abs(x) - imag_area_boundary) / imag_area_width) ^ 8 * (abs(x) - imag_area_boundary > 0.0)

world_tsurf = create_physics_world_1d(Nx_tsurf, Δx, Δt, po_func, imb_func)
runtime_tsurf = create_tdse_rt_1d(world_tsurf);
# plot(norm.(world_tsurf.po_data_im))   # check the imag boundary


# get init_wave by imagary-time propagation (ITP)
# Note: In this section, we first create a 1-D gauss package as seed wave.
#       Then we apply imagary-time propagation method to get the ground state.
#       "fd1d" means that this function use finite element method (crank-nicolson method)
seed_wave = gauss_package_1d(get_linspace(world.xgrid), 2.0, 2.0, 2.0)
init_wave = itp_fd1d(seed_wave, runtime; max_steps=1000)
crt_wave = copy(init_wave)
get_energy_1d(crt_wave, runtime)


# define laser field
# Note: In JuliaProp, laser field is controlled by user. The program will only receive the steps of propagation,
#       and the data of At (or Et, in position gauge).
#       All you need to do is set your own laser depending on the question, then put them into function.
A0 = 1.0
ω = 0.057
nc = 6
At_func(t) = A0 * sin(ω * t / (2 * nc)) ^ 2 * sin(ω * t) * (t < (2 * nc * pi / ω))
steps = Int64((2 * nc * pi / ω) ÷ Δt) * 1
t_linspace = [i * Δt for i = 1: steps]
At_data = At_func.(t_linspace)
# plot(t_linspace, At_data)

# start propagation
tdse_laser_fd1d_mainloop_penta(crt_wave, runtime, world, At_data, steps, 0.0)

# using WOM method to get the energy spectrum
gamma = 0.005
n_wom = 3
Ev_list = 0.0: 2 * gamma: 3.0
Plist_total, Plist = windows_operator_method_1d(crt_wave, gamma, n_wom, Ev_list, runtime, world)
# plot(log10.(Plist), ylimits=(-15, 0))



### tsurf part
# Note: this section is just the same as the above codes.
seed_wave_tsf = gauss_package_1d(get_linspace(world_tsurf.xgrid), 2.0, 2.0, 2.0)
init_wave_tsf = itp_fd1d(seed_wave_tsf, runtime_tsurf; max_steps=1000)
crt_wave_tsf = copy(init_wave_tsf)

steps_tsurf = Int64((2 * nc * pi / ω) ÷ Δt) * 2
t_linspace_tsurf = [i * Δt for i = 1: steps_tsurf]
At_data_tsurf = At_func.(t_linspace_tsurf)
energy_delta = 0.01
energy_linspace = 0: energy_delta: 3.0
k_linspace = sign.(energy_linspace) .* sqrt.(2 * abs.(energy_linspace))

X = 200.0
# tdse with laser, and record the value (and its derivative) on X.
X_pos_vals, X_neg_vals, X_pos_dvals, X_neg_dvals = tdse_laser_fd1d_mainloop_penta(crt_wave_tsf, runtime_tsurf, world_tsurf, At_data_tsurf, steps_tsurf, X)

# execute tsurf
Pk = tsurf_1d(world_tsurf, k_linspace, t_linspace_tsurf, At_data_tsurf, X, X_pos_vals, X_neg_vals, X_pos_dvals, X_neg_dvals)
# plot(energy_linspace, log10.(Pk), ylimits=(-15, 0))



# The final step, let's put then in the same figure (to verify the Correctness of t-SURFF and WOM)
bias = 10
plot(energy_linspace, [log10.(Plist * bias) log10.(Pk)], ylimits=(-15, 0))