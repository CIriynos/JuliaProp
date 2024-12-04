import Pkg
Pkg.activate(".")

using Revise
using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using Printf
using FFTW
println("Number of Threads: $(Threads.nthreads())")


# Define Basic Parameters
Nx = 20000
delta_x = 0.02
delta_t = 0.005
delta_t_itp = 0.01
Lx = Nx * delta_x
Xi = 150
# po_func(x) = -(x^2 + 1) ^ (-0.5) * flap_top_windows_f(x, -Xi, Xi, 1/4)
ZZ = 1.0567
AA = 200
po_func(x) = -ZZ * sqrt(AA / pi) * exp(-AA * x^2)
imb_func(x) = -100im * ((abs(x) - Xi) / (Lx / 2 - Xi)) ^ 8 * (abs(x) > Xi)


# Create Physics World & Runtime
pw1d = create_physics_world_1d(Nx, delta_x, delta_t, po_func, imb_func, delta_t_im=delta_t_itp)
rt1d = create_tdse_rt_1d(pw1d)


# Get Initial Wave
x_linspace = get_linspace(pw1d.xgrid)
seed_wave = gauss_package_1d(x_linspace, 1.0, 1.0, 0.0)
init_wave = itp_fd1d(seed_wave, rt1d, min_error = 1e-10)
get_energy_1d(init_wave, rt1d)

# Define Laser.
E0 = 0.05
omega = 0.05
nc = 6
phi = pi / 2
# -nc * pi / omega ~ nc * pi / omega
shift = nc * pi / omega
Tp = 2 * nc * pi / omega
At(t) = E0 / omega * cos(omega * (t - shift) / 2 / nc) ^ 8 * cos(omega * (t - shift) + phi) * (t > 0 && t < 2pi * nc / omega)

T_total = Tp * 3
steps = Int64(T_total ÷ delta_t)
t_linspace = create_linspace(steps, delta_t)
At_data = At.(t_linspace)

# TDSE 1d
crt_wave = deepcopy(init_wave)
Xi_data, hhg_integral, energy_list = tdse_laser_fd1d_mainloop_penta(crt_wave, rt1d, pw1d, At_data, steps, Xi)


# t-surf
k_delta = 0.002
kmin = -2.0
kmax = 2.0
k_linspace = kmin: k_delta: kmax
Pk = tsurf_1d(pw1d, k_linspace, t_linspace, At_data, Xi, Xi_data)
plot(k_linspace, Pk, yscale=:log10)