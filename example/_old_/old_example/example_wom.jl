using JuliaProp
using Plots
using LinearAlgebra

Nr = 20000
Δr = 0.2
l_num = 15
Δt = 0.05
po_func_r = coulomb_potiential_zero_fixed()
rmax = Nr * Δr
absorb_func = absorb_boundary_r(rmax)


pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# init wave
init_wave_list = itp_fdsh(pw, rt, err=1e-9);
crt_shwave = deepcopy(init_wave_list[1]);

# Define laser here.
E0 = 0.02387
omega = 0.085
nc = 6.0
steps = Int64((2 * nc * pi / omega) ÷ Δt) + 10
t_linspace = [(i - 1) * Δt for i = 1: steps] 

Az(t) = (E0 / omega) * (sin(omega * t / 2.0 / nc) ^ 2) * sin(omega * t) * (t < (2 * nc * pi / omega))
At_data_x = zeros(ComplexF64, steps)
At_data_y = zeros(ComplexF64, steps)
At_data_z = Az.(t_linspace)

# propagation
tdseln_sh_mainloop_record(crt_shwave, pw, rt, At_data_z, steps, 500.0);


Emin = 0.0
Emax = 1.0
gamma = 0.001
energy_range = [e for e in Emin + energy_delta: energy_delta: Emax]

Plist = window_operator_method_sh(crt_shwave, gamma, 3, energy_range, rt, pw, WOM_MODE_PL)
plot(energy_range, [Plist], yscale=:log10, ylimits=(10^-15, 0.0))
