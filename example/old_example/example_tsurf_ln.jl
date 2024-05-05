# using JuliaProp
# using Plots
# using LinearAlgebra
# using SpecialFunctions

# Nr = 2000
# Δr = 0.2
# l_num = 15
# Δt = 0.05
# Z = 1.0
# po_func_r = coulomb_potiential_zero_fixed()
# rmax = Nr * Δr
# absorb_func = absorb_boundary_r(rmax)

# pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
# rt = create_tdse_rt_sh(pw);

# # init wave
# init_wave_list = itp_fdsh(pw, rt, err=1e-9);
# crt_shwave = deepcopy(init_wave_list[1]);

# # Define laser here.
# E0 = 0.02387
# omega = 0.085
# nc = 6.0
# steps = Int64((2 * nc * pi / omega) ÷ Δt) * 1 + 1
# t_linspace = [(i - 1) * Δt for i = 1: steps] 

# Az(t) = (E0 / omega) * (sin(omega * t / 2.0 / nc) ^ 2) * sin(omega * t) * (t < (2 * nc * pi / omega))
# At_data_x = zeros(ComplexF64, steps)
# At_data_y = zeros(ComplexF64, steps)
# At_data_z = Az.(t_linspace)

# # propagation
# Ri_tsurf = 250.0
# phi_record, dphi_record = tdseln_sh_mainloop_record(crt_shwave, pw, rt, At_data_z, steps, Ri_tsurf);

# Emin = 0.002
# Emax = 1.0
# energy_delta = 0.002
# energy_range = [e for e in Emin + energy_delta: energy_delta: Emax]
# k_linspace = sign.(energy_range) .* sqrt.(2 * abs.(energy_range))

# a_tsurff_1 = isurf_sh(pw, rt, phi_record, dphi_record, crt_shwave, At_data_x, At_data_y, At_data_z, Ri_tsurf, t_linspace, k_linspace, TSURF_MODE_PL)

# P1 = tsurf_get_energy_spectrum(a_tsurff_1, k_linspace, pw)
# plot(energy_range, [norm.(P1)], yscale=:log10, ylimits=(10^-15, 0.0))