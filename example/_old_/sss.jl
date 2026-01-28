using JuliaProp

# Define Parameters
Nr = 3000
Δr = 0.2
l_num = 15
Δt = 0.05
Z = 1.0
steps = 10000
Ri_tsurf = 450.0

pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func_r, Z, absorb_func)
rt = create_tdse_rt_sh(pw);

# Define Laser
Ax(t) = (E0 / ω) * (sin(ω * t / 2.0 / nc) ^ 2) * sin(ω * t) * (t < (2 * nc * pi / ω))

# Start Propagation
tdse_elli_sh_mainloop_record_xy(pw, rt, Ax, steps, Ri_tsurf);

# Analysis Photoelectron Spectrum
isurf_sh_vector(pw, rt, TSURF_MODE_ELLI);