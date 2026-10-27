# Run from the project root: julia --project=. --threads=4 samples/ctmc_adaptive_example.jl
using JuliaProp

Δt = 0.1                         # Common output spacing; not the internal step.
t_num = 64
trajs_num = 2
Z = 1.0
rt = create_ctmc_rt(trajs_num, t_num, -5.0, 5.0, 0.1)

# An eccentric Kepler orbit returns to radius 0.01 near t=π.
# The other trajectory stays far from the core and needs fewer internal steps.
e = 0.99
rt.start_point_cc[4][1] = -(1+e)
rt.start_point_cc[2][1] = -sqrt((1-e)/(1+e))
rt.start_point_cc[4][2] = 50.0
rt.start_point_cc[2][2] = 0.1
rt.weight_cc .= 1.0
tid_cc = [1, 4]                  # Births can differ, output times stay aligned.
F1(E, Z, x, y, z, rj) = -E-Z*rj/(x*x+y*y+z*z)^1.5
E_data = [zeros(t_num) for _ = 1:3]
E_hf_data = [zeros(t_num-1) for _ = 1:3]

result = ctmc_adaptive_mainloop(
    F1, rt, t_num, trajs_num, tid_cc, Δt, E_data, E_hf_data, Z, 0.0;
    abstol=1e-12, reltol=1e-10, return_stats=true)
traj_filter_flag = result.traj_filter_flag

# The existing pipeline can consume the same runtime and Vector{Bool} flags.
asym_px, asym_py, asym_pz, bound_flags =
    calculate_asymptotic_momentum(rt, trajs_num, t_num, Z)

for i = 1:trajs_num
    s = result.stats[i]
    println("trajectory $i: accepted=$(s.accepted_steps), rejected=$(s.rejected_steps), " *
            "min_dt=$(s.min_dt), sampled_min_radius=$(s.min_radius)")
end
println("Both trajectories have $t_num samples at t[k]=(k-1)*Δt.")

# In laser samples, keep create_ctmc_rt / update_start_point as before and replace
# only ctmc_mainloop with ctmc_adaptive_mainloop, passing 0.0 as the threshold.
# Optional: use the actual laser functions at substeps rather than interpolating:
# flags = ctmc_adaptive_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt,
#     Et_data, Et_data_hf, Z, 0.0;
#     field=t -> (Ex_fs(t), Ey_fs(t)+E_applied(t), 0.0))
