# Run from the project root:
# julia --project=. --threads=4 samples/ctmc_hybrid_example.jl
using JuliaProp

Δt, t_num, trajs_num, Z = 0.1, 64, 2, 1.0
rt = create_ctmc_rt(trajs_num, t_num, -5.0, 5.0, 0.1)
e = 0.99
rt.start_point_cc[4][1] = -(1+e)
rt.start_point_cc[2][1] = -sqrt((1-e)/(1+e))
rt.start_point_cc[4][2] = 1000.0
rt.start_point_cc[2][2] = 0.1
rt.weight_cc .= 1.0
tid_cc = [1, 4]
E_data = [zeros(t_num) for _ = 1:3]
E_hf_data = [zeros(t_num-1) for _ = 1:3]
F1(E, Z, x, y, z, rj) = -E-Z*rj/(x*x+y*y+z*z)^1.5

result = ctmc_hybrid_mainloop(
    F1, rt, t_num, trajs_num, tid_cc, Δt, E_data, E_hf_data, Z, 0.0;
    switch_radius=1.5, abstol=1e-12, reltol=1e-10, return_stats=true)
traj_filter_flag = result.traj_filter_flag
for i = 1:trajs_num
    s = result.stats[i]
    println("trajectory $i: fixed=$(s.fixed_steps), " *
        "adaptive intervals=$(s.adaptive_intervals), replayed=$(s.replayed_steps), " *
        "adaptive accepted=$(s.adaptive.accepted_steps), " *
        "adaptive rejected=$(s.adaptive.rejected_steps)")
end

# Existing add_to_pmd/add_to_hhg calls can consume rt and traj_filter_flag.
# switch_radius controls numerical integration. The positional 0.0 disables
# distance-based trajectory filtering, so accurate close returns are retained.
