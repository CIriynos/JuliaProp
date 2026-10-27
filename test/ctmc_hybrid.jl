@testset "hybrid far field uses one RK4 and preserves old results" begin
    nt, n, dt, t0 = 31, 3, 0.1, 0.37
    field(t) = (0.01sin(t), 0.02cos(t), 0.0)
    E, Eh = fields(field, nt, dt; t0=t0)
    old, hybrid, callback = new_rt(n, nt), new_rt(n, nt), new_rt(n, nt)
    tids = [1, 5, nt]
    for i = 1:n
        y0 = [0.1, 0.2, 0.0, 1000.0+i, 1.0, 0.0]
        setstate!(old, i, y0)
        setstate!(hybrid, i, y0)
        setstate!(callback, i, y0)
        for j = 1:6
            hybrid.traj_data[j][i] .= 123.0
        end
    end
    ctmc_mainloop(coulomb, old, nt, n, tids, dt, E, Eh, 1.0, 0.0)
    result = ctmc_hybrid_mainloop(coulomb, hybrid, nt, n, tids, dt, E, Eh, 1.0;
        switch_radius=1.0, dtmax=dt/10, t0=t0, return_stats=true)
    @test result.traj_filter_flag == falses(n)
    @test hybrid.traj_data == old.traj_data
    cb = ctmc_hybrid_mainloop(coulomb, callback, nt, n, tids, dt, nothing, nothing, 1.0;
        switch_radius=1.0, field=field, t0=t0, return_stats=true)
    @test all(s -> s.adaptive_intervals == 0, cb.stats)
    for i = 1:n, j = 1:6
        @test callback.traj_data[j][i] ≈ old.traj_data[j][i] atol=1e-12 rtol=1e-13
    end
    for i = 1:n
        @test result.stats[i].fixed_steps == nt-tids[i]
        @test result.stats[i].adaptive_intervals == 0
        @test result.stats[i].replayed_steps == 0
        @test result.stats[i].adaptive.accepted_steps == 0
        @test result.stats[i].adaptive.min_dt == 0
    end
    # Exact force-call count: far steps do not pay for step-doubling.
    one = new_rt(1, nt)
    setstate!(one, 1, [0, 0.1, 0, 1000, 0, 0])
    calls = Ref(0)
    counted = (args...) -> (calls[] += 1; coulomb(args...))
    ctmc_hybrid_mainloop(counted, one, nt, 1, [1], dt, E, Eh, 1.0;
        switch_radius=1.0)
    @test calls[] == 12*(nt-1)
end

@testset "hybrid guards a crossing with both endpoints outside" begin
    E, Eh = fields(zero_field, 2, 1.0)
    rt = new_rt(1, 2)
    setstate!(rt, 1, [3, 0, 0, -1, 0, 0])
    result = ctmc_hybrid_mainloop(electric_only, rt, 2, 1, [1], 1.0, E, Eh, 0.0;
        switch_radius=0.01, return_stats=true)
    @test result.stats[1].fixed_steps == 0
    @test result.stats[1].adaptive_intervals == 1
    @test result.stats[1].replayed_steps == 0 # Predicted before a fixed trial.
    @test state(rt, 1, 2) ≈ [3, 0, 0, 2, 0, 0]
end

@testset "hybrid rolls back an unsafe fixed RK trial" begin
    # Initial velocity predicts no approach. Acceleration makes an RK stage chord
    # cross the sphere, so the full interval must be replayed from the old state.
    E, Eh = fields(t -> (-6.0, 0.0, 0.0), 2, 1.0)
    rt = new_rt(1, 2)
    setstate!(rt, 1, [0, 0, 0, -1, 0, 0])
    result = ctmc_hybrid_mainloop(electric_only, rt, 2, 1, [1], 1.0, E, Eh, 0.0;
        switch_radius=0.1, return_stats=true)
    @test result.stats[1].fixed_steps == 0
    @test result.stats[1].adaptive_intervals == 1
    @test result.stats[1].replayed_steps == 1
    @test result.stats[1].adaptive.rejected_steps == 0
    @test state(rt, 1, 1) == [0, 0, 0, -1, 0, 0]
    @test state(rt, 1, 2) ≈ [6, 0, 0, 2, 0, 0]
end

@testset "hybrid all-near matches adaptive, including callback and filtering" begin
    nt, dt, t0 = 21, 0.1, 1.7
    field(t) = (0.01sin(t), 0.0, 0.0)
    E, Eh = fields(field, nt, dt; t0=t0)
    adaptive, hybrid = new_rt(2, nt), new_rt(2, nt)
    soft(E, Z, x, y, z, rj) = -E-Z*rj/(x*x+y*y+z*z+1e-5)^1.5
    for rt in (adaptive, hybrid)
        setstate!(rt, 1, [2, 0, 0, -1, 0, 0])
        setstate!(rt, 2, [0, 0.1, 0, 100, 0, 0])
    end
    a = ctmc_adaptive_mainloop(soft, adaptive, nt, 2, [1, 4], dt,
        nothing, nothing, 1.0, 0.05;
        field=field, t0=t0, abstol=1e-11, reltol=1e-9, return_stats=true)
    b = ctmc_hybrid_mainloop(soft, hybrid, nt, 2, [1, 4], dt,
        nothing, nothing, 1.0, 0.05;
        switch_radius=1e6, field=field, t0=t0,
        abstol=1e-11, reltol=1e-9, return_stats=true)
    @test hybrid.traj_data == adaptive.traj_data
    @test b.traj_filter_flag == a.traj_filter_flag == [true, false]
    for i = 1:2
        @test b.stats[i].fixed_steps == b.stats[i].replayed_steps == 0
        @test b.stats[i].adaptive_intervals == nt-[1, 4][i]
        for name in fieldnames(CTMCCore.CTMCAdaptiveStats)
            @test getfield(b.stats[i].adaptive, name) == getfield(a.stats[i], name)
        end
    end
end

@testset "hybrid Kepler recollision and switch-radius convergence" begin
    nt, dt = 64, 0.1
    E, Eh = fields(zero_field, nt, dt)
    errors = Float64[]
    for radius in (0.5, 1.5)
        rt = new_rt(2, nt)
        setstate!(rt, 1, kepler(0.0))
        setstate!(rt, 2, [0, 0.1, 0, 1000, 0, 0])
        result = ctmc_hybrid_mainloop(coulomb, rt, nt, 2, [1, 1], dt, E, Eh, 1.0;
            switch_radius=radius, abstol=1e-12, reltol=1e-10, return_stats=true)
        s = result.stats[1]
        @test result.traj_filter_flag == [false, false]
        @test result.stats[2].fixed_steps == nt-1
        @test result.stats[2].adaptive_intervals == 0
        @test s.fixed_steps > 0
        @test s.adaptive_intervals > 0
        @test s.fixed_steps+s.adaptive_intervals == nt-1
        @test s.adaptive.min_dt < dt/100
        @test s.adaptive.max_error_ratio <= 1
        err = maximum(norm(state(rt, 1, k)-kepler((k-1)*dt)) for k=1:nt)
        push!(errors, err)
        @info "Hybrid Kepler" radius err fixed_steps=s.fixed_steps adaptive_intervals=s.adaptive_intervals
    end
    @test errors[2] < errors[1]/10
    @test errors[2] < 1e-3
end

@testset "hybrid validation and work limits" begin
    rt = new_rt(1, 3)
    setstate!(rt, 1, [0, 0, 0, 1000, 0, 0])
    E, Eh = fields(zero_field, 3, 0.1)
    run(; kw...) = ctmc_hybrid_mainloop(electric_only, rt, 3, 1, [1], 0.1,
        E, Eh, 0.0; kw...)
    @test_throws ArgumentError run(switch_radius=0)
    @test_throws ArgumentError run(switch_radius=-1)
    @test_throws ArgumentError run(switch_radius=Inf)
    @test_throws ArgumentError run(switch_radius=1e-300)
    @test_throws ArgumentError run(switch_radius=1e300)
    @test_throws ArgumentError run(switch_radius=1.0, abstol=0)
    @test_throws CompositeException run(switch_radius=1.0, max_steps=1)
    @test_throws CompositeException run(switch_radius=1e6, max_steps=1)
    @test run(switch_radius=1.0) isa Vector{Bool}
    @test isempty(ctmc_hybrid_mainloop(electric_only, rt, 3, 0, Int[], 0.1,
        E, Eh, 0.0; switch_radius=1.0))
    rt.weight_cc .= 1.0
    flags = run(switch_radius=1.0)
    add_to_pmd(rt, 1, 3, 0.0, flags)
    @test rt.pmd_total_w == 1.0
end
