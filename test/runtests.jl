# CTMC-only tests need Julia's standard libraries, not plotting/HDF5 dependencies.
using Test
using LinearAlgebra

module CTMCCore
using LinearAlgebra
include("../src/ctmc.jl")
include("../src/ctmc_adaptive.jl")
include("../src/ctmc_hybrid.jl")
end
using .CTMCCore: create_ctmc_rt, ctmc_mainloop, ctmc_adaptive_mainloop,
    ctmc_hybrid_mainloop, add_to_pmd, calculate_asymptotic_momentum, clear_ctmc_rt

new_rt(n, nt) = create_ctmc_rt(n, nt, -5.0, 5.0, 0.1)
state(rt, i, k) = [rt.traj_data[j][i][k] for j = 1:6]
function setstate!(rt, i, y)
    for j = 1:6
        rt.start_point_cc[j][i] = y[j]
    end
end
fields(f, nt, dt; t0=0.0) = (
    [[f(t0+(k-1)*dt)[j] for k = 1:nt] for j = 1:3],
    [[f(t0+(k-0.5)*dt)[j] for k = 1:nt-1] for j = 1:3])
electric_only(E, Z, x, y, z, rj) = -E
zero_field(t) = (0.0, 0.0, 0.0)
coulomb(E, Z, x, y, z, rj) = -E-Z*rj/(x*x+y*y+z*z)^1.5

@testset "aligned output, arbitrary birth times, sampled/callback fields" begin
    dt, nt, t0 = 0.17, 21, 2.3
    field(t) = (t*t, -2t+1, 0.3)
    E, Eh = fields(field, nt, dt; t0=t0)
    tids = [1, 4, nt]
    y0 = [0.2, -0.1, 0.5, 2.0, 3.0, -1.0]
    sampled, callback = new_rt(3, nt), new_rt(3, nt)
    for rt in (sampled, callback), i = 1:3
        setstate!(rt, i, y0)
        for j = 1:6
            rt.traj_data[j][i] .= 123.0 # Detect stale data on runtime reuse.
        end
    end
    result = ctmc_adaptive_mainloop(electric_only, sampled, nt, 3, tids, dt, E, Eh, 0.0;
        t0=t0, dtmax=0.06, abstol=1e-11, reltol=1e-11, return_stats=true)
    flags = ctmc_adaptive_mainloop(electric_only, callback, nt, 3, tids, dt, nothing, nothing, 0.0;
        field=field, t0=t0, dtmax=0.06, abstol=1e-11, reltol=1e-11)
    @test flags isa Vector{Bool}
    @test flags == result.traj_filter_flag == [false, false, false]
    for i = 1:3
        birth = t0+(tids[i]-1)*dt
        for k = 1:tids[i]-1
            @test state(sampled, i, k) == zeros(6)
        end
        for k = tids[i]:nt
            s = (k-tids[i])*dt
            p = y0[1:3] - [birth^2*s+birth*s^2+s^3/3, (1-2birth)*s-s^2, 0.3s]
            r = y0[4:6] + y0[1:3]*s - [birth^2*s^2/2+birth*s^3/3+s^4/12,
                (1-2birth)*s^2/2-s^3/3, 0.15s^2]
            @test state(sampled, i, k) ≈ [p; r] atol=2e-10 rtol=2e-11
            @test state(callback, i, k) ≈ [p; r] atol=2e-10 rtol=2e-11
        end
    end
    @test result.stats[3].accepted_steps == 0
    @test result.stats[3].min_dt == 0.0
    @test result.stats[1].max_dt <= 0.06
    @test result.stats[1].max_error_ratio <= 1.0
end

# Independent analytic Kepler orbit with a=Z=1, apocenter at t=0.
function kepler(t, e=0.99)
    M = mod(pi+t, 2pi)
    lo, hi = 0.0, 2pi
    for _ = 1:70
        mid = (lo+hi)/2
        if mid-e*sin(mid) < M
            lo = mid
        else
            hi = mid
        end
    end
    u = (lo+hi)/2
    den = 1-e*cos(u)
    return [-sin(u)/den, sqrt(1-e^2)*cos(u)/den, 0.0,
        cos(u)-e, sqrt(1-e^2)*sin(u), 0.0]
end

@testset "hard recollision: analytic orbit, convergence and conservation" begin
    dt, nt = 0.1, 64
    E, Eh = fields(zero_field, nt, dt)
    fixed, loose, tight = (new_rt(1, nt) for _ = 1:3)
    for rt in (fixed, loose, tight)
        setstate!(rt, 1, kepler(0.0))
    end
    # Old interface still works, with identical argument count and flag type.
    @test ctmc_mainloop(coulomb, fixed, nt, 1, [1], dt, E, Eh, 1.0, 0.0) == [false]
    a = ctmc_adaptive_mainloop(coulomb, loose, nt, 1, [1], dt, E, Eh, 1.0;
        abstol=1e-8, reltol=1e-6, return_stats=true)
    b = ctmc_adaptive_mainloop(coulomb, tight, nt, 1, [1], dt, E, Eh, 1.0;
        abstol=1e-12, reltol=1e-10, return_stats=true)
    maxerr(rt) = maximum(norm(state(rt, 1, k)-kepler((k-1)*dt)) for k = 1:nt)
    err_fixed, err_loose, err_tight = maxerr.((fixed, loose, tight))
    @info "Kepler recollision errors" err_fixed err_loose err_tight stats=b.stats[1]
    @test err_tight < err_loose/10
    @test err_tight < 2e-5
    @test err_fixed > 100err_tight
    @test b.traj_filter_flag == [false] # Keep accurate near-core trajectories.
    @test b.stats[1].min_radius < 0.011
    @test b.stats[1].min_dt < dt/100
    @test b.stats[1].max_dt > 10b.stats[1].min_dt
    @test b.stats[1].rejected_steps > 0
    for k = 1:nt
        y = state(tight, 1, k)
        @test abs(dot(y[1:3], y[1:3])/2-1/norm(y[4:6])+0.5) < 2e-6
        @test norm(cross(y[4:6], y[1:3])-[0, 0, sqrt(1-0.99^2)]) < 1e-8
    end
end

@testset "soft-core radial crossing and independent trajectories" begin
    dt, nt = 0.1, 21
    soft(E, Z, x, y, z, rj) = -E-Z*rj/(x*x+y*y+z*z+1e-5)^1.5
    E, Eh = fields(zero_field, nt, dt)
    rt, reference = new_rt(2, nt), new_rt(1, nt)
    setstate!(rt, 1, [2, 0, 0, -1, 0, 0])
    setstate!(reference, 1, [2, 0, 0, -1, 0, 0])
    setstate!(rt, 2, [0, 0.1, 0, 100, 0, 0])
    r = ctmc_adaptive_mainloop(soft, rt, nt, 2, [1, 1], dt, E, Eh, 1.0;
        abstol=1e-11, reltol=1e-9, return_stats=true)
    ctmc_adaptive_mainloop(soft, reference, nt, 1, [1], dt, E, Eh, 1.0;
        abstol=1e-13, reltol=1e-11, dtmax=dt/4)
    @test r.traj_filter_flag == [false, false]
    @test r.stats[1].accepted_steps > r.stats[2].accepted_steps
    @test r.stats[1].min_dt < dt/1000
    @test rt.traj_data[4][1][nt] > 0
    @test state(rt, 1, nt) ≈ state(reference, 1, nt) atol=2e-6 rtol=2e-6
    # Existing downstream momentum routine accepts the same runtime and flags.
    rt.weight_cc .= 1.0
    @test length(calculate_asymptotic_momentum(rt, 2, nt, 1.0)[1]) == 2
    add_to_pmd(rt, 2, nt, 0.0, r.traj_filter_flag)
    @test rt.pmd_total_w == 2.0
    clear_ctmc_rt(rt)
    @test all(iszero, rt.traj_data[4][1])
end

@testset "optional threshold includes internal samples and final endpoint" begin
    rt = new_rt(1, 2)
    setstate!(rt, 1, [2, 0, 0, -1, 0, 0])
    E, Eh = fields(zero_field, 2, 1.0)
    flags = ctmc_adaptive_mainloop(electric_only, rt, 2, 1, [1], 1.0, E, Eh, 0.0, 0.1)
    @test flags == [true] # Endpoints ±1, accepted internal midpoint is zero.
    @test state(rt, 1, 2) ≈ [2, 0, 0, 1, 0, 0]
    setstate!(rt, 1, [-1, 0, 0, 1, 0, 0])
    @test ctmc_adaptive_mainloop(electric_only, rt, 2, 1, [1], 1.0, E, Eh, 0.0, 0.1) == [true]
end

@testset "validation and explicit integration failure" begin
    rt = new_rt(1, 3)
    E, Eh = fields(zero_field, 3, 0.1)
    run(; kw...) = ctmc_adaptive_mainloop(electric_only, rt, 3, 1, [1], 0.1, E, Eh, 0.0; kw...)
    @test_throws ArgumentError run(abstol=0)
    @test_throws ArgumentError run(reltol=-1)
    @test_throws ArgumentError run(abstol=[1e-9, 1e-9])
    @test_throws ArgumentError run(dtmin=0.2)
    @test_throws ArgumentError run(initial_dt=0)
    @test_throws ArgumentError run(max_steps=0)
    @test_throws ArgumentError ctmc_adaptive_mainloop(electric_only, rt, 3, 1, [0], 0.1, E, Eh, 0.0)
    @test_throws DimensionMismatch ctmc_adaptive_mainloop(electric_only, rt, 3, 1, [1], 0.1, E, [Float64[] for _=1:3], 0.0)
    E[1][2] = NaN
    @test_throws ArgumentError run()
    E[1][2] = 0.0
    rt.start_point_cc[1][1] = Inf
    @test_throws ArgumentError run()
    rt.start_point_cc[1][1] = 0.0
    function check_failure(f, message)
        err = try
            f()
            nothing
        catch ex
            ex
        end
        @test err isa CompositeException
        @test occursin(message, sprint(showerror, err))
        @test occursin("trajectory 1, output interval", sprint(showerror, err))
    end
    check_failure(() -> run(max_steps=1), "max_steps exceeded")
    badforce(E, Z, x, y, z, rj) = Inf
    check_failure(() -> ctmc_adaptive_mainloop(badforce, rt, 3, 1, [1], 0.1, E, Eh, 0.0;
        dtmin=0.01), "cannot satisfy tolerance at dtmin")
    oscillator(E, Z, x, y, z, rj) = -1e6*rj
    setstate!(rt, 1, [0, 0, 0, 1, 0, 0])
    check_failure(() -> ctmc_adaptive_mainloop(oscillator, rt, 3, 1, [1], 0.1, E, Eh, 0.0;
        abstol=1e-15, reltol=0.0, dtmin=0.1), "cannot satisfy tolerance at dtmin")
    # Six component tolerances, t_num=1, no trajectories.
    @test run(abstol=fill(1e-9, 6), reltol=zeros(6)) == [false]
    @test ctmc_adaptive_mainloop(electric_only, rt, 1, 1, [1], 0.1, E, [Float64[] for _=1:3], 0.0) == [false]
    @test isempty(ctmc_adaptive_mainloop(electric_only, rt, 3, 0, Int[], 0.1, E, Eh, 0.0))
end

include("ctmc_hybrid.jl")

include("light_pulse_stable.jl")
