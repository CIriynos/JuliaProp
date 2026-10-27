# Run from any directory:
# julia --startup-file=no --threads=4 samples/ctmc_hybrid_benchmark.jl
# Uses only standard libraries and the actual CTMC source files.
using LinearAlgebra
using Statistics

module CTMCBenchmarkCore
using LinearAlgebra
include("../src/ctmc.jl")
include("../src/ctmc_adaptive.jl")
include("../src/ctmc_hybrid.jl")
end

function ctmc_benchmark_measure(f; repeats=7)
    f() # Compile/warm up this exact call signature.
    times, bytes = Float64[], Int[]
    for _ = 1:repeats
        GC.gc()
        result = @timed f()
        push!(times, result.time)
        push!(bytes, result.bytes)
    end
    return (seconds=median(times), bytes=median(bytes))
end

function ctmc_hybrid_benchmark(; trajs_num=32, t_num=4001, Δt=0.1,
    switch_radius=1.0, repeats=7, mixed=false)
    C = CTMCBenchmarkCore
    fixed, adaptive, hybrid = (
        C.create_ctmc_rt(trajs_num, t_num, -1.0, 1.0, 0.1) for _ = 1:3)
    for rt in (fixed, adaptive, hybrid), i = 1:trajs_num
        rt.start_point_cc[4][i] = 1000.0+i
        rt.start_point_cc[2][i] = 0.1
    end
    if mixed
        # One hard-return orbit plus the far-field ensemble.
        # Use t_num=64 to cover one Kepler period, including periapsis.
        e = 0.99
        for rt in (fixed, adaptive, hybrid)
            rt.start_point_cc[4][1] = -(1+e)
            rt.start_point_cc[2][1] = -sqrt((1-e)/(1+e))
        end
    end
    E = [zeros(t_num) for _ = 1:3]
    Eh = [zeros(t_num-1) for _ = 1:3]
    tids = ones(Int, trajs_num)
    force(E, Z, x, y, z, rj) = -E-Z*rj/(x*x+y*y+z*z)^1.5
    oldrun() = C.ctmc_mainloop(force, fixed, t_num, trajs_num, tids, Δt, E, Eh, 1.0, 0.0)
    adaptrun() = C.ctmc_adaptive_mainloop(force, adaptive, t_num, trajs_num, tids, Δt,
        E, Eh, 1.0; abstol=1e-12, reltol=1e-10, return_stats=true)
    hybridrun() = C.ctmc_hybrid_mainloop(force, hybrid, t_num, trajs_num, tids, Δt,
        E, Eh, 1.0; switch_radius=switch_radius, abstol=1e-12, reltol=1e-10,
        return_stats=true)
    a = ctmc_benchmark_measure(oldrun; repeats=repeats)
    b = ctmc_benchmark_measure(adaptrun; repeats=repeats)
    c = ctmc_benchmark_measure(hybridrun; repeats=repeats)
    result = hybridrun()
    println((mixed=mixed, threads=Threads.nthreads(), trajectories=trajs_num,
        output_intervals=t_num-1, switch_radius=switch_radius))
    println((fixed=a, adaptive=b, hybrid=c, hybrid_vs_fixed=c.seconds/a.seconds,
        speedup_vs_adaptive=b.seconds/c.seconds))
    println((fixed_steps=sum(s.fixed_steps for s in result.stats),
        adaptive_intervals=sum(s.adaptive_intervals for s in result.stats),
        replayed_steps=sum(s.replayed_steps for s in result.stats)))
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    ctmc_hybrid_benchmark()
    ctmc_hybrid_benchmark(t_num=64, trajs_num=128, mixed=true)
end
