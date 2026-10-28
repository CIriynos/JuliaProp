# Load the exact legacy definition without the unrelated package dependencies.
module PulseTest
const MIN_ERROR = 1e-50
include("../src/light_pulse_stable.jl")
source = read(joinpath(@__DIR__, "../src/prefabricated.jl"), String)
body = first(split(last(split(source, "function light_pulse("; limit=2)), "function dc_bias("; limit=2))
include_string(@__MODULE__, "function light_pulse(" * body)
end

@testset "type-stable laser preserves legacy field" begin
    for shape in ("sin2", "cos2", "sin8", "cos8", "gauss", "normal", "none"), mode in ("E", "A")
        old = PulseTest.light_pulse(0.05693, 0.0533, 6, 12.0;
            pulse_shape=shape, ellipticity=0.5, phase1=pi, phase2=0.3, mode=mode)
        new = PulseTest.light_pulse_stable(0.05693, 0.0533, 6, 12.0;
            pulse_shape=shape, ellipticity=0.5, phase1=pi, phase2=0.3, mode=mode)
        @test old[4] == new[4]
        for t in (-1.0, 12.0, 100.0, new[4]/2, new[4], new[4]+1.0), j in 1:3
            @test new[j](t) ≈ old[j](t) rtol=2e-15 atol=1e-49
            @test (@inferred new[j](t)) isa Real
        end
    end
    @test_throws ArgumentError PulseTest.light_pulse_stable(0.1, 1.0, 1, 0; pulse_shape="bad")
end

@testset "stable field with hybrid integration" begin
    old = PulseTest.light_pulse(.05693, .0533, 6, 0; phase1=pi)
    new = PulseTest.light_pulse_stable(.05693, .0533, 6, 0; phase1=pi)
    oldfield = let x=old[1],y=old[2]; t -> (x(t),y(t),0.0); end
    newfield = let x=new[1],y=new[2]; t -> (x(t),y(t),0.0); end
    @test (@inferred newfield(12.0)) isa Tuple{Float64,Float64,Float64}
    nt,dt=201,0.25
    a,b=new_rt(1,nt),new_rt(1,nt)
    for rt in (a,b)
        setstate!(rt,1,[0.0,0.2,0.0,100.0,0.0,0.0])
    end
    A=ctmc_hybrid_mainloop(coulomb,a,nt,1,[1],dt,nothing,nothing,1.0;
        field=oldfield,switch_radius=1.5,return_stats=true)
    B=ctmc_hybrid_mainloop(coulomb,b,nt,1,[1],dt,nothing,nothing,1.0;
        field=newfield,switch_radius=1.5,return_stats=true)
    @test A.stats[1].adaptive_intervals == B.stats[1].adaptive_intervals == 0
    for j=1:6
        @test a.traj_data[j][1] ≈ b.traj_data[j][1] rtol=1e-13 atol=1e-13
    end
    soft(E,Z,x,y,z,rj) = -E-Z*rj/(x*x+y*y+z*z+1e-5)^1.5
    a,b=new_rt(1,21),new_rt(1,21)
    for rt in (a,b)
        setstate!(rt,1,[2.0,0.0,0.0,-1.0,0.0,0.0])
    end
    A=ctmc_hybrid_mainloop(soft,a,21,1,[1],0.1,nothing,nothing,1.0;
        field=oldfield,switch_radius=1e6,return_stats=true)
    B=ctmc_hybrid_mainloop(soft,b,21,1,[1],0.1,nothing,nothing,1.0;
        field=newfield,switch_radius=1e6,return_stats=true)
    @test A.stats[1].adaptive_intervals == B.stats[1].adaptive_intervals == 20
    for j=1:6
        @test a.traj_data[j][1] ≈ b.traj_data[j][1] rtol=1e-12 atol=1e-12
    end
end
