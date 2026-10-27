
using Printf
using Plots
include("./2026_10_20_ctmc_perturbation_analysis.jl")

# Laser parameters
ω0 = 0.057
nc = 8
E0 = 0.05
T = 2π * nc / ω0

envelope(t) = (0 <= t <= T) ? sin(ω0*t/(2*nc))^2 : 0.0
Ep(t) = E0 * envelope(t) * cos(ω0*t)

# Transverse field profile
# Et_base(t) = 0.004 * envelope(t) * sin(2*ω0*t + 0.4)
Et_base(t) = 0.0004 * envelope(t) * sin(0.1*ω0*t + 0.4)


# initial conditions
t0 = T / 2 + π / ω0 * 0.3
tf = T
tspan = (t0, tf)

rinit = [-0.5 / Ep(t0), 0, 0]
vinit = [0.0, 0.02, 0.0]

# plot Ep, t0
ts = 0: 0.01: T
fig_Ep_1 = plot(ts, [Ep.(ts) Et_base.(ts) .* 50])
vline!(fig_Ep_1, [t0], ls=:dashdot, label="Threshold") 


# Single calculation
Et(t) = Et_base(t)

dt = 0.05
@time p = solve_perturbative(Ep, Et, rinit, vinit, tspan; dt=dt);
@time e = solve_exact(Ep, Et, rinit, vinit, tspan; dt=dt);

println("Exact final position: ", e.r[:, end])
println("Second-order position: ", p.r_second[:, end])

# fig_curve_1 = plot(p.r_first[1, :], p.r_first[2, :], p.r_first[3, :],
#     xlabel="x", ylabel="y", zlabel="z", label="r_first",
#     lw = 1.5, xlimits=(-40, 40), ylimits=(-40, 40), zlimits=(-40, 40))
# plot!(fig_curve_1, p.r_second[1, :], p.r_second[2, :], p.r_second[3, :], label="r_second", lw = 1.5)
# plot!(fig_curve_1, p.r_ref[1, :], p.r_ref[2, :], p.r_ref[3, :], label="r_ref", lw = 1.5)
# scatter!(fig_curve_1, e.r[1, 1:100:end], e.r[2, 1:100:end], e.r[3, 1:100:end], label="real RK4", markersize=1.0, markerstrokewidth=0, markeralpha=0.5, markercolor=:black)
# scatter!(fig_curve_1, [0], [0], [0], label="(0, 0, 0)")

fig_curve_1 = plot(p.r_first[1, :], p.r_first[2, :],
    xlabel="x", ylabel="y", label="r_first",
    lw = 1.5)
plot!(fig_curve_1, p.r_second[1, :], p.r_second[2, :], label="r_second", lw = 1.5)
plot!(fig_curve_1, p.r_ref[1, :], p.r_ref[2, :], label="r_ref", lw = 1.5)
scatter!(fig_curve_1, e.r[1, 1:100:end], e.r[2, 1:100:end], label="real RK4", markersize=1.0, markerstrokewidth=0, markeralpha=0.5, markercolor=:black)
scatter!(fig_curve_1, [0], [0], label="(0, 0)")

fig_curve_2 = plot(p.r_second[1, :])

xf = p.r_second[1, end]
yf = p.r_second[2, end]
zf = p.r_second[3, end]
pxf = p.v_second[1, end]
pyf = p.v_second[2, end]
pzf = p.v_second[3, end]
rf = sqrt(xf^2 + yf^2 + zf^2)
energy_inf = (pxf^2 + pyf^2 + pzf^2) / 2 - 1.0 / rf

println("Exact final velocity: ", e.v[:, end])
println("Second-order velocity: ", p.v_second[:, end])
println("Final Asym energy: ", energy_inf)

# Amplitude scaling test
println("\nAmplitude scaling test")
@printf("%8s %14s %14s %14s %14s\n",
        "alpha", "err_r1", "err_r2", "err_v1", "err_v2")

for α in (1.0, 0.5, 0.25, 0.125)
    Et_scaled = t -> α * Et_base(t)

    c = compare_solutions(
        Ep, Et_scaled, rinit, vinit, tspan; dt=dt
    )

    @printf("%8.3f %14.5e %14.5e %14.5e %14.5e\n",
        α, c.err_r1, c.err_r2, c.err_v1, c.err_v2)
end