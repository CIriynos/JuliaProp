
using Printf
using Plots
include("./2026_10_20_ctmc_perturbation_analysis.jl")

# Laser parameters
ω0 = 0.05693
nc = 6
E0 = 0.0533
T = 2π * nc / ω0

envelope(t) = (0 <= t <= T) ? sin(ω0*t/(2*nc))^2 : 0.0
Ep(t) = E0 * envelope(t) * cos(ω0*t)

# Transverse field profile
# Et_base(t) = 0.004 * envelope(t) * sin(2*ω0*t + 0.4)
# Et_base(t) = 0.00005 * envelope(t) * sin(0.01*ω0*t + 0.4)
Et_base(t) = 0.0005

# initial conditions
t0 = T / 2
tf = T
tspan = (t0, tf)

k_ratio_list = -0.49: 0.01: 0.49
p_data_list = []
e_data_list = []
fig_Ep_list = []
compare_sol_list = [[] for α in (1.0, 0.5, 0.25, 0.125)]
ionz_rate_list = zeros(Float64, length(k_ratio_list))
energy_inf_list = zeros(Float64, length(k_ratio_list))

for (k, k_ratio) in enumerate(k_ratio_list)

println("k = $k")

# initial conditions
ti = t0 + π / ω0 * k_ratio
# pv = 0.2
pv = 0.3

rinit = [-0.5 / Ep(ti), 0, 0]
vinit = [0.0, pv, 0.0]

# plot Ep, ti
ts = 0: 0.01: T
fig_Ep_1 = plot(ts, [Ep.(ts) Et_base.(ts) .* 50])
vline!(fig_Ep_1, [ti], ls=:dashdot, label="Threshold")
push!(fig_Ep_list, fig_Ep_1)

# Single calculation
Et(t) = Et_base(t)

dt = 0.05
@time p = solve_perturbative(Ep, Et, rinit, vinit, tspan; dt=dt);
@time e = solve_exact(Ep, Et, rinit, vinit, tspan; dt=dt);

println("Exact final position: ", e.r[:, end])
println("Second-order position: ", p.r_second[:, end])

for (iα, α) in enumerate([1.0, 0.5, 0.25, 0.125])
    Et_scaled = t -> α * Et_base(t)
    c = compare_solutions(Ep, Et_scaled, rinit, vinit, tspan; dt=dt)
    @printf("%8.3f %14.5e %14.5e %14.5e %14.5e\n", α, c.err_r1, c.err_r2, c.err_v1, c.err_v2)
    push!(compare_sol_list[iα], c)
end

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

push!(p_data_list, p)
push!(e_data_list, e)
ionz_rate_list[k] = W(Ep(ti), pv, 0.5, 1.0)
energy_inf_list[k] = energy_inf

end



ti_list = t0 .+ (π / ω0) .* k_ratio_list
plot(k_ratio_list, [ionz_rate_list])
u_end = [min(sqrt((p.u[1, end] + p.w[1, end]) ^ 2 + (p.u[2, end] + p.w[2, end]).^2), 50.0) for p in p_data_list]
fig_u_end = plot(k_ratio_list, u_end)
fig_energy = plot(k_ratio_list, energy_inf_list)


k = length(k_ratio_list) ÷ 2
p = p_data_list[k]
e = e_data_list[k]
fig_curve_1 = plot(p.r_first[1, :], p.r_first[2, :],
    xlabel="x", ylabel="y", label="r_first", xlimits=(-50, 50), ylimits=(-50, 50),
    lw = 1.5)
plot!(fig_curve_1, p.r_second[1, :], p.r_second[2, :], label="r_second", lw = 1.5)
plot!(fig_curve_1, p.r_ref[1, :], p.r_ref[2, :], label="r_ref", lw = 1.5)
scatter!(fig_curve_1, e.r[1, 1:100:end], e.r[2, 1:100:end], label="real RK4", markersize=1.0, markerstrokewidth=0, markeralpha=0.5, markercolor=:black)
scatter!(fig_curve_1, [0], [0], label="(0, 0)")

# fig_curve_2 = plot(p.r_second[1, :])

# xf = p.r_second[1, end]
# yf = p.r_second[2, end]
# zf = p.r_second[3, end]
# pxf = p.v_second[1, end]
# pyf = p.v_second[2, end]
# pzf = p.v_second[3, end]
# rf = sqrt(xf^2 + yf^2 + zf^2)
# energy_inf = (pxf^2 + pyf^2 + pzf^2) / 2 - 1.0 / rf
# println("Exact final velocity: ", e.v[:, end])
# println("Second-order velocity: ", p.v_second[:, end])
# println("Final Asym energy: ", energy_inf)

fig_curve_1


for (k, k_ratio) in enumerate(k_ratio_list)
    compare_sol_list[1][k].err_r2
end

c_r2_data = [[compare_sol_list[iα][k].err_r2 for (k, k_ratio) in enumerate(k_ratio_list)] for (iα, α) in enumerate([1.0, 0.5, 0.25, 0.125])]

plot(k_ratio_list, [c_r2_data[1] c_r2_data[2] c_r2_data[3] c_r2_data[4]], yscale=:log10)