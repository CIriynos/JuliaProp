tau_lst = [tau_fs - 2pi/ω2; range(tau_fs - 2pi/ω2 + tau_fs, tau_fs + nc1*2pi/ω1 - tau_fs, 14); tau_fs + nc1*2pi/ω1]
p1 = plot(tau_lst .- mid_point, shg_yields,
    xlabel="Delay Time τ",
    ylabel="SHG Yield (arb.u.)",
    title="SHG Variation at 3.75 THz (0.01ω)",
    label="TDSE SHG",
    guidefont=Plots.font(14, "Times"),
    tickfont=Plots.font(14, "Times"),
    titlefont=Plots.font(18, "Times"),
    legendfont=Plots.font(10),
    linewidth=2,
    margin = 5 * Plots.mm)
plot!(p1, tau_lst .- mid_point, (ss) .* 9.1, label="CTMC SHG", linewidth=2)
plot!(p1, tau_lst .- mid_point, thz_int .* 0.062 .- 0.000, label="original THz", ls=:dash)


tau_fs = 500
tau_lst = [tau_fs - 2pi/ω2, tau_fs + nc1*pi/ω1 - 1.5pi/ω2,
    tau_fs + nc1*pi/ω1 - pi/ω2, tau_fs + nc1*pi/ω1 - 0.5pi/ω2, tau_fs + nc1*2pi/ω1, 0.0]
# scatter!(p1, [mid_point] .- mid_point,[0.0031], label="")

l = [0.0024, 0.0003, 0.0031, 0.00468, 0.0025]
scatter!(p1, tau_lst[1:5] .- mid_point, l, label="samples in Fig.1(a)", legend=:bottomright)
mid_point_shift = mid_point - (-4600)
p1