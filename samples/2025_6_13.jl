import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# task_id = get_task_id_from_cmd_args()
task_id = 1

# Basic Parameters
Nr = 20000
Δr = 0.2
l_num = 120
Δt = 0.05
rmax = Nr * Δr  # 1000
Ri_tsurf = rmax * 0.8


Zeff = 0.9807
Z = 36
N = 36
c1 = 0.71483927
c2 = 0.28516073
c3 = 1.82229150
c4 = -46.64519593
c5 = 5.50450897
c6 = -3.70968729
p1 = 0
p2 = 0
p3 = 1
p4 = 2
p5 = 3
p6 = 4
b1 = 7.65590259
b2 = 1.10011313
b3 = 6.32569313
b4 = 39.52348696
b5 = 5.86328335
b6 = 5.15568993
V_eff(r) = -Zeff * (Z - N + 1 + (N - 1) * (c1 * r^p1 * exp(-b1 * r) +
    c2 * r^p2 * exp(-b2 * r) + c3 * r^p3 * exp(-b3 * r) +
    c4 * r^p4 * exp(-b4 * r) + c5 * r^p5 * exp(-b5 * r) + c6 * r^p6 * exp(-b6 * r))) / r

function po_func(r)
    R_cut = 2000
    Rs = 2500
    if r <= R_cut
        return V_eff(r)
    elseif r <= 1.25 * R_cut
        return V_eff(r) * (1 - cos(pi * (r - Rs) / (0.4 * Rs)))
    else
        return 0.0
    end
end

absorb_func = absorb_boundary_r(rmax, Ri_tsurf)


# Create Physical World & Runtime
pw = create_physics_world_sh(Nr, l_num, Δr, Δt, po_func, Z, absorb_func)
rt = create_tdse_rt_sh(pw);
plot(-pw.po_data_r .+ 1e-50, yscale=:log10)


# Initial Wave
init_wave_list = itp_fdsh(pw, rt, err=1e-8, k = 1);
crt_shwave = deepcopy(init_wave_list[1]);
en = get_energy_sh(init_wave_list[1], rt, pw.shgrid)  # He:-0.944  H:-0.5
println("Energy of ground state: ", en)


# Define the Laser. 
Ip = 0.515
ω0 = 0.05
gamma_list = [1.0, 2.0, 3.2]
gamma = gamma_list[task_id]
E0 = ω0 * sqrt(2 * Ip) / gamma
nc = 8
Tp = 2 * nc * pi / ω0

Ax(t) = (E0 / ω0) * cos(ω0 * (t - Tp / 2) / 2 / nc) ^ 8 * sin(ω0 * (t - Tp / 2)) * (t > 0 && t < Tp)
ts = 0: Δt: Tp
steps = length(ts)
At_data_x = Ax.(ts)
At_data_y = zeros(Float64, length(ts))

# # Propagation
# crt_shwave = deepcopy(init_wave_list[1])
# hhg_integral_t, phi_record, dphi_record = tdseln_sh_mainloop_record_optimized_hhg(crt_shwave, pw, rt, At_data_x, steps, Ri_tsurf);


# # Store Data Manually
# example_name = "2025_6_13_$(task_id)"
# h5open("./data/$example_name.h5", "w") do file
#     write(file, "crt_shwave", hcat(crt_shwave...))
#     write(file, "phi_record", hcat(phi_record...))
#     write(file, "dphi_record", hcat(dphi_record...))
#     write(file, "hhg_integral_t", hhg_integral_t)
# end