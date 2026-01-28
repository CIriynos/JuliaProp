import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using LinearAlgebra
using Plots
using BSplineKit
using BandedMatrices

# Define Physics World
Rmin = 0.0
Rmax = 500.0
po_func(r) = -1 / r
# po_func(r) = -1.0 / r - exp(-2.0329 * r) / r - 0.3953 * exp(-6.1805 * r)

Δt = 0.1
Δt_itp = 0.05


# Spherical Harmonic
l_num = 20
po_func_with_apdix = [r -> po_func(r) + l * (l + 1) / (2 * r^2) for l = 0: l_num - 1]
lmap = create_lmmap(l_num)

# Define B-Spline Basis
N_basis = 2500
N_breakpoints = N_basis + 1
Bspline_order = 3
# bs_breakpoints = range(Rmin, Rmax, N_breakpoints)
flex(num, b) = ((Rmax - Rmin) / (N_breakpoints ^ b - 1)) * (num ^ b - 1) + Rmin
bs_breakpoints = flex.(1: N_breakpoints, 4)
bspline_basis_total = BSplineBasis(BSplineOrder(Bspline_order), copy(bs_breakpoints))
bspline_basis = RecombinedBSplineBasis(bspline_basis_total, Derivative(0))


quad = Galerkin.gausslegendre(Val(100))
S_bs = galerkin_matrix(bspline_basis, quadrature=quad)
# D2_bs = galerkin_matrix(bspline_basis, (Derivative(0), Derivative(2)), quadrature=quad)
D2_bs = -galerkin_matrix(bspline_basis, (Derivative(1), Derivative(1)), quadrature=quad)
Vl_bs = [galerkin_matrix(po_func_with_apdix[l+1], bspline_basis, quadrature=quad) for l = 0: l_num - 1]
Hl_bs = [-0.5 * D2_bs + Vl_bs[l+1] for l = 0: l_num - 1]
Ul_neg_bs = [S_bs - 0.5im * (Hl_bs[l+1]) * Δt for l = 0: l_num - 1]
Ul_pos_bs = [S_bs + 0.5im * BandedMatrix(Hl_bs[l+1]) * Δt for l = 0: l_num - 1]
Ul_neg_bs_itp = [S_bs - 0.5im * BandedMatrix(Hl_bs[l+1]) * (-im * Δt_itp) for l = 0: l_num - 1]
Ul_pos_bs_itp = [S_bs + 0.5im * BandedMatrix(Hl_bs[l+1]) * (-im * Δt_itp) for l = 0: l_num - 1]
U_bs = (S_bs + 0.5im * Hl_bs[1] * Δt) \ (S_bs - 0.5im * Hl_bs[1] * Δt)

D1_bs = galerkin_matrix(bspline_basis, (Derivative(0), Derivative(1)))
D1_bs + adjoint(D1_bs)
D2_bs - adjoint(D2_bs)


# galerkin_matrix(bspline_basis, (Derivative(1), Derivative(1)), quadrature=quad)

norm.(det(U_bs))

# Get Initial Wave
x0::Float64 = 0.0
k0::Float64 = 0.0
omega_x::Float64 = 1.0
# init_wave_func(x) = (1.0 / (2π) ^ 0.25) * exp(1im * k0 * x) * exp(-((x - x0) / (2 * omega_x)) ^ 2)
init_wave_func(x) = x * exp(-x)

initial_shwave = [zeros(ComplexF64, N_basis) for i = 1: l_num^2]
initial_shwave[1] = galerkin_projection(r->real(init_wave_func(r)), bspline_basis) .+ im * galerkin_projection(r->imag(init_wave_func(r)), bspline_basis)


# Runtime Buffer for Execution
tmp_shwave_1 = [zeros(ComplexF64, N_basis) for i = 1: l_num^2]
tmp_shwave_2 = [zeros(ComplexF64, N_basis) for i = 1: l_num^2]
tmp_shwave_3 = [zeros(ComplexF64, N_basis) for i = 1: l_num^2]


# Imaginary Time Propagation (only for Ground state)
for i = 1: 1000
    mul!(tmp_shwave_1[1], Ul_neg_bs_itp[1], initial_shwave[1])
    ldiv!(initial_shwave[1], Ul_pos_bs_itp[1], tmp_shwave_1[1])
    wave_norm = real(dot(initial_shwave[1], S_bs * initial_shwave[1]))
    initial_shwave[1] ./= sqrt(wave_norm)
    # println(real(dot(initial_shwave[1], S_bs * initial_shwave[1])))
end

A_buffer = create_pentamat(Ul_pos_bs_itp[1])
B_buffer = deepcopy(tmp_shwave_1[1])
vec = rand(length(initial_shwave[1]))
mat_pos = create_pentamat(Ul_pos_bs_itp[1])
mat_neg = create_pentamat(Ul_neg_bs_itp[1])
@time mul!(tmp_shwave_1[1], Ul_neg_bs_itp[1], vec);
@time ldiv!(tmp_shwave_1[1], Ul_pos_bs_itp[1], vec);
@time penta_mul(tmp_shwave_2[1], mat_neg, vec);
@time pentamat_elimination(tmp_shwave_2[1], mat_pos, vec, A_buffer, B_buffer);



wave_norm = dot(initial_shwave[1], S_bs * initial_shwave[1])

energy = real(dot(initial_shwave[1], (Hl_bs[1] * initial_shwave[1])))

println("H atom 1s energy = $energy")
# plot(real.(initial_shwave[1]))

# factorize(BandedMatrix(D2_bs))
