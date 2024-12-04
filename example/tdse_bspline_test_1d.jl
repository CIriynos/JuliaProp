import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using LinearAlgebra
using Plots
using BSplineKit


cnt = 1000
delta_x = 0.05
delta_t = 0.01
L = cnt * delta_x
S = diagm(0 => fill(2/3, cnt), 1 => fill(1/6, cnt - 1), -1 => fill(1/6, cnt - 1)) * delta_x
D = diagm(0 => fill(-2, cnt), 1 => fill(1, cnt - 1), -1 => fill(1, cnt - 1)) * (1 / delta_x)
H = -0.5 * D

D2 = SymTridiagonal(fill(-2, cnt), fill(1, cnt - 1)) * (1.0 / delta_x ^ 2)
M2 = SymTridiagonal(fill(10, cnt), fill(1, cnt - 1)) * (-1.0 / 6.0)

# S \ H

U1 = (S + 0.5im * H * delta_t) \ (S - 0.5im * H * delta_t)
U2 = (M2 + 0.5im * D2 * delta_t) \ (M2 - 0.5im * D2 * delta_t)
U3 = (I - 0.25im * D2 * delta_t) \ (I + 0.25im * D2 * delta_t)


###########################

# B-Spline

po_func(x) = -1 / x

breakpoints = range(-L / 2, L / 2, cnt + 1)
bspline_basis_total = BSplineBasis(BSplineOrder(3), breakpoints)
b_basis = RecombinedBSplineBasis(bspline_basis_total, Derivative(0))

quadrature = Galerkin.gausslegendre(Val(100))
S_bs = galerkin_matrix(b_basis, quadrature = quadrature)
D_bs = galerkin_matrix(b_basis, (Derivative(0), Derivative(2)))
V_bs = galerkin_matrix(po_func, b_basis, (Derivative(0), Derivative(0)))
H_bs = -0.5 * D_bs
U_bs = (S_bs + 0.5im * H_bs * delta_t) \ (S_bs - 0.5im * H_bs * delta_t)


######
# propagation

x0::Float64 = 0.0
k0::Float64 = 5.0
omega_x::Float64 = 1.0

wavepkg_func(x) = @. (1.0 / (2π) ^ 0.25) * exp(1im * k0 * x) * exp(-((x - x0) / (2 * omega_x)) ^ 2)
wavepkg_func_real(x) = @. real((1.0 / (2π) ^ 0.25) * exp(1im * k0 * x) * exp(-((x - x0) / (2 * omega_x)) ^ 2))
wavepkg_func_imag(x) = @. imag((1.0 / (2π) ^ 0.25) * exp(1im * k0 * x) * exp(-((x - x0) / (2 * omega_x)) ^ 2))

xs = range(-L / 2, L / 2, cnt)

vec1 = wavepkg_func.(xs)
vec2 = copy(vec1)

vec_bs = galerkin_projection(wavepkg_func_real, b_basis) .+ im * galerkin_projection(wavepkg_func_imag, b_basis)

for i = 1: 200
    vec2 = U2 * vec2
    vec_bs = U_bs * vec_bs
end

plot(xs, [norm.(normalize(vec1)) norm.(normalize(vec2)) normalize(norm.(vec_bs))[1:cnt]], label=["Initial wave" "Classic FD" "FD with B-Spline"], title="1D TDSE Propagation In Free Space")
# plot(1: length(b_basis), norm.(vec_bs))