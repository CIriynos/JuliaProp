import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using LinearAlgebra
using Plots
using BSplineKit

# cnt = 8

# M = diagm(1 => fill(1.0, cnt - 1), -1 => fill(-1.0, cnt - 1))
# α = 0.05
# mat_list = [zeros(cnt, cnt) for i = 1: cnt - 1]
# for i = 1: cnt - 1
#     mat_list[i][i:i+1, i:i+1] = M[i:i+1, i:i+1]
# end


# # update_strategy = [1: 2: cnt - 2; 2: 2: cnt - 2]
# update_strategy = 1: cnt - 2

# H1 = exp(α * M)
# H2 = I
# for i = update_strategy
#     H2 = H2 * exp(0.5α * mat_list[i])
# end
# H2 *= exp(α * mat_list[cnt - 1])
# for i = reverse(update_strategy)
#     H2 = H2 * exp(0.5α * mat_list[i])
# end

# sum(H1 .- H2)



# α = 0.05
# cnt = 5
# P = [rand(cnt, cnt) for i = 1: 10]

# P_sum = sum(P)
# mat1 = exp(α * P_sum)

# mat2 = I
# for i = 1: cnt - 1
#     mat2 *= exp(0.5α * P[i])
# end
# mat2 *= exp(α * P[cnt])
# for i = cnt - 1: 1
#     mat2 *= exp(0.5α * P[i])
# end
# mat1 .- mat2



# cnt = 50
# α = 0.05
# delta_x = 0.2
# D = diagm(0 => fill(-2, cnt), 1 => fill(1, cnt - 1), -1 => fill(1, cnt - 1)) * (1 / delta_x)
# U1 = exp(-im * D * α)

# U2 = (I + 0.5im * α * D) \ (I - 0.5im * α * D) 

# norm.(U1 - U2)

# <l1m1|l2m2|l3m3> 
function get_SH_integral(l1, m1, l2, m2, l3, m3)
    if abs(-m1) > l1 || abs(m2) > l2 || abs(m3) > l3
        return 0.0
    end
    if -m1 + m2 != -m3
        return 0.0
    end
    if l3 > l1 + l2 || l3 < abs(l1 - l2)
        return 0.0
    end
    return (-1)^m1 * sqrt((2l1 + 1) * (2l2 + 1) * (2l3 + 1) / 4pi) * wigner3j(l1, l2, l3, 0, 0, 0) * wigner3j(l1, l2, l3, -m1, m2, m3)
end


Nl = 5
l1 = 1
m1 = 0
for l2 = 0: Nl - 1
    for m2 = -l1: l1
        a = get_SH_integral(l1, m1, 1, -1, l2, m2) - get_SH_integral(l1, m1, 1, 1, l2, m2)
        println("l2=$l2, m2=$m2, a=$a")
    end
end


# Nl = 5
# l1 = 1
# m1 = 0
# for l2 = 0: Nl - 1
#     for m2 = -l1: l1
#         a = sqrt(4pi / 3) * get_SH_integral(l1, m1, 1, 0, l2, m2)
#         println("l2=$l2, m2=$m2, a=$a")
#     end
# end

# clm(l, m) = sqrt(((l + 1)^2 - m^2) / ((2l + 1) * (2l + 3)))

# clm(1, 0)
# sqrt(4pi / 3) * get_SH_integral(1, 0, 1, 0, 2, 0)