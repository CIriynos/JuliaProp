import Pkg
Pkg.activate(".")

using Revise
using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using Printf
using FFTW
using DSP
using SpecialFunctions
println("Number of Threads: $(Threads.nthreads())")

function bell_polynomial(m::Int, k::Int, a::Vector)
    if m == 0 && k == 0
        return 1
    elseif k == 0 || k > m
        return 0
    end

    t = m - k + 1
    if length(a) < t
        error("a must have at least $(t) elements")
    end

    solutions = generate_solutions(m, k, t)
    m_fact = factorial(m)
    total = 0

    for sol in solutions
        denom = 1
        product_a = 1
        for i in 1:t
            ji = sol[i]
            denom *= factorial(ji)
            product_a *= (a[i] / factorial(i))^ji
        end
        total += m_fact / denom * product_a
    end

    return total
end

function generate_solutions(m::Int, k::Int, t::Int)
    solutions = Vector{Vector{Int}}()
    current = zeros(Int, t)

    function backtrack(i::Int, rem_k::Int, rem_m::Int)
        if i == 0
            if rem_k == 0 && rem_m == 0
                push!(solutions, copy(current))
            end
            return
        end
        max_j = min(rem_k, rem_m ÷ i)
        for j in 0:max_j
            current[i] = j
            backtrack(i-1, rem_k - j, rem_m - i*j)
        end
        current[i] = 0  # 回溯
    end

    backtrack(t, k, m)
    return solutions
end

a = [1, 2, 3, 4]  # 对应 a₁=1, a₂=2, a₃=3, a₄=4
result = bell_polynomial(5, 2, a)
println(result)  # 输出 5 * 1 * 4 + 10 * 2 * 3 = 20 + 60 = 80


# ChatGPT's Taylor Method with exp
F0 = 0.05
Fs = 0.01: 0.001: 0.05

# 计算 ln(W(F)) 关于 F 在 F0=0.03 附近的泰勒展开系数
A0 = log(4) - log(F0) - 2 / (3 * F0)
A1 = -1 / F0 + 2 / (3 * F0 ^ 2)
A2 = 0.5 * (1 / F0^2 - 4 / (3 * F0^3))
A3 = (-2/F0^3 + 4/F0^4) / 6

function Cm(m, A_list)
    cm = 0.0
    for k = 1: m
        cm += (1 / (factorial(k))) * bell_polynomial(m, k, A_list[2:(m-k+1)+1])
    end
    cm *= exp(A_list[1]) # A0
    return cm
end

C1 = Cm(1, [A0, A1, A2, A3])
C2 = Cm(1, [A0, A1, A2, A3])
C3 = Cm(1, [A0, A1, A2, A3])

