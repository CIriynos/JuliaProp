import Pkg
Pkg.activate(".")

using Revise
using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using Printf

N = 2000
L = 64
M = 500

v1_list = [rand(N) for i = 1: L]
v2_list = [rand(N) for i = 1: L]
D = SymTridiagonal(diagm(0 => fill(-2, N), 1 => fill(1, N - 1), -1 => fill(1, N - 1)))

function func1(v1_list, v2_list, D, L, M)
    Threads.@threads for i = 1: L
        for j = 1: M
            mul!(v2_list[i], D, v1_list[i])
            ldiv!(v1_list[i], D, v2_list[i])
        end
    end
end

function func2(v1_list, v2_list, D, L, M)
    for i = 1: L
        for j = 1: M
            mul!(v2_list[i], D, v1_list[i])
            ldiv!(v1_list[i], D, v2_list[i])
        end
    end    
end

func1(v1_list, v2_list, D, L, M)
func2(v1_list, v2_list, D, L, M)
@time func1(v1_list, v2_list, D, L, M)
@time func2(v1_list, v2_list, D, L, M)

example_name = "test_hdf5"
h5open("./data/$example_name.h5", "w") do file
    write(file, "D1", rand(N, N))
end