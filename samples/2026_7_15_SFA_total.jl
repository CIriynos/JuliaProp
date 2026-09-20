import Pkg
Pkg.activate(".")
# using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

e_fs_list = 0.01: 0.005: 0.12
ω_fs_list = [0.057, 0.057 * 0.5, 0.057 * 2.0]
nc = 6

for E_fs in e_fs_list
    for ω_fs in ω_fs_list
        example_name = "2026_7_14_$(E_fs)_$(ω_fs)_$(nc)_short_range_10"
        try
            h5open("./data/$example_name.h5", "r")
            println("Sample $example_name exists.")
        catch e
            println("Sample $example_name does not exist. Exceute a new sample.")
            
            JuliaProp.params[:E_fs] = E_fs
            # JuliaProp.params[:E_dc] = e_dc
            JuliaProp.params[:ω_fs] = ω_fs
            JuliaProp.params[:nc] = nc
            include("2026_7_14_final_version_SFA.jl")

            pw = nothing
            rt = nothing
            GC.gc(true)
            ccall(:malloc_trim, Cint, (Csize_t,), 0)
        end
    end
end