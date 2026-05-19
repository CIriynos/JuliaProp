import Pkg
Pkg.activate(".")
# using Revise

using JuliaProp
using Plots
using LinearAlgebra
using HDF5
using FFTW

# e_fs_list = 0.03: 0.0025: 0.1
# e_dc_list = [0.0, 0.0005, 0.005]
e_fs_list = 0.01: 0.0025: 0.1
e_dc_list = [0.0, 0.001, 0.002]
ω_fs = 0.057 * 0.5
nc = 12 / 2

for e_dc in e_dc_list
    for e_fs in e_fs_list

        example_name = "2026_5_15_hg_massive_$(e_fs)_$(e_dc)_$(ω_fs)_$(nc)_short_range"
        try
            h5open("./data/$example_name.h5", "r")
            println("Sample $example_name exists.")
        catch e
            println("Sample $example_name does not exist. Exceute a new sample.")
            
            JuliaProp.params[:E_fs] = e_fs
            JuliaProp.params[:E_dc] = e_dc
            JuliaProp.params[:ω_fs] = ω_fs
            JuliaProp.params[:nc] = nc
            JuliaProp.params[:Nr] = 40000
            include("2026_5_15_hg_massive.jl")

            pw = nothing
            rt = nothing
            GC.gc(true)
            ccall(:malloc_trim, Cint, (Csize_t,), 0)
        end

        # JuliaProp.params[:E_fs] = e_fs
        # JuliaProp.params[:E_dc] = e_dc
        # JuliaProp.params[:ω_fs] = ω_fs
        # JuliaProp.params[:nc] = nc
        # JuliaProp.params[:Nr] = 40000
        # include("2026_5_15_hg_massive.jl")

        # pw = nothing
        # rt = nothing
        # GC.gc(true)
        # ccall(:malloc_trim, Cint, (Csize_t,), 0)
    end
end