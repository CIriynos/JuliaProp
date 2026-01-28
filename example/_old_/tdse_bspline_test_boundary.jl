import Pkg
Pkg.activate(".")
using Revise

using JuliaProp
using LinearAlgebra
using Plots
using BSplineKit


Nx = 1000
xmin = -50
xmax = 50
delta_t = 0.01


