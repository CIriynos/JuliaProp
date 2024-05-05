
function coulomb_potiential_zero_fixed(;Rco::Float64 = 25.0)
    return r -> begin
        if r < Rco
            return -1.0 / r
        elseif r < 2 * Rco
            return -(2*Rco - r) / Rco^2
        else
            return 0.0
        end
    end
end

function coulomb_potiential_zero_fixed_plus(;Rco::Float64 = 25.0, a::Float64 = 3.0, b::Float64 = 2.5)
    return r -> begin
        if r <= Rco
            return -1.0 / r
        elseif r > Rco
            return -1.0 / r * exp(-((r - Rco) / (a * Rco)) ^ b)
        end
    end
end

function coulomb_potiential_helium_zero_fixed_plus(;Rco::Float64 = 25.0, a::Float64 = 3.0, b::Float64 = 2.5)
    return r -> begin
        if r <= Rco
            return (-1.0 / r - exp(-2.0329 * r) / r - 0.3953 * exp(-6.1805 * r))
        elseif r > Rco
            return (-1.0 / r - exp(-2.0329 * r) / r - 0.3953 * exp(-6.1805 * r)) * exp(-((r - Rco) / (a * Rco)) ^ b)
        end
    end
end


function absorb_boundary_r(rmax; ratio::Float64 = 0.85, pow_value::Float64 = 8.0, max_value::Float64 = 100.0)
    return r -> -im * max_value * ((r - rmax * ratio) / (rmax - rmax * ratio)) ^ pow_value * (r - rmax * ratio > 0.0)
end

function absorb_boundary_r(rmax, rbound; pow_value::Float64 = 8.0, max_value::Float64 = 100.0)
    return r -> -im * max_value * ((r - rbound) / (rmax - rbound)) ^ pow_value * (r - rbound > 0.0)
end

function create_linspace(steps, delta; zero_start::Bool = true)
    if zero_start == true
        return [(i - 1) * delta for i = 1: steps]
    end
    return [i * delta for i = 1: steps]
end

function retrieve_mat(example_name, var_name)
    var = h5open("./data/$example_name.h5", "r") do file
        read(file, var_name)
    end
    return var
end


function retrieve_obj(example_name, var_name)
    obj = h5open("./data/$example_name.h5", "r") do file
        read(file, var_name)
    end
    var = [(obj[:, i]) for i = 1: size(obj)[2]]
    return var
end