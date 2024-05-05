
"""
A simple grid in one dimension, containing 3 params (count, delta, shift)

    - `count` : N, the number of grid points
    - `delta` : Δx, the interval between two adjacent point.
    - `shift` : x0, the actual location of the first point in physics world.
"""
@kwdef struct Grid1D
    count::Int64
    delta::Float64
    shift::Float64
end


"""
GridSH represents the r-l grid, which is used when expanding a 3D wavefunction in spherical harmonics.
    
    - `rgrid` : 1D grid in r-space, the type of which is Grid1D.
    - `l_num` : The number of l. The total grid number in l-space equals l_num ^ 2.
"""
@kwdef struct GridSH
    rgrid::Grid1D
    l_num::Int64
end



# (x, y, z) -> (r, θ, ϕ)
"""
A simple function that cast cartesian coordinate to spherical coordinate.
* Input: x, y, z in cartesian coordinate system.
* Output: Tuple(r, θ, ϕ) in spherical coordinate system.

Be careful that this function is unsafe, which means it does not check the input params (x, y, z).
For example, if x was zero, the return values would be inf.
"""
# xyz_to_sphere(x, y, z) = sqrt(x^2 + y^2 + z^2), atan(sqrt(x^2 + y^2) / z), atan(y / x)

function xyz_to_sphere(x::Float64, y::Float64, z::Float64)
    r::Float64 = sqrt(x^2 + y^2 + z^2)
    theta::Float64 = 0.0
    phi::Float64 = 0.0
    if z > 0
        theta = atan(sqrt(x^2 + y^2) / z)
    elseif z < 0
        theta = pi + atan(sqrt(x^2 + y^2) / z)
    elseif z == 0.0 && x * y != 0
        theta = pi / 2
    else
        theta = NaN
    end

    if x > 0
        phi = atan(y / x)
    elseif x < 0 && y >= 0
        phi = atan(y / x) + pi
    elseif x < 0 && y < 0
        phi = atan(y / x) - pi
    elseif x == 0.0 && y > 0
        phi = pi / 2
    elseif x == 0.0 && y < 0
        phi = -pi / 2
    else
        phi = NaN
    end

    if phi < 0
        phi += 2 * pi
    end

    return r, theta, phi
end

sphere_to_xyz(r, theta, phi) = r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)

CG_coefficient(l1, m1, l2, m2, l, m) = @fastmath (-1) ^ (-l1 + l2 - m) * sqrt(2 * l + 1) * wigner3j(l1, l2, l, m1, m2, -m)

spherical_bessel_func(n, x) = sqrt(pi / (2 * x)) * besselj(n + 0.5, x)

get_length(grid::Grid1D) = grid.delta * grid.count

get_left_bound(grid::Grid1D) = grid.shift

get_right_bound(grid::Grid1D) = grid.delta * grid.count - grid.delta + grid.shift

get_linspace(grid::Grid1D) = [grid.shift + (i - 1) * grid.delta for i = 1: grid.count]

grid_index(grid::Grid1D, i::Int64) = grid.delta * i - grid.delta + grid.shift

grid_reduce(grid::Grid1D, x) = round(Int64, (x - grid.shift) / grid.delta) + 1

create_empty_shwave_array(shgrid::GridSH) = [zeros(ComplexF64, shgrid.rgrid.count) for i in range(1, shgrid.l_num * shgrid.l_num)]

create_empty_shwave_array_fixed(shgrid::GridSH) = [ones(ComplexF64, shgrid.rgrid.count) * 1e-50 for i in range(1, shgrid.l_num * shgrid.l_num)]

# modified here if evaluation is slow.
# const create_empty_shwave::Function = create_empty_shwave_array_fixed
const create_empty_shwave::Function = create_empty_shwave_array

const shwave_t = Vector{Vector{ComplexF64}}

const wave_t = Vector{ComplexF64}

function copy_shwave(to::shwave_t, from::shwave_t)
    for i in eachindex(from)
        to[i] .= from[i]
    end
end

function trimat_elimination(X, A, B, A_add, B_add)
    A_add[1] = B_add[1] = 0
    cnt = size(A)[1]
    tmp_scalar::ComplexF64 = zero(eltype(A))
    for i = 2: cnt
        @inbounds @fastmath tmp_scalar = A[i, i - 1] / (A[i - 1, i - 1] + A_add[i - 1])
        @inbounds @fastmath A_add[i] = -tmp_scalar * A[i - 1, i]
        @inbounds @fastmath B_add[i] = -tmp_scalar * (B[i - 1] + B_add[i - 1])
    end
    @inbounds @fastmath X[cnt] = (B[cnt] + B_add[cnt]) / (A[cnt, cnt] + A_add[cnt])
    for i = cnt - 1: -1: 1
        @inbounds @fastmath X[i] = (B[i] + B_add[i] - A[i, i + 1] * X[i + 1]) / (A[i, i] + A_add[i])
    end
end

# band matrix (penta-diagonal matrix)
const pentamat_t = Matrix{ComplexF64}

create_empty_pentamat(n) = zeros(ComplexF64, n, 5)

ptxid(i::Int64, j::Int64) = min(i, j)
ptyid(i::Int64, j::Int64) = j - i + 3
penta_xid(i::Int64, j::Int64) = min(i, j)
penta_yid(i::Int64, j::Int64) = j - i + 3

function create_pentamat(element_list, n)
    res = zeros(ComplexF64, n, 5)
    for i in 1: 5
        res[:, i] .= element_list[i]
    end
    return res
end

function create_identity_pentamat(n)
    return create_pentamat([0, 0, 1, 0, 0], n)
end

function create_diag_pentamat(arr, n)
    res = zeros(ComplexF64, n, 5)
    res[:, 3] = arr
    return res
end

function create_pentamat(M)
    cnt = size(M)[1]
    res = create_empty_pentamat(cnt)
    for i in 1: cnt
        for j in range(i - 2, i + 2)
            if j >= 1 && j <= cnt
                res[penta_xid(i, j), penta_yid(i, j)] = M[i, j]
            end
        end
    end
    return res
end

function pentamat_to_mat(ptmat)
    cnt = size(ptmat)[1]
    M = zeros(cnt, cnt)
    for i in 1: cnt
        for j in range(i - 2, i + 2)
            if j >= 1 && j <= cnt
                M[i, j] = ptmat[penta_xid(i, j), penta_yid(i, j)]
            end
        end
    end
    return M
end

function penta_mul(ans, pentamat, vec)
    cnt = size(pentamat)[1]
    @fastmath ans[1] = vec[1] * pentamat[penta_xid(1, 1), penta_yid(1, 1)] + 
        vec[2] * pentamat[penta_xid(1, 2), penta_yid(1, 2)] + 
        vec[3] * pentamat[penta_xid(1, 3), penta_yid(1, 3)]
    @fastmath ans[2] = vec[2 - 1] * pentamat[penta_xid(2, 2 - 1), penta_yid(2, 2 - 1)] +
        vec[2] * pentamat[penta_xid(2, 2), penta_yid(2, 2)] +
        vec[2 + 1] * pentamat[penta_xid(2, 2 + 1), penta_yid(2, 2 + 1)] +
        vec[2 + 2] * pentamat[penta_xid(2, 2 + 2), penta_yid(2, 2 + 2)]
    for i = 3: cnt - 2
        @inbounds @fastmath ans[i] = vec[i - 2] * pentamat[penta_xid(i, i - 2), penta_yid(i, i - 2)] +
            vec[i - 1] * pentamat[penta_xid(i, i - 1), penta_yid(i, i - 1)] +
            vec[i] * pentamat[penta_xid(i, i), penta_yid(i, i)] +
            vec[i + 1] * pentamat[penta_xid(i, i + 1), penta_yid(i, i + 1)] +
            vec[i + 2] * pentamat[penta_xid(i, i + 2), penta_yid(i, i + 2)]
    end
    @fastmath ans[cnt - 1] = vec[cnt - 3] * pentamat[penta_xid(cnt - 1, cnt - 3), penta_yid(cnt - 1, cnt - 3)] +
        vec[cnt - 2] * pentamat[penta_xid(cnt - 1, cnt - 2), penta_yid(cnt - 1, cnt - 2)] +
        vec[cnt - 1] * pentamat[penta_xid(cnt - 1, cnt - 1), penta_yid(cnt - 1, cnt - 1)] +
        vec[cnt] * pentamat[penta_xid(cnt - 1, cnt), penta_yid(cnt - 1, cnt)]
    @fastmath ans[cnt] = vec[cnt - 2] * pentamat[penta_xid(cnt, cnt - 2), penta_yid(cnt, cnt - 2)] +
        vec[cnt - 1] * pentamat[penta_xid(cnt, cnt - 1), penta_yid(cnt, cnt - 1)] +
        vec[cnt] * pentamat[penta_xid(cnt, cnt), penta_yid(cnt, cnt)]
end


function pentamat_elimination(X, A, B, A_buffer, B_buffer)
    
    cnt = size(A)[1]
    tmp_scalar = A[1, 1] * 0    # it is zero, but the type is belong to A's element
    A_buffer .= 0               # all clear zero
    B_buffer .= 0   

    for i = 2: cnt - 1
        for k = i: i + 1
            @inbounds @fastmath tmp_scalar = (A[ptxid(k, i - 1), ptyid(k, i - 1)] + A_buffer[ptxid(k, i - 1), ptyid(k, i - 1)]) / (A[ptxid(i - 1, i - 1), ptyid(i - 1, i - 1)] + A_buffer[ptxid(i - 1, i - 1), ptyid(i - 1, i - 1)])
            @inbounds @fastmath A_buffer[ptxid(k, i), ptyid(k, i)] += -tmp_scalar * (A[ptxid(i - 1, i), ptyid(i - 1, i)] + A_buffer[ptxid(i - 1, i), ptyid(i - 1, i)])
            @inbounds @fastmath A_buffer[ptxid(k, i + 1), ptyid(k, i + 1)] += -tmp_scalar * (A[ptxid(i - 1, i + 1), ptyid(i - 1, i + 1)] + A_buffer[ptxid(i - 1, i + 1), ptyid(i - 1, i + 1)])
            @inbounds @fastmath B_buffer[k] += -tmp_scalar * (B[i - 1] + B_buffer[i - 1])
        end
    end
    # i = cnt
    @fastmath tmp_scalar = (A[ptxid(cnt, cnt - 1), ptyid(cnt, cnt - 1)] + A_buffer[ptxid(cnt, cnt - 1), ptyid(cnt, cnt - 1)]) / (A[ptxid(cnt - 1, cnt - 1), ptyid(cnt - 1, cnt - 1)] + A_buffer[ptxid(cnt - 1, cnt - 1), ptyid(cnt - 1, cnt - 1)])
    @fastmath A_buffer[ptxid(cnt, cnt), ptyid(cnt, cnt)] += -tmp_scalar * (A[ptxid(cnt - 1, cnt), ptyid(cnt - 1, cnt)] + A_buffer[ptxid(cnt - 1, cnt), ptyid(cnt - 1, cnt)])
    @fastmath B_buffer[cnt] += -tmp_scalar * (B[cnt - 1] + B_buffer[cnt - 1])

    @fastmath X[cnt] = (B[cnt] + B_buffer[cnt]) / (A[ptxid(cnt, cnt), ptyid(cnt, cnt)] + A_buffer[ptxid(cnt, cnt), ptyid(cnt, cnt)])
    @fastmath X[cnt - 1] = (B[cnt - 1] + B_buffer[cnt - 1] - X[cnt] * (A[ptxid(cnt - 1, cnt), ptyid(cnt - 1, cnt)] + A_buffer[ptxid(cnt - 1, cnt), ptyid(cnt - 1, cnt)])) /
        (A[ptxid(cnt - 1, cnt - 1), ptyid(cnt - 1, cnt - 1)] + A_buffer[ptxid(cnt - 1, cnt - 1), ptyid(cnt - 1, cnt - 1)])

    @inbounds @fastmath for i = cnt - 2: -1: 1
        X[i] = ((B[i] + B_buffer[i]) - (X[i + 1] * (A[ptxid(i, i + 1), ptyid(i, i + 1)] + A_buffer[ptxid(i, i + 1), ptyid(i, i + 1)]) + X[i + 2] * (A[ptxid(i, i + 2), ptyid(i, i + 2)] + A_buffer[ptxid(i, i + 2), ptyid(i, i + 2)]))) / 
            (A[ptxid(i, i), ptyid(i, i)] + A_buffer[ptxid(i, i), ptyid(i, i)]) 
    end
end


function four_order_difference(X, id, dx)
    return (X[id - 2] - 8 * X[id - 1] + 8 * X[id + 1] - X[id + 2]) / (12 * dx)
end

function two_order_difference(X, id, dx)
    return -(X[id - 1] - X[id + 1]) / (2 * dx)
end

function get_derivative_two_order(X, dx)
    res = similar(X)
    len = length(res)
    res[1] = (X[2] - X[1]) / dx
    res[len] = (X[len] - X[len - 1]) / dx
    for i = 2: len - 1
        res[i] = two_order_difference(X, i, dx)
    end
    return res
end

function get_integral(X, dx)
    Y = similar(X)
    for i in eachindex(X)
        if i == 1
            Y[1] = X[1]
        else
            Y[i] = Y[i - 1] + X[i]
        end
    end
    Y .*= dx
    return Y
end

# SH

get_m_from_mm(mm::Int64) = (mm % 2 == 0) ? (mm - (mm ÷ 2) * 3) : (mm - (mm - 1) ÷ 2)

function create_lmmap(l_num::Int64)
    l::Int64 = 0
    m::Int64 = 0
    flag::Bool = false
    lmap = Int64[]
    mmap = Int64[]
    for i = 1: l_num * l_num
        flag = (l + 1) == l_num;
        push!(lmap, l)
        push!(mmap, m)
        m = flag * ((m <= 0) ? (-m + 1) : (-m)) + (!flag) * m
        l = (flag) ? (abs(m)) : (l + 1)
    end
    return (lmap, mmap)
end

# +1 (for index starts with 1)
function get_index_from_lm(l::Int64, m::Int64, l_num::Int64)
    if l < abs(m) || l < 0 || l >= l_num || abs(m) >= l_num    # invalid l and m
        return -1
    end
    lmax = l_num - 1
    mm = (m <= 0) ? (-2 * m) : (2 * m - 1)
    return l + (lmax * (mm ÷ 2) - (mm ÷ 2 - 1) * (mm ÷ 2) ÷ 2) * 2 + (lmax - mm ÷ 2) * (mm % 2) + 1
end


# save/open an object(shwave)

function save_object(filename, obj)
    path = "data/$filename.txt"
    open(path, "w+") do io
        writedlm(io, obj)
    end
end

function open_object(filename, type)
    path = "data/$filename.txt"
    readdlm(path, '\t', type, '\n')
end

function convert_shwave_to_mat(shwave)
    Nr = length(shwave[1])
    Nlm = length(shwave)
    mat = zeros(ComplexF64, Nr, Nlm)
    for i in eachindex(shwave)
        mat[:, i] .= shwave[i]
    end
    return mat
end

function convert_mat_to_shwave(mat)
    return [mat[:, i] for i in 1: size(mat)[2]]
end

function save_shwave(filename, shwave)
    mat = convert_shwave_to_mat(shwave)
    save_object(filename, mat)
end

function open_shwave(filename)
    mat = open_object(filename, ComplexF64)
    return convert_mat_to_shwave(mat)
end