
using LinearAlgebra


function W0(E, Ip, Z)
    F = abs(E) + 1e-50
    D = (4 * Z^3 / F)
    return (F * D^2 / (8 * pi * Z)) * exp(-2 * (2 * Ip) ^ (3/2) / (3 * F))
end

function W1(p⊥, E, Ip)
    F = abs(E) + 1e-50
    return sqrt(2 * Ip) / (pi * F) * exp(- (p⊥^2) * sqrt(2 * Ip) / F)
end

W(E, p⊥, Ip, Z) = W0(E, Ip, Z) * W1(p⊥, E, Ip)


# ============================================================
# 1. 通用 RK4 积分器
# ============================================================

function rk4(rhs!, yinit, tspan, dt, Ep, Et)
    t0, tf = tspan
    @assert tf > t0 && dt > 0

    n = ceil(Int, (tf - t0) / dt)
    h = (tf - t0) / n
    ts = collect(range(t0, tf; length=n+1))

    y = Float64.(yinit)
    Y = Matrix{Float64}(undef, length(y), n+1)
    Y[:, 1] .= y

    k1 = similar(y)
    k2 = similar(y)
    k3 = similar(y)
    k4 = similar(y)
    tmp = similar(y)

    for k in 1:n
        t = ts[k]

        rhs!(k1, y, t, Ep, Et)

        @. tmp = y + h*k1/2
        rhs!(k2, tmp, t+h/2, Ep, Et)

        @. tmp = y + h*k2/2
        rhs!(k3, tmp, t+h/2, Ep, Et)

        @. tmp = y + h*k3
        rhs!(k4, tmp, t+h, Ep, Et)

        @. y = y + h*(k1+2*k2+2*k3+k4)/6
        Y[:, k+1] .= y
    end

    return ts, Y
end


# ============================================================
# 2. 原始非线性 ODE
# 状态: [r(3); v(3)]
# ============================================================

function exact_rhs!(dy, y, t, Ep, Et)
    r2 = sum(abs2, @view y[1:3])
    r2 > 1e-12 || error("Trajectory too close to Coulomb singularity")

    invR3 = r2^(-1.5)
    ep, et = Ep(t), Et(t)

    @inbounds for i in 1:3
        dy[i] = y[3+i]
        dy[3+i] = -y[i]*invR3 -
                  (i == 1 ? ep : 0.0) -
                  (i == 2 ? et : 0.0)
    end
end


# ============================================================
# 3. 二阶微扰方程
# 状态: [r0; v0; u; du; w; dw], 每组均为3维
# ============================================================

function perturb_rhs!(dy, y, t, Ep, Et)
    r  = @view y[1:3]
    v  = @view y[4:6]
    u  = @view y[7:9]
    du = @view y[10:12]
    w  = @view y[13:15]
    dw = @view y[16:18]

    r2 = dot(r, r)
    r2 > 1e-12 || error("Reference trajectory too close to Coulomb singularity")

    invR3 = r2^(-1.5)
    invR5 = invR3 / r2
    invR7 = invR5 / r2

    ru = dot(r, u)
    rw = dot(r, w)
    uu = dot(u, u)

    # Second-order Coulomb source Q_C
    qfac = 1.5*uu*invR5 - 7.5*ru^2*invR7

    ep, et = Ep(t), Et(t)

    @inbounds for i in 1:3
        # Reference trajectory
        dy[i] = v[i]
        dy[3+i] = -r[i]*invR3 -
                  (i == 1 ? ep : 0.0)

        # First-order perturbation
        dy[6+i] = du[i]
        dy[9+i] = -u[i]*invR3 +
                   3*ru*r[i]*invR5 -
                   (i == 2 ? et : 0.0)

        # Second-order perturbation
        dy[12+i] = dw[i]
        dy[15+i] = -w[i]*invR3 +
                    3*rw*r[i]*invR5 +
                    3*ru*u[i]*invR5 +
                    qfac*r[i]
    end
end


# ============================================================
# 4. 求解二阶微扰
# ============================================================

function solve_perturbative(Ep, Et, rinit, vinit, tspan; dt=0.05)
    @assert length(rinit) == 3 && length(vinit) == 3

    yinit = vcat(rinit, vinit, zeros(12))
    t, Y = rk4(perturb_rhs!, yinit, tspan, dt, Ep, Et)

    r_ref = Y[1:3, :]
    v_ref = Y[4:6, :]

    u  = Y[7:9, :]
    du = Y[10:12, :]

    w  = Y[13:15, :]
    dw = Y[16:18, :]

    return (
        t=t,
        r_ref=r_ref, v_ref=v_ref,
        u=u, du=du,
        w=w, dw=dw,
        r_first=r_ref+u,
        v_first=v_ref+du,
        r_second=r_ref+u+w,
        v_second=v_ref+du+dw
    )
end


# ============================================================
# 5. 直接求解原始非线性 ODE
# ============================================================

function solve_exact(Ep, Et, rinit, vinit, tspan; dt=0.05)
    @assert length(rinit) == 3 && length(vinit) == 3

    yinit = vcat(rinit, vinit)
    t, Y = rk4(exact_rhs!, yinit, tspan, dt, Ep, Et)

    return (
        t=t,
        r=Y[1:3, :],
        v=Y[4:6, :]
    )
end


# ============================================================
# 6. 比较一阶、二阶微扰与原始 ODE
# ============================================================

function compare_solutions(Ep, Et, rinit, vinit, tspan; dt=0.05)
    p = solve_perturbative(Ep, Et, rinit, vinit, tspan; dt=dt)
    e = solve_exact(Ep, Et, rinit, vinit, tspan; dt=dt)

    # Final-time absolute errors
    err_r1 = norm(e.r[:, end] - p.r_first[:, end])
    err_r2 = norm(e.r[:, end] - p.r_second[:, end])

    err_v1 = norm(e.v[:, end] - p.v_first[:, end])
    err_v2 = norm(e.v[:, end] - p.v_second[:, end])

    # Maximum errors along the trajectory
    max_r2 = maximum(
        norm(e.r[:, k] - p.r_second[:, k])
        for k in eachindex(e.t)
    )

    max_v2 = maximum(
        norm(e.v[:, k] - p.v_second[:, k])
        for k in eachindex(e.t)
    )

    return (
        perturb=p,
        exact=e,
        err_r1=err_r1, err_r2=err_r2,
        err_v1=err_v1, err_v2=err_v2,
        max_r2=max_r2, max_v2=max_v2
    )
end




