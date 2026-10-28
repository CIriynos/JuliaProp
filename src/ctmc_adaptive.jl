# Independent adaptive propagator; ctmc.jl and ctmc_rt remain unchanged.

"""Per-trajectory integration diagnostics (internal steps, not output samples).
`min_radius` samples accepted half-step/end states, not the exact periapsis.
"""
mutable struct CTMCAdaptiveStats
    accepted_steps::Int
    rejected_steps::Int
    min_dt::Float64
    max_dt::Float64
    min_radius::Float64
    max_error_ratio::Float64
end

function _ctmc_adaptive_tolerance(value, name; positive)
    scalar = value isa Real
    scalar || length(value) == 6 ||
        throw(ArgumentError("$name must be a scalar or have six entries [px,py,pz,x,y,z]"))
    values = ntuple(j -> Float64(scalar ? value : value[j]), 6)
    all(v -> isfinite(v) && (positive ? v > 0 : v >= 0), values) ||
        throw(ArgumentError("$name must be finite and $(positive ? "positive" : "nonnegative")"))
    return values
end

# Quadratic Lagrange interpolation through the existing start/mid/end fields.
# τ is local to output interval k: no interpolation of trajectory states.
@inline function _ctmc_adaptive_field(::Nothing, E, Eh, k, τ, Δt, t0)
    u = τ / Δt
    a, b, c = (1-u)*(1-2u), 4u*(1-u), u*(2u-1)
    return ntuple(j -> a*E[j][k] + b*Eh[j][k] + c*E[j][k+1], 3)
end

@inline function _ctmc_adaptive_field(field, E, Eh, k, τ, Δt, t0)
    value = field(t0 + (k-1)*Δt + τ)
    length(value) == 3 || throw(ArgumentError("field(t) must return three components"))
    all(isfinite, value) || throw(ArgumentError("field(t) must return finite components"))
    return value
end

@inline function _ctmc_adaptive_rhs!(dy, y, F1, electric, Z)
    all(isfinite, y) || return false
    x, yy, z = y[4], y[5], y[6]
    for j = 1:3
        dy[j] = F1(electric[j], Z, x, yy, z, y[j+3])
        dy[j+3] = y[j]
    end
    return all(isfinite, dy)
end

function _ctmc_adaptive_step!(out, y, τ, h, F1::F, field::G, E, Eh, k, Δt, t0, Z, ks, tmp) where {F,G}
    e0 = _ctmc_adaptive_field(field, E, Eh, k, τ, Δt, t0)
    em = _ctmc_adaptive_field(field, E, Eh, k, τ+h/2, Δt, t0)
    e1 = _ctmc_adaptive_field(field, E, Eh, k, τ+h, Δt, t0)
    _ctmc_adaptive_rhs!(ks[1], y, F1, e0, Z) || return false
    for stage = 2:4
        hs = stage == 4 ? h : h/2
        for j = 1:6
            tmp[j] = y[j] + hs*ks[stage-1][j]
        end
        _ctmc_adaptive_rhs!(ks[stage], tmp, F1, stage == 4 ? e1 : em, Z) || return false
    end
    for j = 1:6
        out[j] = y[j] + (h/6)*(ks[1][j] + 2ks[2][j] + 2ks[3][j] + ks[4][j])
    end
    return all(isfinite, out)
end

@inline _ctmc_adaptive_radius(y) = hypot(y[4], y[5], y[6])

function _ctmc_adaptive_failure(i, k, h, ratio, reason)
    error("adaptive CTMC trajectory $i, output interval $k: $reason " *
          "(dt=$h, error_ratio=$ratio). Check F1/field, tolerances, dtmin and max_steps; " *
          "a point-Coulomb collision at r=0 requires regularization.")
end

# Shared interval integrator. Each call lands exactly on the next output point.
function _ctmc_adaptive_interval!(y, full, half, fine, tmp, ks, F1::F,
    i, k, Δt, E, Eh, Z, atol, rtol, dtmin, dtmax, proposed_h,
    max_steps, field::G, t0, stats) where {F,G}
    τ = 0.0
    while τ < Δt
        stats.accepted_steps + stats.rejected_steps < max_steps ||
            _ctmc_adaptive_failure(i, k, proposed_h, NaN, "max_steps exceeded")
        remaining = Δt - τ
        h = min(proposed_h, remaining)
        τ + h/2 > τ ||
            _ctmc_adaptive_failure(i, k, h, NaN, "time step cannot advance in Float64")

        # One full RK4 step and two half steps; the fine result is order 4.
        ok = _ctmc_adaptive_step!(full, y, τ, h, F1, field, E, Eh, k, Δt, t0, Z, ks, tmp) &&
             _ctmc_adaptive_step!(half, y, τ, h/2, F1, field, E, Eh, k, Δt, t0, Z, ks, tmp) &&
             _ctmc_adaptive_step!(fine, half, τ+h/2, h/2, F1, field, E, Eh, k, Δt, t0, Z, ks, tmp)
        ratio = Inf
        if ok
            ratio = 0.0
            for j = 1:6
                scale = atol[j] + rtol[j]*max(abs(y[j]), abs(fine[j]))
                ratio = max(ratio, abs((fine[j]-full[j])/15)/scale)
            end
            isfinite(ratio) || (ratio = Inf)
        end

        if ratio <= 1.0
            stats.accepted_steps += 1
            stats.min_dt = min(stats.min_dt, h)
            stats.max_dt = max(stats.max_dt, h)
            stats.max_error_ratio = max(stats.max_error_ratio, ratio)
            stats.min_radius = min(stats.min_radius,
                _ctmc_adaptive_radius(half), _ctmc_adaptive_radius(fine))
            copyto!(y, fine)
            # Land exactly on each output boundary, without accumulated drift.
            τ = h == remaining ? Δt : τ + h
            factor = ratio == 0.0 ? 5.0 : clamp(0.9*ratio^(-0.2), 0.2, 5.0)
            proposed_h = clamp(h*factor, dtmin, dtmax)
        else
            stats.rejected_steps += 1
            h > dtmin ||
                _ctmc_adaptive_failure(i, k, h, ratio, "cannot satisfy tolerance at dtmin")
            factor = isfinite(ratio) ? clamp(0.9*ratio^(-0.2), 0.1, 0.5) : 0.1
            proposed_h = max(dtmin, h*factor)
        end
    end
    return proposed_h
end

# Parameters and dimensions are validated once by the public mainloop.
function _ctmc_adaptive_procedure!(F1::F, i, rt, tid, t_num, Δt, E, Eh, Z,
    threshold, atol, rtol, dtmin, dtmax, initial_dt, max_steps, field::G, t0) where {F,G}
    y = [rt.start_point_cc[j][i] for j = 1:6]
    full, half, fine, tmp = (similar(y) for _ = 1:4)
    ks = [similar(y) for _ = 1:4]
    # Clear also the pre-birth region when the runtime is reused.
    for j = 1:6
        fill!(rt.traj_data[j][i], 0.0)
        rt.traj_data[j][i][tid] = y[j]
    end
    radius = _ctmc_adaptive_radius(y)
    stats = CTMCAdaptiveStats(0, 0, Inf, 0.0, radius, 0.0)
    filter_flag = radius < threshold
    proposed_h = min(initial_dt, dtmax, Δt)
    for k = tid:(t_num-1)
        proposed_h = _ctmc_adaptive_interval!(y, full, half, fine, tmp, ks, F1,
            i, k, Δt, E, Eh, Z, atol, rtol, dtmin, dtmax, proposed_h,
            max_steps, field, t0, stats)
        filter_flag |= stats.min_radius < threshold
        for j = 1:6
            rt.traj_data[j][i][k+1] = y[j]
        end
    end
    stats.accepted_steps == 0 && (stats.min_dt = 0.0)
    return filter_flag, stats
end

# Shared validation for the adaptive and hybrid public interfaces.
function _ctmc_validate_adaptive_inputs(rt, t_num, trajs_num, tid_cc, Δt,
    E_data, E_hf_data, Z, filter_threshold, abstol, reltol,
    dtmin, dtmax, initial_dt, max_steps, field, t0; active_fields_only=false)
    t_num isa Integer && 1 <= t_num <= rt.t_num ||
        throw(ArgumentError("t_num must be in 1:rt.t_num"))
    trajs_num isa Integer && 0 <= trajs_num <= rt.trajs_num ||
        throw(ArgumentError("trajs_num must be in 0:rt.trajs_num"))
    isfinite(Δt) && Δt > 0 || throw(ArgumentError("Δt must be finite and positive"))
    isfinite(Z) || throw(ArgumentError("Z must be finite"))
    isfinite(t0) && isfinite(t0+(t_num-1)*Δt) || throw(ArgumentError("time range must be finite"))
    isfinite(filter_threshold) && filter_threshold >= 0 ||
        throw(ArgumentError("filter_threshold must be finite and nonnegative"))
    all(v -> isfinite(v) && v > 0, (dtmin, dtmax, initial_dt)) ||
        throw(ArgumentError("dtmin, dtmax and initial_dt must be finite and positive"))
    dtmin <= min(dtmax, Δt) || throw(ArgumentError("dtmin must not exceed min(dtmax, Δt)"))
    max_steps isa Integer && max_steps > 0 || throw(ArgumentError("max_steps must be a positive integer"))
    atol = _ctmc_adaptive_tolerance(abstol, "abstol"; positive=true)
    rtol = _ctmc_adaptive_tolerance(reltol, "reltol"; positive=false)
    length(tid_cc) >= trajs_num || throw(DimensionMismatch("tid_cc is too short"))
    all(i -> tid_cc[i] isa Integer && 1 <= tid_cc[i] <= t_num, 1:trajs_num) ||
        throw(ArgumentError("tid_cc must contain integer indices in 1:t_num"))
    length(rt.traj_data) == length(rt.start_point_cc) == 6 ||
        throw(DimensionMismatch("runtime must contain six state components"))
    for j = 1:6
        length(rt.traj_data[j]) >= trajs_num && length(rt.start_point_cc[j]) >= trajs_num ||
            throw(DimensionMismatch("runtime has too few trajectories"))
        for i = 1:trajs_num
            length(rt.traj_data[j][i]) >= t_num || throw(DimensionMismatch("runtime time buffer is too short"))
            isfinite(rt.start_point_cc[j][i]) || throw(ArgumentError("initial state must be finite"))
        end
    end
    if field === nothing
        first_field = 1
        if active_fields_only
            first_field = trajs_num == 0 ? t_num+1 : minimum(@view tid_cc[1:trajs_num])
            # A trajectory born at the last output point performs no integration.
            first_field == t_num && (first_field = t_num+1)
        end
        for (data, n) in ((E_data, t_num), (E_hf_data, t_num-1))
            length(data) == 3 && all(v -> length(v) >= n, data) ||
                throw(DimensionMismatch("fields must have three components and enough time samples"))
            if first_field <= n
                all(v -> all(isfinite, @view(v[first_field:n])), data) ||
                    throw(ArgumentError("field samples must be finite in the active time range"))
            end
        end
    end
    return atol, rtol
end

"""
    ctmc_adaptive_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt,
        E_data, E_hf_data, Z, filter_threshold=0.0;
        abstol=1e-9, reltol=1e-7, dtmin=Δt*1e-12, dtmax=Δt,
        initial_dt=dtmax, max_steps=1_000_000, field=nothing, t0=0.0,
        return_stats=false)

独立的变步长 RK4 主循环，沿用 ctmc_mainloop 的位置参数与 ctmc_rt。
每条轨线独立以一步/两个半步的差值除以 15 估计细步解的局部误差；
六分量的最大归一化误差不大于 1 才接受两个半步的结果。
abstol、reltol 可为标量或按 [px,py,pz,x,y,z] 排列的六分量容差。
这是局部误差控制，长期全局误差仍需通过收紧容差做收敛检查。

Δt 是输出间隔，内部步长不超过 min(dtmax, Δt)。各输出区间末端缩短步长，
直接将积分结果写入 rt.traj_data[j][i][k]，对应 t0+(k-1)*Δt；不插值轨线。
tid_cc 之前置零，tid_cc 处保持原初值。权重、初值和累计 PMD/HHG 不变。

默认用每个输出区间的整步/半步/下一整步场作二次插值。此时误差控制针对
插值后的场，不包含场采样误差。推荐 field=t -> (Ex(t),Ey(t),Ez(t))，
以直接计算子步场；提供 field 时 E_data/E_hf_data 被忽略，可传 nothing。
field/F1 应无副作用且可在线程间安全调用。

默认返回 Vector{Bool}，可继续传给 add_to_pmd/add_to_hhg。默认 threshold=0
保留近核轨线；显式传入正阈值仍会标记接受的内部半步/端点处进入阈值的轨线，
但不停止传播。这是离散采样检测，并非精确的连续距离事件定位。
return_stats=true 返回 (traj_filter_flag=flags, stats=各轨线诊断)。

遇到步长下限、不可表示的时间步、非有限状态或步数上限且无法满足容差时抛错，
不静默筛除轨线、不强制接受误差超限的步骤。线程中错误由 Julia 包装传播；
失败后的 rt 可能含部分结果，不应继续用于统计。点库仑 r=0 奇点需另行正则化。
"""
function ctmc_adaptive_mainloop(F1, rt::ctmc_rt, t_num, trajs_num, tid_cc, Δt,
    E_data, E_hf_data, Z, filter_threshold=0.0;
    abstol=1e-9, reltol=1e-7, dtmin=Δt*1e-12, dtmax=Δt,
    initial_dt=dtmax, max_steps=1_000_000, field=nothing, t0=0.0,
    return_stats=false)
    atol, rtol = _ctmc_validate_adaptive_inputs(rt, t_num, trajs_num, tid_cc, Δt,
        E_data, E_hf_data, Z, filter_threshold, abstol, reltol,
        dtmin, dtmax, initial_dt, max_steps, field, t0)

    flags = zeros(Bool, trajs_num)  # Vector, not packed BitVector: thread-safe writes.
    stats = Vector{CTMCAdaptiveStats}(undef, trajs_num)
    Threads.@threads for i = 1:trajs_num
        flags[i], stats[i] = _ctmc_adaptive_procedure!(F1, i, rt, tid_cc[i], t_num,
            Float64(Δt), E_data, E_hf_data, Z, filter_threshold, atol, rtol,
            Float64(dtmin), Float64(min(dtmax, Δt)), Float64(max(initial_dt, dtmin)),
            max_steps, field, t0)
    end
    return return_stats ? (traj_filter_flag=flags, stats=stats) : flags
end
