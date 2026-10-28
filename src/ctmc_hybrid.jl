# Fixed-step RK4 away from the core; adaptive replay for near-core intervals.
# Uses the shared validation and adaptive interval kernel in ctmc_adaptive.jl.

"""Per-trajectory hybrid diagnostics.
`fixed_steps` counts accepted full-Δt RK4 steps. `adaptive_intervals` counts
output intervals integrated adaptively. `replayed_steps` counts discarded
fixed-step trials, separately from `adaptive.rejected_steps` (error rejection).
`adaptive` describes only adaptive propagation; it does not certify fixed steps.
"""
mutable struct CTMCHybridStats
    fixed_steps::Int
    adaptive_intervals::Int
    replayed_steps::Int
    adaptive::CTMCAdaptiveStats
end

"""
    CTMCHybridWorkspace(rt; precleared=false)

可复用的出生前清理记录，绑定到 rt.traj_data，不修改 ctmc_rt。
默认保守地认为任意缓冲区可能有旧数据；首次使用清理完整出生前区域，
之后只清理上次传播新变成出生前的部分。若 rt 刚由 create_ctmc_rt 创建，
或整个 traj_data 已清零，可设 precleared=true，首次调用也不重复清理。

绑定的轨线数据若被其他积分器或调用者改写，下一次使用前须调用
reset_ctmc_hybrid_workspace!，或新建 workspace。不能并发运行使用同一 rt
的多个主循环。改变初值/权重不会使记录失效。
"""
struct CTMCHybridWorkspace
    traj_data::Vector{Vector{Vector{Float64}}}
    first_written::Vector{Int}
end

function CTMCHybridWorkspace(rt::ctmc_rt; precleared=false)
    precleared isa Bool || throw(ArgumentError("precleared must be true or false"))
    return CTMCHybridWorkspace(rt.traj_data,
        fill(precleared ? rt.t_num+1 : 1, rt.trajs_num))
end

"""Invalidate cached birth ranges after other code writes to the bound trajectories.
This does not modify trajectory data; the next hybrid call cleans conservatively.
"""
function reset_ctmc_hybrid_workspace!(workspace::CTMCHybridWorkspace)
    fill!(workspace.first_written, 1)
    return workspace
end

# Closest point on a straight segment to the origin. Used only as a switch guard,
# not as a dense trajectory representation or a continuous event locator.
@inline function _ctmc_hybrid_segment_near(ax, ay, az, bx, by, bz, radius2)
    # A segment whose endpoints share an outside coordinate half-space cannot
    # intersect the sphere. This exact broad-phase test avoids the projection
    # and division for the common far-field case; it does not weaken the guard.
    if ((ax > 0 && bx > 0) || (ax < 0 && bx < 0)) && min(abs(ax), abs(bx))^2 > radius2
        return false
    end
    if ((ay > 0 && by > 0) || (ay < 0 && by < 0)) && min(abs(ay), abs(by))^2 > radius2
        return false
    end
    if ((az > 0 && bz > 0) || (az < 0 && bz < 0)) && min(abs(az), abs(bz))^2 > radius2
        return false
    end
    dx, dy, dz = bx-ax, by-ay, bz-az
    d2 = dx*dx + dy*dy + dz*dz
    u = d2 == 0.0 ? 0.0 : clamp(-(ax*dx+ay*dy+az*dz)/d2, 0.0, 1.0)
    x, y, z = ax+u*dx, ay+u*dy, az+u*dz
    return x*x + y*y + z*z <= radius2
end

@inline function _ctmc_hybrid_guard(y, x, yy, z, radius2)
    isfinite(x) && isfinite(yy) && isfinite(z) || return true
    return _ctmc_hybrid_segment_near(y[4], y[5], y[6], x, yy, z, radius2)
end

# In the far field use the original three field samples directly, no interpolation.
@inline function _ctmc_hybrid_fields(::Nothing, E, Eh, k, Δt, t0)
    return (ntuple(j -> E[j][k], 3), ntuple(j -> Eh[j][k], 3),
            ntuple(j -> E[j][k+1], 3))
end
@inline function _ctmc_hybrid_fields(field, E, Eh, k, Δt, t0)
    return (_ctmc_adaptive_field(field, E, Eh, k, 0.0, Δt, t0),
            _ctmc_adaptive_field(field, E, Eh, k, Δt/2, Δt, t0),
            _ctmc_adaptive_field(field, E, Eh, k, Δt, Δt, t0))
end

# Returns false without advancing y if an RK stage/chord/end needs adaptive replay.
function _ctmc_hybrid_fixed_step!(out, y, F1::F, field::G, E, Eh, k, Δt, t0,
    Z, radius2, ks, tmp) where {F,G}
    e0, em, e1 = _ctmc_hybrid_fields(field, E, Eh, k, Δt, t0)
    @inbounds for stage = 1:4
        if stage == 1
            current = y
        else
            hs = stage == 4 ? Δt : 0.5*Δt
            for j = 1:6
                tmp[j] = y[j] + hs*ks[stage-1][j]
            end
            current = tmp
        end
        x, yy, z = current[4], current[5], current[6]
        stage != 1 && _ctmc_hybrid_guard(y, x, yy, z, radius2) && return false
        electric = stage == 1 ? e0 : stage == 4 ? e1 : em
        for j = 1:3
            ks[stage][j] = F1(electric[j], Z, x, yy, z, current[j+3])
            ks[stage][j+3] = current[j]
        end
    end
    @inbounds for j = 1:6
        out[j] = y[j] + (Δt/6)*(ks[1][j]+2*ks[2][j]+2*ks[3][j]+ks[4][j])
    end
    all(isfinite, out) || return false
    return !_ctmc_hybrid_guard(y, out[4], out[5], out[6], radius2)
end

function _ctmc_hybrid_procedure!(F1::F, i, rt, tid, t_num, Δt, E, Eh, Z,
    threshold, radius2, atol, rtol, dtmin, dtmax, initial_dt, max_steps,
    field::G, t0, clear_prebirth, clear_from) where {F,G}
    y = [rt.start_point_cc[j][i] for j = 1:6]
    candidate, tmp = similar(y), similar(y)
    ks = [similar(y) for _ = 1:4]
    # Allocate the additional adaptive buffers only if this trajectory needs them.
    adaptive_buffers = nothing
    for j = 1:6
        # The birth point and subsequent outputs are overwritten by propagation.
        # Preserve the old safe pre-birth semantics without clearing that suffix.
        if clear_prebirth && tid > clear_from
            fill!(@view(rt.traj_data[j][i][clear_from:tid-1]), 0.0)
        end
        rt.traj_data[j][i][tid] = y[j]
    end
    radius = _ctmc_adaptive_radius(y)
    adaptive = CTMCAdaptiveStats(0, 0, Inf, 0.0, radius, 0.0)
    stats = CTMCHybridStats(0, 0, 0, adaptive)
    filter_flag = radius < threshold
    proposed_h = min(initial_dt, dtmax, Δt)
    for k = tid:(t_num-1)
        attempts = stats.fixed_steps + stats.replayed_steps +
                   adaptive.accepted_steps + adaptive.rejected_steps
        attempts < max_steps ||
            _ctmc_adaptive_failure(i, k, Δt, NaN, "max_steps exceeded")

        # Also detect an incoming trajectory crossing the sphere between samples.
        near = _ctmc_hybrid_guard(y, y[4]+Δt*y[1], y[5]+Δt*y[2],
            y[6]+Δt*y[3], radius2)
        if !near
            if _ctmc_hybrid_fixed_step!(candidate, y, F1, field, E, Eh, k, Δt,
                t0, Z, radius2, ks, tmp)
                copyto!(y, candidate)
                stats.fixed_steps += 1
                if threshold > 0
                    filter_flag |= _ctmc_adaptive_radius(y) < threshold
                end
                proposed_h = min(initial_dt, dtmax, Δt)
            else
                stats.replayed_steps += 1
                near = true
            end
        end
        if near
            if adaptive_buffers === nothing
                adaptive_buffers = (similar(y), similar(y), similar(y))
            end
            full, half, fine = adaptive_buffers
            adaptive.min_radius = min(adaptive.min_radius, _ctmc_adaptive_radius(y))
            proposed_h = _ctmc_adaptive_interval!(y, full, half, fine, tmp, ks, F1,
                i, k, Δt, E, Eh, Z, atol, rtol, dtmin, dtmax, proposed_h,
                max_steps-stats.fixed_steps-stats.replayed_steps, field, t0, adaptive)
            stats.adaptive_intervals += 1
            filter_flag |= adaptive.min_radius < threshold
        end
        for j = 1:6
            rt.traj_data[j][i][k+1] = y[j]
        end
    end
    adaptive.accepted_steps == 0 && (adaptive.min_dt = 0.0)
    return filter_flag, stats
end

"""
    ctmc_hybrid_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt,
        E_data, E_hf_data, Z, filter_threshold=0.0;
        switch_radius, abstol=1e-9, reltol=1e-7,
        dtmin=Δt*1e-12, dtmax=Δt, initial_dt=dtmax,
        max_steps=1_000_000, field=nothing, t0=0.0, return_stats=false,
        clear_prebirth=true, workspace=nothing)

远核固定 Δt RK4，近核使用误差控制 RK4。switch_radius 为正的切换半径，
与 filter_threshold（可选的轨线筛选半径）独立，后者默认为 0 保留近核轨线。

每个输出区间先检查当前位置及匀速预测线段。若安全则试算一次完整 Δt RK4，
再检查各 RK 阶段位置、从步首到阶段/步末的线段与球的相交情况。发现靠近核心
或非有限候选结果则丢弃试步，从原状态自适应重算整个区间；下个区间重新判断。
这些几何检查是保守的切换启发式，不是连续事件定位或对任意力场的严格界。

所有结果写回同一输出网格，不插值轨线，不修改旧接口和 ctmc_rt。
远核固定步不做误差估计，abstol/reltol/dtmax 只约束自适应区间；应对 Δt、
switch_radius 和目标物理量做收敛检查。近核积分到区间末端后才可切回固定步。
max_steps 限制固定步、丢弃的固定试步及自适应接受/拒绝步骤的总数。

默认返回 Vector{Bool}。return_stats=true 返回
(traj_filter_flag=flags, stats=Vector{CTMCHybridStats})；每项含 fixed_steps、
adaptive_intervals、replayed_steps 和 adaptive::CTMCAdaptiveStats。
adaptive.max_error_ratio 仅反映自适应部分，不能证明远核固定步的精度。
无自适应步骤时其 accepted_steps/rejected_steps/min_dt/max_dt/max_error_ratio
均为 0；此时 min_radius 仅是初值半径，不是整条轨线的最小半径。
其他电场、容差、失败语义与 ctmc_adaptive_mainloop 一致。

clear_prebirth=true（默认）仅将出生前区域清零；出生点至 t_num 被积分覆盖，
t_num 以后的缓冲区保持不变。clear_prebirth=false 完全跳过清理，适用于
新建/已清零的 rt，或调用者自行管理出生前数据的情况。若复用旧轨线且出生
索引后移，应使用默认清理或自行清理新增的出生前区域，否则旧数据会污染统计。
采样场只检查本批轨线实际使用时段的有限性，数组形状仍在传播前验证。
workspace=CTMCHybridWorkspace(rt) 可记录上次写入范围，复用时避免重复清理
已知为零的前缀。新建且全零的 rt 可使用 precleared=true 构造 workspace。
其他代码改写轨线后须重置 workspace，不能通过缓存推断外部写入。
"""
function ctmc_hybrid_mainloop(F1, rt::ctmc_rt, t_num, trajs_num, tid_cc, Δt,
    E_data, E_hf_data, Z, filter_threshold=0.0;
    switch_radius, abstol=1e-9, reltol=1e-7,
    dtmin=Δt*1e-12, dtmax=Δt, initial_dt=dtmax,
    max_steps=1_000_000, field=nothing, t0=0.0, return_stats=false,
    clear_prebirth=true, workspace=nothing)
    clear_prebirth isa Bool || throw(ArgumentError("clear_prebirth must be true or false"))
    radius2 = Float64(switch_radius)^2
    isfinite(switch_radius) && switch_radius > 0 && isfinite(radius2) && radius2 > 0 ||
        throw(ArgumentError("switch_radius and its square must be finite and positive"))
    atol, rtol = _ctmc_validate_adaptive_inputs(rt, t_num, trajs_num, tid_cc, Δt,
        E_data, E_hf_data, Z, filter_threshold, abstol, reltol,
        dtmin, dtmax, initial_dt, max_steps, field, t0; active_fields_only=true)
    if workspace !== nothing
        workspace isa CTMCHybridWorkspace ||
            throw(ArgumentError("workspace must be a CTMCHybridWorkspace or nothing"))
        workspace.traj_data === rt.traj_data ||
            throw(ArgumentError("workspace is bound to a different trajectory buffer"))
        length(workspace.first_written) >= trajs_num &&
            all(x -> x >= 1, @view workspace.first_written[1:trajs_num]) ||
            throw(ArgumentError("workspace has invalid trajectory ranges"))
    end
    flags = zeros(Bool, trajs_num)
    stats = Vector{CTMCHybridStats}(undef, trajs_num)
    Threads.@threads for i = 1:trajs_num
        clear_from = workspace === nothing ? 1 : workspace.first_written[i]
        if workspace !== nothing
            # Record possible writes before propagation, so an exception cannot
            # leave the cache overlooking an earlier partially written trajectory.
            workspace.first_written[i] = min(clear_from, tid_cc[i])
        end
        flags[i], stats[i] = _ctmc_hybrid_procedure!(F1, i, rt, tid_cc[i], t_num,
            Float64(Δt), E_data, E_hf_data, Z, filter_threshold, radius2,
            atol, rtol, Float64(dtmin), Float64(min(dtmax, Δt)),
            Float64(max(initial_dt, dtmin)), max_steps, field, t0, clear_prebirth, clear_from)
        if workspace !== nothing && clear_prebirth
            workspace.first_written[i] = tid_cc[i]
        end
    end
    return return_stats ? (traj_filter_flag=flags, stats=stats) : flags
end
