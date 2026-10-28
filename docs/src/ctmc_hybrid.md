# CTMC 远核固定步、近核自适应混合积分

新增 `src/ctmc_hybrid.jl`，主接口为 `ctmc_hybrid_mainloop`。远核输出区间只做一次原步长 `Δt` 的 RK4；近核区间使用与 `ctmc_adaptive_mainloop` 相同的误差控制积分。`ctmc.jl`、`ctmc_rt` 和旧接口均不修改；全程自适应接口也保持兼容。

## 调用

```julia
result = ctmc_hybrid_mainloop(
    F1, rt, t_num, trajs_num, tid_cc, Δt,
    Et_data, Et_data_hf, Z, 0.0;
    switch_radius=1.5,
    abstol=1e-10, reltol=1e-8,
    return_stats=true)

traj_filter_flag = result.traj_filter_flag
add_to_pmd(rt, trajs_num, t_num, Z, traj_filter_flag)
add_to_hhg(rt, trajs_num, t_num, Δt, tid_cc, traj_filter_flag)
```

`switch_radius` 必须显式指定，单位与轨线坐标一致。这里的 `1.5` 仅用于展示调用，并非适合任意势和初值的推荐常数。

**`switch_radius` 与位置参数 `filter_threshold` 的用途不同**：前者决定在哪里提高计算精度，后者决定哪些轨线在下游统计中被排除。上例的 `0.0` 关闭距离筛选，保留准确计算的近核轨线。若只需原来的 `Vector{Bool}` 返回值，省略 `return_stats=true`。

保留现有的初值与权重生成流程即可。可选的 `field=t -> (Ex(t),Ey(t),Ez(t))`、`t0`、六分量容差以及失败处理，均与全程自适应接口一致。提供连续场时，场数组可以传 `nothing`；默认远核直接读取整步/半步采样，近核子步使用二次插值。

`dtmax`、`initial_dt` 和容差只控制自适应区间。远核仍使用原输出间隔 `Δt`，即使 `dtmax < Δt`。每条轨线独立选择模式，所有数据仍写入 `t[k]=t0+(k-1)Δt` 的统一网格。

通过 `using JuliaProp` 可直接使用；若手动 include，则按 `ctmc.jl`、`ctmc_adaptive.jl`、`ctmc_hybrid.jl` 的顺序加载。示例见 `samples/ctmc_hybrid_example.jl`。

## 为什么不能只检查步首半径

若轨线速度很高，可能一步从球的一侧穿到另一侧，步首和步末都在球外。如果只检查 `r[k] < switch_radius`，可能完全漏掉回碰。

本实现采用以下切换过程：

1. 在输出区间开始时，检查当前位置及 `r → r+Δt*p` 的匀速预测线段是否进入切换球。若是，直接自适应推进整个区间。
2. 否则试算一次完整 `Δt` 的 RK4。在每个中间阶段检查阶段位置及从步首到该位置的线段；最终还检查步末位置及首末连线。
3. 若检测到接近核心，或候选结果不是有限值，丢弃固定试步，从原状态自适应重算整个区间。固定试步不写入最终轨线，也不改变当前状态。
4. 自适应计算完成这个区间后，在下个输出时刻重新判断是否可恢复固定步。

这些检查不需要额外求力，远核区间仍只计算 4 次方程右端，即 12 次标量 `F1` 调用；全程自适应版本即使不缩步，也需要 12 次方程右端、36 次标量 `F1` 调用。

几何线段检查是切换启发式，不是对任意强弯曲轨道的严格界，也不是精确的阈值穿越事件定位。它可能提前启动自适应，这会增加计算量；不能据此证明没有任何漏检。应通过增大切换半径、减小 `Δt` 及与全程自适应结果比较来验证适用范围。

## 诊断量

`result.stats[i]` 是 `CTMCHybridStats`：

| 字段 | 含义 |
| --- | --- |
| `fixed_steps` | 接受的完整 `Δt` 固定步数量 |
| `adaptive_intervals` | 用自适应积分完成的输出区间数量 |
| `replayed_steps` | 因几何检查或非有限结果而丢弃的固定试步数量 |
| `adaptive` | 自适应部分的 `CTMCAdaptiveStats` |

成功运行应满足 `fixed_steps + adaptive_intervals == t_num - tid_cc[i]`。`max_steps` 限制固定接受步、丢弃的固定试步以及自适应接受/拒绝步骤的总数。

`replayed_steps` 与 `adaptive.rejected_steps` 不同：前者是固定试步触发切换，后者是自适应误差估计不通过。

`adaptive.max_error_ratio` **仅约束自适应部分，不能证明固定步或整条轨线满足同一容差**。无自适应区间时，它等于 0，表示没有估计误差，不表示轨线误差为 0。此时 `adaptive.min_dt/max_dt` 也为 0，`adaptive.min_radius` 仅记录初始半径。存在自适应区间时，后者覆盖初值及自适应区间的步首、接受的半步与端点，仍不覆盖远核固定区间内部。

自适应统计的 `min_dt` 指每个接受步骤推进的总时长 `h`；该步骤接受的结果实际由两个 `h/2` 的 RK4 子步构成。

## 如何选择切换半径

切换半径应覆盖原步长开始失准的区域，不能只根据数值奇点本身来选择。在库仑问题中，快速掠过时间尺度约为 `r/|p|`，受力显著变化的动力学尺度约为 `sqrt(r^3/Z)`；当它们与 `Δt` 可比时，应考虑提前进入自适应区。此估计依赖当前速度、核电荷、软核参数和外场，无法给出通用半径。

建议固定一批初值，比较逐渐增大的切换半径，例如 `R, 2R, 4R`，同时检查减小 `Δt` 后末态动量、散射角、回碰特征和目标谱是否稳定。仅收紧近核容差无法消除进入近核区以前已经积累的固定步误差。也应选取回碰轨线与全程自适应结果作对照。

解析开普勒测试 `a=Z=1, e=0.99, Δt=0.1` 中，保持近核容差 `abstol=1e-12, reltol=1e-10`：

| 切换半径 | 固定输出区间 | 自适应输出区间 | 最大六维状态误差 |
| --- | --- | --- | --- |
| `0.5` | 58 | 5 | `2.22e-4` |
| `1.5` | 37 | 26 | `1.03e-6` |

增大半径可以降低固定步误差，但会增加自适应区间。该结果也说明近核容差不是整条混合轨线的全局误差保证。点库仑正面撞击 `r=0` 仍需正则化；高频 HHG 仍受输出采样率限制。

## 速度验证

可在自己的计算环境运行：

```bash
julia --startup-file=no --threads=1 samples/ctmc_hybrid_benchmark.jl
julia --startup-file=no --threads=4 samples/ctmc_hybrid_benchmark.jl
```

基准仅需标准库，直接加载实际 CTMC 源文件；先预热后统计多次运行的中位数，排除首次编译。包含全远核轨线以及一条硬回碰轨线混合大量远核轨线的情况。

在本运行环境的单线程基准中：

| 情况 | 原固定 RK4 | 全程自适应 | 混合积分 | 相对全程自适应加速 |
| --- | --- | --- | --- | --- |
| 32 条远核轨线，各 4000 个区间 | 42.3 ms | 137.2 ms | 42.6 ms | 3.22 倍 |
| 128 条轨线中一条硬回碰，各 63 个区间 | 2.98 ms | 8.95 ms | 3.89 ms | 2.30 倍 |

速度取决于力函数、电场实现、近核区间比例及线程调度。这里的原固定 RK4 仅作耗时参考，不能作为硬回碰的精确解。数值测试还检查远核结果与原 RK4 完全一致、全近核结果与全程自适应完全一致，以及穿越检测、回退重算、时间对齐和切换半径收敛。

`julia --startup-file=no --threads=1 test/runtests.jl` 与四线程运行均通过 373 项断言，包含原有 279 项测试和新增混合积分测试。未运行全部历史 samples 或依赖完整绘图环境的测试。


## 减少重复清理（性能优化）

这里的轨线缓冲区是原有 `rt.traj_data`，不是 RK4 额外要求保存的一份历史。
默认现在只清零出生前区域；出生点到 `t_num` 会被传播覆盖，不再提前清零，
`t_num` 以后的数据保持不变。场的有限值检查也只扫描本批实际使用的时段。
几何检查新增同侧坐标排除，远核时可免去线段投影除法；穿越保护未关闭。

对于新建的全零 runtime，推荐创建一次可复用的清理记录：

```julia
rt = create_ctmc_rt(trajs_num, t_num, p_min, p_max, p_delta)
workspace = CTMCHybridWorkspace(rt; precleared=true)
# 后续批次复用同一 rt 和 workspace；update_start_point 不影响该记录。
flags = ctmc_hybrid_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt,
    Et_data, Et_data_hf, Z, 0.0; switch_radius=1.0, workspace=workspace)
```

这个 workspace 不复制轨线，只保存对原数据的引用和每条轨线上次的写入起点。
出生索引改变时，仅清理上次轨线新落入出生前的部分。它不用于 RK4 计算，
也不改变积分精度。`precleared=true` 只可用于确实全零的数据；若无法保证，
使用默认的 `CTMCHybridWorkspace(rt)`，首次调用会保守清理，之后仍可复用记录。

若其他积分器或调用者直接改写了轨线，须先
`reset_ctmc_hybrid_workspace!(workspace)`，使下一次调用保守清理。
workspace 不能用于另一个 rt，也不能用于同时写同一 rt 的多个主循环。
失败传播会保守记录可能已写入的范围，避免下一次遗漏清理。

还可设 `clear_prebirth=false` 完全跳过清理，行为接近原固定步程序。
这适用于调用者自行保证出生前数据正确的情形；复用旧数据时不能盲目关闭。
只要默认清理仍必须处理一大段未知旧数据，就不应承诺与从不清理的旧函数等时。
对新建/已清零数据使用上述 workspace，则没有这部分多余成本。

基准脚本现在覆盖有库仑力、无库仑力，以及网格 60001 点但出生后仅传播
100 步的情况。均预热后比较；`use_workspace=false` 可查看保守清理的成本。
不同设备、线程及出生分布会影响耗时，不能保证每次计时严格相同。

本次优化后的单线程与四线程核心测试均通过 420 项断言。

## 连续激光场的分配优化

`light_pulse` 内部捕获的变量可能以 `Core.Box` 保存。即使在外层用 `let` 捕获其返回函数，也可能在每次连续场调用时产生大量分配。新增 `light_pulse_stable`，保留相同的位置参数、关键字和返回值；原 `light_pulse` 保持不变。

例如，按原脚本的定义替换两处生成函数，并在外层捕获实际的场函数：

```julia
Ex_fs, Ey_fs, Ez_fs, _ = light_pulse_stable(ω_fs, E_fs, nc, tau_fs;
    ellipticity=0.0, phase1=0.5pi+cep, phase2=cep)
E_thz_1, E_thz_2, E_thz_3, tmax_thz = light_pulse_stable(
    ω_thz, E_thz, nc_thz, tau_thz; pulse_shape="sin2", phase1=0.5pi)

field_thz = let ex=Ex_fs, ey=Ey_fs, ethz=E_thz_1
    t -> (ex(t), ey(t)+ethz(t), 0.0)
end
field_zero = let ex=Ex_fs, ey=Ey_fs
    t -> (ex(t), ey(t), 0.0)
end

result = ctmc_hybrid_mainloop(F1, rt, t_num, trajs_num, tid_cc, Δt,
    Et_data, Et_data_hf, Z, 0.0; field=field_thz, switch_radius=1.5,
    return_stats=true, workspace=workspace)
```

场数组及初值应使用同一组场函数重新生成。若函数在全局作用域中定义，组合场时也要使用 `let`；不要在闭包中继续调用访问非 `const` 全局变量的 `E_applied`。闭包若调用其他类型不稳定的函数，仍可能分配内存；可使用 `@code_warntype field_thz(1.0)` 检查。

在 Julia 1.10.10、单线程、预热后的 3496 个远核固定步测试中，原 `light_pulse` 连续场约 12 ms、8.2 MiB；新接口约 1.43 ms、3.8 KiB。实际耗时随激光函数、线程和轨线而变化。两者在支持的七种包络和 `E`/`A` 模式下进行了数值对照；近核区间和远核区间的轨线对照也通过。全程自适应仍有计算开销，输入容差和切换半径应做收敛检查。
