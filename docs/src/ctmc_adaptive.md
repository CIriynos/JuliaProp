# CTMC 变步长积分

新增 `src/ctmc_adaptive.jl`，沿用原有 `ctmc_rt`。`src/ctmc.jl`、旧函数接口、结构体、旧 samples 均未修改；`src/JuliaProp.jl` 只增加新文件的 include 和新接口的 export。没有新增包依赖。

正常 `using JuliaProp` 即可使用新接口。若原来直接 `include("src/ctmc.jl")`，则在其后另加 `include("src/ctmc_adaptive.jl")`。

## 在现有 sample 中使用

保留初值生成、`update_start_point`、数据分析代码，只替换主循环：

```julia
traj_filter_flag = ctmc_adaptive_mainloop(
    F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, 0.0;
    abstol=1e-9, reltol=1e-7)

add_to_pmd(rt, trajs_num, t_num, Z, traj_filter_flag)
add_to_hhg(rt, trajs_num, t_num, Δt, tid_cc, traj_filter_flag)
```

**最后一个位置参数设为 `0.0`，以保留近核轨线。** 复制旧的正 `filter_threshold` 会继续标记近核轨线，随后旧分析程序仍会排除它们。省略此参数时新接口默认为 `0.0`。它只用于标记，完全不参与步长控制；被标记的轨线仍继续传播。

默认返回与原函数一致的 `Vector{Bool}`。对通常不产生数值失败、且禁用距离筛选的运行，所有标志均为 `false`。束缚态的识别仍由 `calculate_asymptotic_momentum` 单独处理。

若要查看每条轨线的步数、最小步长和接近核心的程度：

```julia
result = ctmc_adaptive_mainloop(
    F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, 0.0;
    abstol=[1e-10, 1e-10, 1e-10, 1e-9, 1e-9, 1e-9],
    reltol=1e-8, return_stats=true)
traj_filter_flag = result.traj_filter_flag
stats = result.stats
println(stats[1].accepted_steps, " ", stats[1].rejected_steps)
println(stats[1].min_dt, " ", stats[1].max_dt)
```

容差排列为 `[px, py, pz, x, y, z]`，也可全部使用标量。`abstol` 每项须为正数，`reltol` 每项可为零。诊断结构 `CTMCAdaptiveStats` 还包含 `min_radius`（初值及接受的半步/端点中最小半径）与 `max_error_ratio`（接受步骤的最大归一化误差估计）。`min_radius` 不是精确近心点距离。若出生时刻就是最后一个输出点，步数与 `min_dt/max_dt` 均为零。

可运行 `samples/ctmc_adaptive_example.jl` 查看两条轨线分别采用不同内部步长的示例。

## 误差控制和时间对齐

对每个候选内部步长 `h`，从同一状态分别做一次 RK4 整步和两次 RK4 半步，计算

```math
\epsilon = \max_{j=1,\ldots,6}
\frac{|(y_{h/2,h/2,j}-y_{h,j})/15|}
{\mathrm{abstol}_j+\mathrm{reltol}_j\max(|y_{n,j}|,|y_{h/2,h/2,j}|)}.
```

仅当 `ε ≤ 1` 时接受两个半步的结果；失败步骤不改变当前状态或输出数据。按 `0.9 ε^(-1/5)` 调整下一步长度，并限制缩放幅度。这里不做 Richardson 外推，接受解仍是四阶 RK4。一步加倍约需三次 RK4 的计算量；近核处增加子步，不为整个轨线集合统一缩小步长。

`Δt` 现在是共同的**输出间隔**。每条轨线独立推进，每个区间末端将内部步长截断到剩余长度，直接积分到输出时刻，再写入原有的 `rt.traj_data[j][i][k]`：

```math
t_k=t_0+(k-1)\Delta t.
```

不对轨线状态做插值，也不改变数组维度。出生索引 `tid_cc[i]` 和初值定义不变，出生以前的数据清零；线程只写各自的轨线缓冲区。初值、权重、已累积的 PMD/HHG 均不修改。

可选参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `abstol`, `reltol` | `1e-9`, `1e-7` | 六分量的局部误差控制容差 |
| `dtmin` | `Δt*1e-12` | 内部自适应步长下限 |
| `dtmax` | `Δt` | 步长上限，实际还受输出间隔限制 |
| `initial_dt` | `dtmax` | 各轨线首个候选步长，限制在允许范围内 |
| `max_steps` | `1_000_000` | 每条轨线接受和拒绝步骤的总数上限 |
| `field` | `nothing` | 连续电场函数 `field(t) -> (Ex,Ey,Ez)` |
| `t0` | `0.0` | 首个输出网格点的绝对时间 |
| `return_stats` | `false` | 是否额外返回逐轨线诊断 |

输出边界的最后一个余量可以小于 `dtmin`，但同样必须通过误差检查，不能继续细分到下限以下来强行通过。浮点数无法表示更小时间增量、达到步数上限、或在步长下限仍不能满足容差时抛错，错误消息包含轨线号和输出区间号。`Threads.@threads` 会将错误包装成 `CompositeException`。失败后 `rt` 可能只包含部分传播结果，不应继续用于统计。

## 子步电场与精度边界

原程序只提供 `E_data[j][k]` 和 `E_hf_data[j][k]`。新接口默认在每个输出区间内，用整步/半步/下一整步三个值作二次 Lagrange 插值，能准确复现原来的阶段采样值，且无需更改现有 sample 的场准备过程。

步长细分无法恢复场采样丢失的信息。若已知连续激光函数，推荐直接传入：

```julia
traj_filter_flag = ctmc_adaptive_mainloop(
    F1, rt, t_num, trajs_num, tid_cc, Δt, Et_data, Et_data_hf, Z, 0.0;
    field=t -> (Ex_fs(t), Ey_fs(t)+E_applied(t), 0.0),
    abstol=1e-10, reltol=1e-8)
```

传入 `field` 后，`E_data/E_hf_data` 不参与积分，可以传 `nothing`；如仍需通过 `update_start_point` 生成初值，则保留该函数所需的场数组。`t0` 必须与数组对应的时间原点一致。`F1` 和 `field` 应无副作用且支持多线程调用。非有限的场值会报错；RK 阶段出现非有限状态或导数时先拒绝该步骤、减小步长。

容差控制的是积分器的**局部误差估计**，不保证全局误差小于同一数值。对具体物理量，应收紧容差、减小 `dtmax` 并检查收敛。距离筛选标志只检查接受步骤的半步与端点，不是连续事件定位。

变步长能解析有限近心距离的硬回碰，也能积分现有 samples 的软核势。它不改变 `F1` 的物理模型：samples 中 `+1e-5` 是软核参数，不能通过收紧积分容差消除其对近核动力学的影响。严格点库仑势下正面撞击 `r=0` 是方程的奇点，需要另行采用正则化等方法。

保留原输出间隔意味着高频信号仍受采样率限制。内部子步更精确，不代表输出网格足以解析极短的加速度脉冲；研究硬回碰 HHG 时还须减小输出 `Δt` 并检查频谱收敛。

## 原有实现的建议（本次未修改）

1. `RK4_procedure` 只检查每一步开始处的半径，漏掉最后一个输出点，也无法识别两网格点之间的近核运动。新积分器改为检查初值、接受的内部半步与端点，但也不声称精确定位阈值穿越事件。
2. 旧 `RK4_procedure` 在复用 runtime 时不清除 `tid_cc[i]` 以前的数据。出生时刻改变时可能残留上一批轨线；旧实现应在复用前清理对应轨线缓冲区。新积分器自行清理。
3. `combine_trajs_data` 和 `traj_get_acc` 收到 `tid_cc/tid` 却未使用，直接对包含出生前零值的整段动量求差分。非零初始动量会在出生附近引入跳变，硬回碰脉冲也可能因输出采样稀疏而失真。建议另外提供按出生索引限制有效区间、并直接调用力函数计算加速度的分析接口；本次保留旧实现。
4. `calculate_asymptotic_momentum` 使用无场点库仑势的无穷远映射，而 samples 的积分力常含软核参数。应在外场结束且远离软核区时使用，并验证进一步延长传播时间的收敛性；近核或尚在外场中时直接映射可能产生模型不一致。

## 验证

运行 `julia --startup-file=no --threads=4 test/runtests.jl`，仅需标准库。测试直接加载实际的 CTMC 源文件，避免加载与积分器无关的绘图和 HDF5 包；这不是运行全部历史 samples。

覆盖解析时变电场、不同出生时刻的输出对齐、两种电场接口、runtime 复用、`a=Z=1, e=0.99` 的解析开普勒近核轨道、容差收敛、能量/角动量守恒、软核径向穿越、各轨线独立自适应、旧主循环和 PMD 接口、可选距离筛选及显式失败路径。

在 Julia 1.10.10 的单线程和四线程下，均通过 279 项断言。开普勒测试使用输出间隔 `Δt=0.1`，比较所有输出点的六分量状态与解析解的欧氏距离：

| 积分方式 | 最大状态误差 |
| --- | --- |
| 原固定步长 RK4 | `5.82e1` |
| 自适应，`abstol=1e-8, reltol=1e-6` | `4.05e-4` |
| 自适应，`abstol=1e-12, reltol=1e-10` | `1.80e-7` |

严格容差运行接受 635 步、拒绝 2 步，内部步长范围约为 `7.27e-6` 至 `8.03e-2`，采样最小半径约为 `0.010000025`。这些数值是该测试轨道的结果，不是对所有轨线的全局精度保证。
