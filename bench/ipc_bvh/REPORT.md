# T0：CGQ 默认 broad phase 与上游 Genesis `bvh.py` 的对比（pair-exact）

## 结论

1. **正确性：通过。** 5 个场景、150 帧（每帧一次 Newton 迭代）、共 1848 万个活跃接触对上，用上游 `bvh.py`
   搭出的 PT / EE / PH 查询经 CGQ 的 narrow phase 后，活跃对集合与 CGQ 完全相同（按通道、全局顶点 stencil 与
   距离类型 flag 比较，每帧对称差为 0）。用两种独立方式验证：(a) 经过验证的 narrow phase 移植（`narrow_ref.py`，
   在 CGQ 自己的 1955 万个候选上 flag 与 active 位 0 误差）；(b) 直接用 CGQ 的分类器头文件编译的 CUDA kernel
   （`cgq_filter/cgq_filter.cu`）对 `bvh.py` 的候选分类。
2. **候选集也基本相同。** 移植了 CGQ 的距离规则（`candidate_reach.h`，各向同性 + 方向性）之后，`bvh.py` 的候选集
   包含 CGQ 的全部候选（每帧、每个场景都没有漏掉一个），多出的只有 14,798 个（占 CGQ 候选总数 0.08%，其中
   14,769 个在 Kimono，主要是第 0 帧布料下落时）。抽查证实这些多出的对都通过 AABB 测试而不通过 CGQ 的 fp32
   14-DOP 测试：差别来自包围体，不来自谓词。
3. **速度：CGQ 每个阶段都更快。** 每次 Newton 迭代的 broad phase + narrow phase 总时间（各帧中位数的中位数）：

   | 场景 | 三角形 | CGQ 候选 / 帧 | CGQ ms | bvh.py deferred ms | 倍数 | bvh.py inline ms | 倍数 |
   |---|---|---|---|---|---|---|---|
   | two_bunnies（小，FEM 自接触） | 4,304 | 102 | 0.362 | 1.012 | 2.8x | 1.991 | 5.5x |
   | cloth_stack（中，布-布 + 布-刚体） | 17,204 | 12,062 | 0.835 | 2.552 | 3.1x | 7.658 | 9.2x |
   | wrecking_ball（中，575 个刚体，对照） | 18,636 | 16,934 | 0.508 | 1.727 | 3.4x | 3.257 | 6.4x |
   | kimono（大，布 + Franka + 桌子） | 50,380 | 24,655 | 2.569 | 29.0 | 11.3x | 131.7 | 51.3x |
   | cloth_stack_x3（很大，60 万候选 / 帧） | 153,204 | 598,001 | 12.0 | 18.2 | 1.5x | 49.3 | 4.1x |

   - **deferred**（查询只记录原始命中，距离规则放在之后的一个 filter kernel 里，与 CGQ 生产版的 append 调度相同）
     是 `bvh.py` 的可行用法。**inline**（距离规则写在 `BoxQuery.is_kept` 里、遍历中执行）慢 2 到 5 倍：每个命中都
     在遍历线程里做 fp64 最近点计算，分支发散。
4. **差距的来源分两类：**
   - **小、中场景（2.8 到 3.4x）：启动与同步开销，不是 kernel 时间。** `bvh.py` 一次 Newton 迭代发出 39 到 125 个
     kernel 和约 14 次 host 拷贝 / 同步（nsys：two_bunnies 墙钟 1.02 ms 中 kernel 只占 0.67 ms，cloth_stack 2.41 ms
     中占 1.72 ms）。CGQ 整个 broad phase 在一个 CUDA graph 里，没有 host 往返。读回候选计数（溢出检查）本身就要
     约 0.2 ms。建树也更贵：`bvh.py` 每种图元一棵树，建三角形树和边树两棵（共 25 到 111 个 kernel，树大时用设备
     基数排序），CGQ 只建一棵三角形代理树 M（外加一棵很小的 body 树），build 2.7 到 7.0x。
   - **Kimono（11.3x）：一个 kernel 的负载不均衡。** EE 查询各帧中位 26.5 ms；最大帧上一个 EE 遍历 kernel 就占
     31.4 ms（总 34.1 ms）。第 20 帧原始 EE 命中的分布：中位数 11、p99.9 为 64、最大 28,171。最重的查询来自桌子（body 10，8 个顶点，
     最长边 0.85 m），它的盒子盖住了铺在桌上的整件和服（边长约 7 mm）。`bvh.py` 是一个线程一个查询的深度优先
     遍历，这一个线程独自走完 28k 个叶子。去重规则（哪一侧保留）无济于事，因为这些叶子都是先访问再被 `is_kept`
     拒绝的。CGQ 用一个 warp 处理一个查询（32 个 lane 共享 frontier 栈），命中的展开和过滤放到独立的、按命中均分的
     kernel 里，所以同样的几何只要 1.1 ms 遍历。
   - **cloth_stack_x3（1.5x）：两边都是真实工作量。** 每帧约 60 万候选、1300 万原始命中，CGQ 自己的 pass A 也要
     11.4 ms，固定开销被摊薄。`bvh.py` 的 AABB 加上对称的 EE 自查询产生更多原始命中（EE 1000 万，CGQ pass A 命中缓冲
     峰值 1680 万含 PT），差距主要在 EE 遍历（9.6 ms）和 EE 距离过滤（3.7 ms）。
5. **`bvh.py` 要怎样才能匹配 CGQ：**
   - **pair set：已经能表达。** 用 `bvh.py` 的公共函数（`build_bvh`、`func_bvh_query_leaves`、`BoxQuery.is_kept`）加上
     CGQ 的盒子膨胀规则、叶子谓词和距离规则，得到同一个活跃集合和几乎同一个候选集合。不需要改 `bvh.py`。
   - **速度：** 按收益排序：(1) 一 warp 一查询（或 frontier 分片）的遍历，消除大图元造成的长尾，这是 Kimono 一类
     场景（布铺在大三角形的桌子、地面上，正是第一阶段要做的 Cloth + Rigid）的必要条件；(2) 遍历与命中处理分离
     （deferred，已验证必要）；(3) 整个 broad phase 不经 host：设备端计数和溢出标志、一次 graph 或少量融合 kernel；
     (4) 单树 EF（三角形代理树 + 边查询），省掉第二棵树和对称自查询的一半命中；(5) 可选：fp32 14-DOP 包围体，
     Kimono 第 0 帧可少 20% 到 25% 候选（PT 7564 对 5662，EE 23128 对 18413）。

## 两边做的是什么

**输入。** 用 CGQ `0496c4fa` 默认配置跑每个场景（`bvh/type = ef_dop14` 等全部默认，实测值记录在
`frames.npz` 的 `bvh_config` 中），通过 contact closure 导出每帧第一次 Newton 迭代的起点位置、轨迹终点、路径膨胀、
路径速度与类型，以及拓扑、每顶点 d_hat 与厚度、接触表和忽略的刚体对。打包用 CGQ 自己的
`benchmarks/bvh_ee_dual_dataset.pack_capture`。两边读同一个 `frames.npz`。

| 场景 | 来源 | 帧 |
|---|---|---|
| two_bunnies | `benchmarks/capture_two_bunnies_bvh_frames.py` | 1 到 30 |
| cloth_stack | `examples/cloth_stack.py`（四层布落在 16 个固定方块上），本文脚本内复现 | 21 到 50 |
| wrecking_ball | `benchmarks/capture_wrecking_ball_bvh_frames.py` | 81 到 110 |
| kimono | `benchmarks/capture_kimono_bvh_frames.py`（Franka + 和服，B 样条拟合的 Cloth） | 1 到 30 |
| cloth_stack_x3 | cloth_stack 的布网格各边加密 3 倍；接触厚度按网格间距缩为 1/3（否则同一块布不相邻的图元一开始就在彼此的厚度壳内，CGQ 拒绝） | 21 到 50 |

**CGQ 一侧（真值）。** 每帧用 `scene.contact_system.recompute()` 重放导出的迭代状态。它借用生产的
`bvh_triangle_build` / `bvh_edge_build` / `trajectory_query` 子图（即 EFBroadPhaseDop14，含 body cull、ignore cull），
再用 narrow phase 的孪生 kernel 给每个候选写出 stencil、flag、d2 和 active。`verify_port.py` 检查了 recompute 的候选
集合与生产 Newton 迭代自己导出的候选集合每帧相同（5 个场景全部相同）。

**`bvh.py` 一侧。** `pipeline.py`，只用 `bvh.py` 的公共接口，不改 `bvh.py`：

- bounds kernel：三角形和边的叶子盒子、点和边的查询盒子，完全按 CGQ 对同一查询的膨胀和扫掠（起点与终点的扫掠盒，
  依次加厚度、路径膨胀、图元 d_hat；点查询加 gap `(d_hat + thickness) + inflation`；EE 再加全局 d_hat 作为两盒间隙），
  fp64 计算后向外取整到 `bvh.py` 的 fp32 盒子；
- `build_bvh`：三角形树和边树各一棵；
- PT：所有表面点查询三角形树；EE：边树自查询，每对只保留一次（低下标边作查询，与 CGQ 的方向相同，flag 才可比）；
  `BoxQuery.is_kept` 执行 CGQ 的叶子谓词：同一刚体且无自碰撞、忽略的刚体对、点是三角形顶点 / 两边共享顶点、
  PT 与 EE 各自的接触表；
- 距离规则（CGQ 谓词 7 / 5）：`candidate_reach.h` 的移植（各向同性旅行上界 + 方向性 per-side / relative 测试，
  Ericson 最近点），deferred 时在查询后的 filter kernel 里，inline 时在 `is_kept` 里；
- PH：每个表面顶点对每个半平面，CGQ 的 `halfplane_query_kernel` 规则（两边都不用 BVH）；
- narrow phase：Quadrants kernel，在设备端候选数上循环，计算 flag、平方距离和激活条件；
- 计数读回：每次迭代读回候选数（溢出检查需要）。

CGQ 用 EF（边-面）候选：边带着它拥有的点去查三角形代理树 M，一次 pass A 同时产生 PT 和 EE。`bvh.py` 每棵树只有一种
叶子，所以用两棵树、两种查询表达同一个集合；因此 CGQ 的 "PT + EE query" 只有一个数，`bvh.py` 分开报告 PT 与 EE。
CGQ 的 body cull 和 ignore cull 只裁剪不产生候选的子树，不改变集合；`bvh.py` 一侧用谓词得到同一集合，没有节点级裁剪。

## 正确性

| 场景 | 活跃集合相同（移植） | 相同（CGQ kernel 分类 `bvh.py` 候选） | CGQ 候选全部找到 | 只在 `bvh.py` 中的候选（全部帧） |
|---|---|---|---|---|
| two_bunnies | True | True | True | 0 |
| cloth_stack | True | True | True | 1 |
| wrecking_ball | True | True | True | 23 |
| kimono | True | True | True | 14,769 |
| cloth_stack_x3 | True | True | True | 5 |

- narrow phase 移植的验证（验收 2）：`verify_port.py` 对 CGQ 每帧全部候选（共 1955 万个，PT / EE / PH）计算 flag 和
  active，与 recompute 逐个比较：5 个场景 0 个 flag 不同、0 个 active 不同。torch 的每个 float64 运算单独舍入，等于
  CGQ 分类器用的不可融合运算（`nofma_*`），所以 flag 逐位相同；CGQ 用 Eigen 算的平方距离可能被 FMA 融合，只差舍入，
  实测没有一个 active 位翻转。
- CGQ 分类器 kernel（`cgq_narrow.py`）：在生产导出的候选上与 recompute 相同；在 `bvh.py` 的候选上活跃集合与 CGQ 相同，
  5 个场景全部帧。
- Quadrants 版 narrow phase（`fast_math` 开启，Genesis 默认）与 IEEE 参考在 cloth_stack 和 cloth_stack_x3 上每帧有
  8 个 PT、2 个 EE 的 flag 不同（每个场景 30 帧共 300 个），active 位没有不同。原因是 FMA 收缩打破了分类器在精确零值处的
  平局规则（平铺的布、轴对齐的方块）。移植进 Genesis 时分类器需要禁止收缩的算术。

## 计时

全部在独占 GPU lease 下测得，热启动，每帧 5 次预热、20 次重复，取每帧中位数和 p90，表中是这些值在 30 帧上的中位数。
完整的逐帧数据在 `results/frames.csv`，所有阶段的表在 `results/tables.md`。

- CGQ 的 build / query / PH / broad phase：nsys（`--cuda-graph-trace=node`）记录的 recompute 图内生产 kernel 的 GPU
  时间跨度（第一个 kernel 开始到最后一个结束，含图内节点间隙），即每次 Newton 迭代中这一阶段的代价。带不带 nsys
  的 recompute 墙钟时间相差 3% 到 7%，而且带追踪的反而略快，追踪开销在噪声以内。
- CGQ 的 narrow：`cgq_filter.cu`（CGQ 分类器 + `cipc_count_active_pt/ee_kernel` 的编排，只计数不写逐对结果），
  CUDA event。生产中它与梯度 / Hessian 装配融合在 `cipc_filter_assemble_*` 里；这里只计分类和激活，与 `bvh.py` 一侧
  的 narrow 同口径。（recompute 的 export 孪生 kernel 还算 barrier 导数，时间另列在 `frames.csv` 的
  `cgq_narrow_export_median_ms`。）
- CGQ total = broad phase 跨度 + narrow。`bvh.py` 各阶段是 GPU 同步的墙钟（`qd.sync()` 前后），total 是整段序列只在
  最后同步一次（含 narrow 和计数读回）。

deferred，毫秒，中位数（p90）：

| 场景 | 阶段 | CGQ | bvh.py | 倍数 |
|---|---|---|---|---|
| two_bunnies | build | 0.193 (0.194) | 0.513 (0.585) | 2.7x |
| two_bunnies | PT + EE query | 0.121 (0.122) | 0.342 (0.383)（PT 0.157，EE 0.188） | 2.8x |
| two_bunnies | narrow | 0.034 (0.051) | 0.063 (0.093) | 1.9x |
| two_bunnies | total | 0.362 (0.381) | 1.012 (1.151) | 2.8x |
| cloth_stack | build | 0.223 (0.225) | 0.957 (1.954) | 4.3x |
| cloth_stack | PT + EE query | 0.559 (0.563) | 1.419 (1.521)（PT 0.376，EE 1.045） | 2.5x |
| cloth_stack | narrow | 0.050 (0.051) | 0.078 (0.172) | 1.6x |
| cloth_stack | total | 0.835 (0.842) | 2.552 (3.444) | 3.1x |
| wrecking_ball | build | 0.220 (0.223) | 1.014 (2.045) | 4.6x |
| wrecking_ball | PT + EE query | 0.210 (0.211) | 0.468 (0.545)（PT 0.188，EE 0.287） | 2.2x |
| wrecking_ball | PH | 0.007 (0.007) | 0.056 (0.136) | 7.8x |
| wrecking_ball | narrow | 0.049 (0.068) | 0.074 (0.167) | 1.5x |
| wrecking_ball | total | 0.508 (0.531) | 1.727 (2.930) | 3.4x |
| kimono | build | 0.263 (0.273) | 1.583 (2.812) | 6.0x |
| kimono | PT + EE query | 2.214 (2.259) | 26.9 (27.2)（PT 0.460，EE 26.5） | 12.2x |
| kimono | PH | 0.009 (0.009) | 0.209 (0.386) | 23.6x |
| kimono | narrow | 0.049 (0.053) | 0.109 (0.244) | 2.2x |
| kimono | total | 2.569 (2.632) | 29.0 (30.0) | 11.3x |
| cloth_stack_x3 | build | 0.370 (0.376) | 2.588 (2.903) | 7.0x |
| cloth_stack_x3 | PT + EE query | 11.4 (11.9) | 15.1 (15.5)（PT 2.550，EE 12.6） | 1.3x |
| cloth_stack_x3 | narrow | 0.159 (0.169) | 0.242 (0.317) | 1.5x |
| cloth_stack_x3 | total | 12.0 (12.5) | 18.2 (18.6) | 1.5x |

`bvh.py` 的计数读回每帧另有 0.2 到 0.4 ms（已含在 total 中）。CGQ 的分类器 kernel 在 `bvh.py` 的候选上与在自己的
候选上耗时相同（候选集几乎一致），所以 narrow 的差距只来自实现（Quadrants 6 个 kernel 对 CUDA 2 个）。

**最大帧的 nsys 分解**（`results/<scene>_upstream_deferred_nsys_breakdown.json`）：

| 场景 | 墙钟 ms | kernel 合计 ms | kernel 数 | host 拷贝 / 同步 | 最重的 kernel |
|---|---|---|---|---|---|
| two_bunnies | 1.02 | 0.67 | 39 | 14 | EE 遍历 0.16 ms |
| cloth_stack | 2.41 | 1.72 | 99 | 14 | EE 遍历 0.84 ms |
| wrecking_ball | 1.51 | 0.83 | 101 | 15 | EE 遍历 0.20 ms |
| kimono | 34.1 | 33.2 | 114 | 15 | EE 遍历 31.4 ms |
| cloth_stack_x3 | 18.2 | 17.4 | 125 | 14 | EE 遍历 9.6 ms，EE 距离过滤 3.7 ms |

原始命中分布（`hit_distribution.py`，第 20 帧）：Kimono EE 每查询命中中位数 11、p99.9 为 64、最大 28,171（桌子的一条边）；
cloth_stack_x3 EE 中位数 28、p99.9 为 133、最大 3,236。

## 内存（broad phase 缓冲，MiB）

| 场景 | bvh.py deferred | 其中原始命中 | bvh.py inline | CGQ（运行结束时的分配） |
|---|---|---|---|---|
| two_bunnies | 1.6 | 0.2 | 1.4 | 约 4 |
| cloth_stack | 10.1 | 3.8 | 6.3 | 约 56 |
| wrecking_ball | 8.0 | 1.2 | 6.7 | 约 10 |
| kimono | 41.3 | 19.2 | 22.1 | 约 70 |
| cloth_stack_x3 | 216 | 151 | 66 | 约 365 |

`bvh.py` 一侧是精确值（树、叶子与查询盒、结果缓冲按全部帧的最大需求分配）。CGQ 一侧是估计：包围盒与树 M 按大小计算
（DOP14f 每个 64 B），命中、子对和候选缓冲按捕获日志里它们最后增长到的容量（`results/cgq_buffer_growth.txt`，CGQ 按
1.5 倍增长，初值由场景的 `contact/init_collision_pair_capacity` 决定，所以 cloth_stack 偏大）。

## 复现

前提：CGQ 在 `0496c4fa` 的 Release 构建（`python build.py --release`，在 CGQ 工作树），Genesis 环境带 Quadrants 1.4.0。

```powershell
# CGQ 一侧：捕获输入与真值（data/<scene>/frames.npz, gt.npz）
powershell -File bench\ipc_bvh\run_cgq.ps1 -Stage capture
# 验收 2：移植的 narrow phase 对 CGQ 候选
<cgq>\.venv\Scripts\python.exe bench\ipc_bvh\verify_port.py
# 计时（独占 GPU）：bvh.py 一侧、CGQ 一侧（nsys）、最大帧分解、汇总表
powershell -File bench\ipc_bvh\run_measure.ps1
```

`run_measure.ps1` 依次调用 `run_upstream.ps1 -All`（`bvh.py` 一侧，`results/<scene>_upstream_<schedule>.json`）、
`run_cgq.ps1 -Stage time`（CGQ 一侧，`results/<scene>_cgq.json` 与 `<scene>_cgq_narrow_deferred.json`）和
`make_report.py`（`results/frames.csv`、`summary.json`、`tables.md`）。

## 环境

- GPU：NVIDIA GeForce RTX 5090（32 GB），驱动 610.60，CUDA 12.8（nvcc 12.8.61），Windows 11，Intel Core Ultra 9 285K
- CGQ：Genesis-Embodied-AI/cuda-graph-qipc `0496c4fa`，Release，`QIPC_RUNTIME_CHECK=OFF`，torch 2.11.0+cu128
- Genesis：`cdd9eb87`（上游 main）+ 本分支的 `bench/ipc_bvh/`，Quadrants 1.4.0（`b2f658e5`），`gs.init(precision="32",
  performance_mode=True)`，torch 2.11.0+cu128

## 不确定之处

- CGQ 的阶段时间来自 recompute 图（它借用生产的 build 和 query 子图），不是 Newton 图本身；两者 kernel 相同，图内
  邻接节点不同，跨度可能略有差别。
- CGQ 的 narrow 用我编译的 kernel：分类器是 CGQ 的头文件，编排照抄 `cipc_count_active_*`，但省去了 EE 计数 kernel
  里只影响 doublet 计数的 mollifier 判断（它不决定活跃对）。
- `bvh.py` 的距离规则是移植的，并且在 Genesis 的 `fast_math` 下编译；候选集只在包围体造成的方向上有差别，说明它与
  CGQ 一致，但没有逐对验证它的判决（活跃集合的验证不依赖它）。
- 没有覆盖 codim 图元（杆、IPC 粒子：PE / PP 通道），5 个场景都没有 codim 顶点。
- cloth_stack_x3 是本文自建的场景变体（接触厚度缩小），不是 CGQ 现有的捕获。
- CGQ 的内存是估计值。
- 每帧只取第一次 Newton 迭代（与 CGQ 自己的捕获脚本一致）。
