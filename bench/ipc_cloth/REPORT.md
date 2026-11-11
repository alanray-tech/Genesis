# 布料（Newton 求解器原生）与 CGQ 的一致性

**结论：PASS。** 三个场景（悬挂、平铺落地、倾斜落地）的每一帧单步回放都在界限 2 * velocity_tol * dt = 2e-8 m 以内，
CPU 和 GPU 都是如此；整条轨迹与 CGQ 的最大偏差 1.8e-8 m（150 帧）；落地后的静止高度与 CGQ 相差 1.3e-12 m。

Genesis：分支 `ipc-cloth`（`genesis-world-ipc-cloth`）。CGQ：`cuda-graph-qipc` main@0496c4fa（构建于
`CudaGraphQIPC/cuda-graph-qipc-bvh-bench`）。

## 一条命令

```powershell
powershell -File bench\ipc_cloth\run_all.ps1
```

先跑 CGQ（`cgq_side.py`，CGQ 自己的 venv，CUDA），把每帧的位置和速度存到 `data/`（已 gitignore），再跑 Genesis 的 CPU
和 GPU（`run_genesis.ps1` -> `genesis_side.py`，Quadrants 1.4.0 overlay，双精度），结果在 `results/genesis_{cpu,gpu}_n10.json`
（每帧回放误差和轨迹误差都在里面）。

## 设置（两边相同，`scenes.py`）

- 网格：1 m x 1 m，10 x 10 格（121 顶点，200 三角形），CGQ `cloth_drape.py` 的同一网格和同一顶点顺序。
- 坐标系用 CGQ 的：y 向上，重力 (0, -9.8, 0)，dt = 0.01，每帧一个子步。Baraff-Witkin 膜的拉伸轴由静止三角形的平面坐标系
  决定（各向异性），所以 Genesis 不旋转场景，直接在同一坐标系里跑。
- 材料：CGQ `Cloth`，E = 2e4，G = 2e3，rho = 200，单侧厚度 1e-3，strain_limit_multiplier = 100（默认），弯曲杨氏模量 1e6。
- Newton：`velocity_tol` = 1e-6（两边），max_iter 1024，line search 12 次。Genesis 双精度，`fast_math` 保持默认（开）。
- 接触（落地场景）：地面半平面 P = (0, 0, 0)，N = (0, 1, 0)；Genesis 用固定的 `gs.morphs.Plane`。d_hat = 0.01，
  resistance 1e4，friction_rate 0.05，eps_v 1e-2（都是默认值）。CGQ 关掉 adaptive kappa（`contact/adaptive_kappa_mode=off`，
  属于 P2b）。

| 场景 | 帧数 | 固定点 | 弯曲 | 接触 |
|---|---|---|---|---|
| drape | 150 | 顶点 0 和 10（前缘两角） | hinge（`cloth_drape.py` 默认） | 关 |
| ground_flat | 120 | 无 | quadratic（`Cloth` 默认） | 地面，布在 y = 0.3 水平 |
| ground_tilted | 120 | 无 | quadratic | 地面，布绕 x 轴转 30°，中心 y = 0.4 |

## 判据

- 单步回放：把 CGQ 第 k 帧的位置和速度写进 Genesis，走一步，和 CGQ 第 k+1 帧比较顶点位置的最大绝对误差；按顺序回放
  每一帧（这样某帧的摩擦对就是上一步的接触对，和 CGQ 的规则一致）。界限 2 * velocity_tol * dt = 2e-8 m：两边的求解都在
  解的 velocity_tol * dt 之内停止。
- 整条轨迹：Genesis 从第 0 帧自己跑，逐帧和 CGQ 比较（只报告，不设界限：误差会累积）。

## 结果

单步回放（最大误差，最差帧，超出界限的帧数）：

| 场景 | CPU | GPU |
|---|---|---|
| drape | 2.44e-9（帧 149），0 帧超界 | 2.44e-9（帧 149），0 帧超界 |
| ground_flat | 5.53e-10（帧 1），0 帧超界 | 5.53e-10（帧 1），0 帧超界 |
| ground_tilted | 1.46e-9（帧 29），0 帧超界 | 1.46e-9（帧 29），0 帧超界 |

整条轨迹与 CGQ 的偏差（CPU 和 GPU 打印到的位数完全相同）：

| 场景 | 帧 10 | 帧 50 | 帧 100 | 最后一帧 | 最大（帧） |
|---|---|---|---|---|---|
| drape | 1.90e-9 | 7.91e-9 | 1.80e-8 | 4.76e-9 | 1.83e-8（101） |
| ground_flat | 4.51e-9 | 3.40e-10 | 3.35e-10 | 3.38e-10 | 7.46e-9（23） |
| ground_tilted | 4.06e-10 | 2.08e-9 | 2.79e-9 | 3.02e-9 | 7.41e-9（33） |

静止高度（最后一帧最低顶点的 y）：

| 场景 | Genesis | CGQ |
|---|---|---|
| ground_flat | 0.010794561093 | 0.010794561094 |
| ground_tilted | 0.010794561093 | 0.010794561093 |

即接触厚度 0.001 加上势垒与重力平衡处的 0.009795，在 d_hat 带内，没有任何一帧越过平面或进入接触厚度。

## 用时（只报告；每步中位数，n = 10）

| 场景 | Genesis GPU | Genesis CPU | CGQ（CUDA graph） | CGQ 每帧最多 Newton 迭代 |
|---|---|---|---|---|
| drape | 10.27 ms | 20.03 ms | 4.05 ms | 10 |
| ground_flat | 3.17 ms | 3.47 ms | 1.70 ms | 8 |
| ground_tilted | 6.56 ms | 14.72 ms | 3.82 ms | 12 |

Genesis 每次 Newton 迭代从主机读回几个标志（line search、收敛），这么小的网格时间主要花在启动和同步上；没做性能优化。

## 两边不完全一样的地方

- 收敛退出：CGQ 第一次 Newton 迭代即使收敛也不退出（`converged && newton_iter > 1`），Genesis 收敛即退出；都在解的
  velocity_tol * dt 之内，回放界限覆盖了这一点（lead 同意）。
- 预条件器：CGQ 用 MAS，Genesis 用块 Jacobi；只影响迭代路径，不影响收敛到的解。
- 摩擦的切向基：Genesis 用 `gu.qd_orthogonals`，CGQ 用 x/y 轴叉乘；摩擦能量在切平面内各向同性，结果不变（P1 已核对）。
- 地面的摩擦和刚度只取布料材料的值（CGQ 默认接触行），平面的刚体材料在 P3 之前不参与。
