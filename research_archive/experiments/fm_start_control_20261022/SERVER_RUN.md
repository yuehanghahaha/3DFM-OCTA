# 同数据、同初始化、同预算的三组起点监督对照

服务器：`research-server.example.org`，物理 GPU0。

总目录：`/srv/oct_research/fm_start_control_20261022`。

三组均为相同3D主干、相同初始参数、相同10301固定体块、AdamW lr2e-4、2400次成功更新；每次训练更新调用一次网络。

| 目录 | 训练时间分配 | 正常推理 |
|---|---|---|
| `fm_uniform` | 2400次随机时间 | 从OCT出发Euler8 |
| `direct_t0` | 2400次固定起点 | 从OCT单步直接预测 |
| `fm_half_start` | 1200次起点＋1200次随机时间 | 从OCT出发Euler8 |

三组均使用同一速度MSE；直接监督组固定t=0时，与直接OCTA重建MSE在代数和梯度上等价。混合组通过调整采样比例加入起点监督，没有增加每次更新的网络调用数。原FM与混合组的随机时间按更新编号复用同一序列。

| 内容 | 相对于总目录的位置 |
|---|---|
| 共同初始权重 | `initial.pt` |
| 冻结训练时间序列 | `schedule.npz` |
| 输入与仅用于评价的层边界 | `inputs/`、`annotations/` |
| 每组800/1600/2400次更新权重 | 各组目录中的 `update_*.pt` |
| 每次成功更新的时间、损失与梯度 | 各组目录中的 `steps.jsonl` |
| 每200次更新的训练块评价 | 各组目录中的 `history.json` |
| 最终三维预测 | 各组目录中的 `predictions/` |
| 运行日志与状态 | `run.log`、`status.json` |
| 初始协议与等价性检查 | `protocol.json`、`preflight.json` |
| 实际训练及独立输出复核 | `final_audit.json`、`verification.json` |

此实验复用并校验上一轮冻结的主干与评价函数，其源码依赖位于 `/srv/oct_research/three_diagnostics_20261021`。依赖路径和SHA256写在 `protocol.json` 中。

本地报告入口：[验证报告](验证报告.md)。报告在本地生成；大体积权重和三维预测保留服务器。

这是单体块、单种子的机制对照。未训练患者只有两位，不作为全数据泛化性能或方法排名的证据。
