# 公开代码 FlowOCT + 三维模块累计删减

日期：2026-10-22。针对前轮三维生成效果差，增加作者公开实现对照、累计移除模块、同预算延长训练，以及实际损失梯度诊断。

## 主实验

| 编号 | 方法 | 备注 |
|---|---|---|
| 00 | 公开代码 FlowOCT | 作者二维网络、二维 Haar、四项损失与 Adam；同一冻结输入 |
| 01 | 完整三维模型 | 3D Haar + FM + RDCA + 分层投影 + 结构加权 + 重建 |
| 02 | 01 去 RDCA | 第一次移除 |
| 03 | 02 再去分层投影 | 累计移除 RDCA、投影 |
| 04 | 03 再去结构加权 | 累计移除 RDCA、投影、结构加权 |
| 05 | 04 再去重建 | 仅剩三维小波 FM |
| 06 | 05 再去小波 | 仅剩体素三维 FM |
| 07 | 完整三维模型＋50% 起点采样 | 模块均保留，只改变时间采样，单独列为机制对照 |

每组从头训练 **2,400 次成功更新**，主实验共 19,200 次。每 400 次评价，800/1600/2400 次同时保存权重和预测。固定 4 位训练患者＋4 位验证患者，每例两个 `[256,64,64]` 体块；每个训练体块最终恰好使用 300 次。病例与像素数据沿用上一轮，不按模型分数重新选择。

这是累计删减路径，不能把相邻差异当成脱离其他模块的独立主效应。前轮单独去模块的结果保留在 `../flowoct3d_full_20261022`，可结合阅读。

三维组第一段 800 次的病例顺序和随机时间逐元素复用前轮；后两段在训练前冻结。相同表示的模型初始化一致。AMP 跳步重试原病例与时间，不扣减成功更新预算。主表固定使用同预算检查点，最佳验证权重作为另存结果。

## 原版对照与方法差异

公开代码来源为 `flowoct_reproduction/flowoct` 中已审计的作者模型、小波与 ROI 修复快照，原提交 `2309126956d2b80a95f2d307d85a50799cc658a6`。本实验冻结在 `source/flowoct`，哈希写入 `protocol.json`。

- 保留 MONAI `DiffusionModelUNet(spatial_dims=2,in_channels=4,out_channels=4)` 默认网络，包括其注意力层；不是三维扩展的替代实现。
- 保留二维 Haar、裁框速度/重建、切片 InfoNCE、抽片最大投影，以及 `10 / 0.5 / 0.005 / 0.1` 权重。
- 保留 Adam(beta1=0.5)、FP32、无梯度裁剪、整批网络计算。
- 使用同一 `[0,1]` 冻结输入，只做 `ZYX↔YXZ` 无损轴转换；不重复作者 loader 的 resize/min–max/crop/flip。故这是公开模型/损失的同数据对照，不是原论文全流程数值复现。
- 公开代码按首个空间轴 Y 抽取 64/4=16 张 B-scan，每张独立时间；三维模型每次使用全体块、每块一个时间。每次体素曝光和计算量不同。
- 三维模型采用已有时间条件 3D U-Net、固定 OCT 拼接条件、AdamW、AMP、clip1，损失权重为 `1 / 0.5 / 0.1 / 0.005`；主干和训练协议的差异不能单独归因为二维/三维。
- 所有主表统一 Euler8，仅最终 clamp。公开代码另外保存原生 Euler3、每步 clamp 的结果，不拿两种求解设置混排。

## 为什么效果差的机制检查

`diagnose.py` 对上轮 800 次模型和本轮最终模型，比较：

1. 纯 OCT 起点单步估计。
2. 含 50% 真实 OCTA 的训练状态终点估计。
3. 纯 OCT 八步自由生成。
4. 生成/真值高频能量比。
5. 各项损失乘实际权重后的生成器梯度范数与两两余弦。

训练状态含真值只用于诊断，不作为有效生成分数。局部梯度冲突也不能单独证明某模块有害，要与累计删减的实际训练结果一起判断。

另设单变量权重检查 `diagnostics/fm_weight10`，仅把完整三维模型 `fm_weight` 从 1 调为 10，其他模块、初始化、病例、时间、优化器与 2400 次预算全部相同。额外 2400 次更新；含该控制共 21,600 次。该组独立配置和输出，不写入主 8 组协议。

## 执行与恢复

本轮在本机 RTX 3060 Laptop GPU 上运行。独立权重控制与主实验可有重叠计算，不用这些墙钟时间作方法速度排名。

```powershell
# 项目根目录，激活含 MONAI 1.4 与 PyTorch 2.5.1 的环境
python -m unittest discover -s experiments/flowoct3d_progressive_20261022 -p test_progressive.py -v
python experiments/flowoct3d_progressive_20261022/run.py --resume

# 独立权重控制
python experiments/flowoct3d_progressive_20261022/diagnostics/fm_weight10/run.py --resume

# 全部训练完成后
python experiments/flowoct3d_progressive_20261022/diagnose.py --new-final
python experiments/flowoct3d_progressive_20261022/verify.py
python experiments/flowoct3d_progressive_20261022/diagnostics/fm_weight10/verify.py
python experiments/flowoct3d_progressive_20261022/report.py
```

`run.py` 使用本目录冻结的 `source/`。恢复会核对代码、数据与配置，保存模型、Adam/AdamW、AMP 与已完成成功更新数。未保存区间重算，日志截到最近检查点。

## 输出

- `reports/实验报告.md`、`reports/index.html`：完整原因说明、800/1600/2400 次同预算表格和固定窗图。
- `reports/summary.csv`：患者宏平均结果；每例两个体块先在患者内平均。
- 每组 `metrics_*.json`：逐病例、逐患者与汇总指标。
- 每组 `update_00800.pt`、`update_01600.pt`、`update_02400.pt`：完整可恢复状态。
- 每组 `predictions_*/`：三维预测、同网格目标、固定窗 B-scan/投影/误差图。
- `verification/results_audit.json`：源码、输入、输出、实际更新、优化器计数和主要指标独立重算。
- `verification/diagnosis_*.json`：训练状态和生成落差、损失梯度诊断。

本实验为单种子、固定体块小样本机制对照，没有运行测试集或全数据 200 epochs，不能据此宣称最终泛化性能或临床效果。

