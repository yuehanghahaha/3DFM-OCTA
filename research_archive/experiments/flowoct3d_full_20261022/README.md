# 2026-10-22：真正三维的 FlowOCT 完整模块实验

工程入口与全部公式见 [flowoct3d/README.md](../../flowoct3d/README.md)。

本目录保存本轮实际使用的冻结源代码、16 个固定真实体块、8 组配置、12 个独立训练运行，以及全部原始权重/预测。数据选择与此前小样本实验一致，加入训练层标注后的用途已重新写入 `data/selection.json`。

从项目根目录运行：

```bash
python experiments/flowoct3d_full_20261022/run.py --resume
python -m flowoct3d.verify --root experiments/flowoct3d_full_20261022
python -m flowoct3d.report --root experiments/flowoct3d_full_20261022
```

`run.py` 优先加载本目录 `source/` 的实际训练快照，避免后续工作区更改影响恢复。源码、配置、输入哈希与时间/病例序列在每个运行 `protocol.json` 中冻结。没有后台调度或服务器任务；本轮实际计算在本机 RTX 3060 Laptop GPU 上完成。

- `reports/实验报告.md`、`reports/index.html`：中文结果及可视化。
- `reports/summary.csv`：两个阶段、各组、训练/验证的患者宏平均指标。
- `verification/results_audit.json`：独立指标重算、哈希、成功更新数、优化器计数和 RDCA 头变化。
- `verification/foreground_metrics.json`：统一灰度阈值前景的补充评价；不冒充官方血管分割或论文原评价。
- `verification/unit_tests.txt`：核心测试与完整训练/恢复/预测 CLI 测试。
- `verification/full_volume_smoke.json`：额外一例 10319 的真实完整视野推理检查。
- `verification/full_volume_10319/`：该例整卷预测和可视化；不混入固定体块主表。
- 每个组的 `one_case/` 或 `few_case/`：初始、最近、最佳验证权重、逐步日志、训练曲线、最终逐例指标和三维预测。

单病例阶段 4×400，四病例阶段 8×800，共 8,000 次成功更新。每个运行独立从头开始，所有小波组初始化完全相同。只报告固定预算结果，不根据验证集增加某一组训练时长。

本轮只验证短程拟合、实现完整性和少量未训练病例表现，没有全量 200 轮训练或测试集结果。

