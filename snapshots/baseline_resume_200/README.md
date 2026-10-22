# 基线续训至 200 轮的训练脚本快照

`train.py` 原样保存自 2026-10-21 的基线续训代码，增加了 `--window-best-every`，
用于按不重叠的轮次区间保留验证 MAE 最低的 checkpoint。
该脚本使用体素空间 3DFM，与仓库主目录的共享训练入口分别保存。

如需复用这一保存策略，请在单独的代码目录中使用该脚本。以下命令从仓库根目录执行，
只在新建的 `baseline_window/` 中放置副本：

```bash
mkdir baseline_window
cp -r fm3d baseline_window/fm3d
cp snapshots/baseline_resume_200/train.py baseline_window/fm3d/train.py
cd baseline_window
```

随后可按基线参数调用 `python -m fm3d.train`，并添加
`--epochs 200 --val-every 1 --window-best-every 10`。仅续训已有运行时添加 `--resume`，
并使用原来的数据清单、运行目录和关键配置。
这一快照使用 `--window-best-every`；主目录共享入口的 `--save-every` 属于另一种保存策略。

在每轮均验证的条件下，`best_window_0101_0110.pt` 等文件保存对应十轮区间的最低验证 MAE 权重；
`best.pt` 保存全程验证最佳权重，`latest.pt` 保存最后完成轮次。
未完成的区间文件会随验证改善而更新，完成区间时写出对应 JSON 记录。
模型选择始终使用验证集，测试集用于最终评价。
