# 3DFM-OCTA

用于 OCT → OCTA 的 **3D Flow Matching 基线**与 **3D Flow Matching + 三维 Haar 小波**代码。
两种方法共享数据加载、3D U-Net、训练与全体积推理实现，使用不同训练入口。

| 方法 | 训练入口 | 表示 | 详细说明 |
| --- | --- | --- | --- |
| 3DFM 基线 | `python -m fm3d.train` | 原始三维体素 | [基线说明](fm3d/README.md) |
| 3DFM + 3D Haar 小波 | `python -m fm3d.train_wavelet` | 三维 Haar 八子带 | [小波说明](fm3d/WAVELET.md) |

本仓库发布源代码、测试、依赖和操作说明；模型权重、医学影像、患者名单、训练日志与实验输出均由使用者在本地管理。
源代码快照整理于 2026-10-22。既有说明中的服务器路径、压缩包名和结果目录属于历史运行示例，运行时替换为自己的路径。
这是独立研究实现，方法源于 FlowOCT 的配对直线路径思路；FlowOCT 官方代码见 [cnu-medilab/FlowOCT](https://github.com/cnu-medilab/FlowOCT)。

## 安装

建议使用 Python 3.10。根据运行机器安装兼容的 PyTorch 2.5.1 后，在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

需要 SSIM、LPIPS、FID 等扩展评价与图表时再安装：

```bash
python -m pip install -r requirements-eval.txt
```

更多环境说明见 [Ubuntu 运行说明](fm3d/UBUNTU.md)。

## 数据

数据与病例划分由使用者自行准备。每例 NPZ 包含同尺寸的 `oct`、`octa` 三维数组；CSV 使用
`case_id,patient_id,path,split`，其中 `split` 为 `train`、`validation` 或 `test`。
CSV 的 `path` 相对 CSV 所在目录。必须根据真实数据确认轴序、患者对应关系和灰度范围。

代码校验 3M 的 140/10/50 或 6M 的 180/20/100 配对体划分。下面假设已核实
`ZYX` 轴序、原始 8-bit 灰度，以及位于 `data/OCTA500_common_v1/` 的病例清单。
数据准备步骤见 [基线说明](fm3d/README.md)；`tools/prepare_common_octa500.py` 为原始 BMP 批量准备工具。

## 训练两种方法

所有命令在包含 `fm3d/` 的仓库根目录执行。以下两组均设为 200 轮以对齐训练预算；
基线入口原有默认值为 100 轮，小波入口默认值为 200 轮。

```bash
# 3DFM 基线：体素空间
python -m fm3d.train \
  --manifest data/OCTA500_common_v1/split_3M_z256.csv --fov 3M \
  --axis-order ZYX --intensity-scale 255 --depth 256 \
  --patch 256 64 64 --base-channels 16 --condition \
  --batch-size 1 --epochs 200 --steps-per-epoch 200 \
  --lr 2e-4 --val-every 5 --ode-steps 8 --overlap 0.5 \
  --seed 2026 --workers 0 --device cuda --amp --save-every 5 \
  --output runs/fm3d_3M_200

# 3DFM + 三维 Haar 小波：保持相同的原始体块和训练预算
bash fm3d/scripts/train_wavelet_200.sh \
  data/OCTA500_common_v1/split_3M_z256.csv \
  3M runs/fm3d_haar3d_3M_200
```

Windows PowerShell 可使用 `python -m fm3d.train_wavelet` 并传入相同参数；`--patch` 始终填写体素空间尺寸。
续训时保持原运行路径与关键配置，并在原命令末尾增加 `--resume`。
三维小波保留全部八个子带，使用可微正交变换；无需额外小波库。

另保留了曾用于基线从第 100 轮续训至第 200 轮、每 10 轮保存区间最佳权重的
[独立训练脚本快照](snapshots/baseline_resume_200/README.md)。该快照与共享入口分开保存。

## 评价和可视化

统一评价入口读取本地 checkpoint 中的表示与数据协议，支持上述两种方法：

```bash
python -m fm3d.evaluate \
  --checkpoint runs/fm3d_3M_200/best.pt \
  --split test --device cuda --output runs/fm3d_3M_test

python -m fm3d.visualize \
  --input runs/fm3d_3M_test --output runs/fm3d_3M_test_report
```

小波模型只需换用对应运行目录的 checkpoint。生成的体积与图片保留在本地，并由 `.gitignore` 排除。
扩展指标见 [评价说明](fm3d/EVALUATION.md)。
共享代码还保留了独立的 [Z 投影监督扩展](fm3d/PROJECTION.md)；上述两种基础方案的投影损失权重均为 0。

## 验证

核心训练、断点恢复、数据协议与小波测试只依赖基础环境；PyWavelets 数值对照为可选测试：

```bash
python -m unittest fm3d.test_core fm3d.test_wavelet -v
```

安装扩展评价依赖后可运行全部现有测试：

```bash
python -m unittest discover -s fm3d -t . -p 'test_*.py' -v
```

## 主要文件

```text
fm3d/
  model.py                 时间条件 3D U-Net 与 FM 损失
  wavelets.py              三维 Haar DWT / IDWT
  data.py                  数据协议、BMP 打包、划分与配对体块
  train.py                 共享训练与体素基线入口
  train_wavelet.py         三维小波训练入口
  inference.py             全体积滑窗 ODE 推理
  evaluate.py              按 checkpoint 协议评价
  scripts/                 服务器启动脚本
  test_*.py                现有验证测试
tools/                     数据准备与评价辅助工具
snapshots/baseline_resume_200/
                           基线 200 轮续训训练脚本快照
```
