# 3D Flow Matching + 3D Haar + FlowOCT 完整模块

这是独立的新实验工程。保留现有 `fm3d/` 的时间条件三维 U-Net，加入 FlowOCT 的三维小波、深度对比对齐、体素重建、分层投影和结构区域加权。`upstream/`、`working/`、`flowoct_reproduction/` 及原 `fm3d/` 文件均无需修改。

**本工程是明确记录实现选择的三维研究扩展，不能称为作者代码的逐行复现，也不能用小样本结果代替完整测试。**

## 模块与明确公式

所有体张量轴序为 `[B,1,Z,Y,X]`，Z 是视网膜轴向深度。OCT 与 OCTA 固定除以 255；无逐块 min-max。

```text
c = OCT, y = OCTA
R = 单级正交 3D Haar，或体素消融中的恒等映射
cR = R(c), yR = R(y)
t ~ Uniform[0,1)
xt = (1-t)cR + t*yR
v = UNet3D(xt, t, cR)
Lfm = weighted_MSE(v, yR-cR)

训练辅助终点：yhat = R^-1(xt + (1-t)*v)
Lrec = weighted_L1(yhat, y)
Lproj = 两个视网膜范围内 Z 最大投影 L1 的均值
Lrdca = 同患者内，预测深度切片到真实同深度切片的交叉熵
L = 1.0*Lfm + 0.5*Lrec + 0.1*Lproj + 0.005*Lrdca
```

训练辅助终点包含用于构造 FM 训练路径的目标信号；它**不是正常生成结果**。所有验证、选模和导出的预测都从纯 OCT 开始，以 8 步 Euler 自由积分到 t=1，再逆小波变换。只对最终评价图像裁剪 `[0,1]`，中间系数和训练终点均不裁剪。`auxiliary="source"` 可改用纯 OCT 单步辅助终点，但本次标准消融全部使用 `path`。

| 模块 | 本实现 |
|---|---|
| 三维主干 | 原 `VelocityUNet3D`，base16，三次三维下采样，固定 OCT 条件；没有用 2D 切片模拟 3D 主干 |
| 三维小波 | Z/Y/X 三方向低通/高通，完整保留 LLL 到 HHH 八子带；128×32×32 网络网格对应 256×64×64 体块 |
| RDCA | 共享 Conv2d(1,64)→ReLU→全局池化→Linear(64,128)，只作为深度切片损失头；生成主干仍为 3D |
| 深度负样本 | 每患者最多 32 个非空 Z 平面，等距选取；同深度为正样本、其他深度为负样本；温度 0.1；batch1 仍有负样本 |
| 分层投影 | ILM–OPL 和 OPL–BM 两个区域，分别沿 Z 做 max，再平均两个 L1；病例间从不做 max |
| 结构聚焦 | 训练目标 `M = y > 0.05*max(y)`；前景权重 1、背景权重 0.1；每患者按权重和归一化 |
| 频域结构权重 | 2³ 小波支持内的 M 平均占比，同一位置作用于八子带；不是对二值 mask 做有符号 DWT 后当权重 |
| 投影与 RDCA 的结构权重 | 投影依据该视线在层范围内是否含前景赋权；RDCA 输入按同一前景/背景系数加权 |

层坐标来自 `GT_Layers/*.mat` 的 `Layer`，原生 Z=640、MATLAB 1 基，顺序 ILM/IPL/OPL/ISOS/RPE/BM。重采样后中心坐标为 `(z+0.5)*640/Z-0.5`，减 1 的层边界按闭区间比较。无有效采样中心的投影射线置零、训练时排除。训练块必须保留完整预处理 Z；Y/X 裁剪同时裁层标注。

**额外监督的边界：** 完整模型使用训练病例的层标注定义投影损失。验证层标注只用于生成后的评价，不参与训练或网络输入。官方血管 mask 仅用于血管区误差评价；训练结构 mask 由训练 OCTA 灰度生成。部署时只需要 OCT。

## 与论文、公开代码的关系

参考 [FlowOCT 论文 §3.2–3.6](https://www.ijcai.org/proceedings/2026/0213.pdf) 与[公开仓库](https://github.com/cnu-medilab/FlowOCT)，本地锁定公开代码版本 `2309126956d2b80a95f2d307d85a50799cc658a6`。

论文描述三维小波、固定 OCT 条件、轴向 RDCA、分层投影和结构聚焦。公开代码实际采用二维 MONAI 主干/二维小波、对抽片合并后的批次最大投影、预测与目标联合裁框，并且 FM 系数为 10。

本工程采用论文的模块意图，并明确以下自定选择：

- 主干沿用现有 3D U-Net，不是把 MONAI 二维模型权重膨胀为三维。
- 采用论文式 `1 / 0.5 / 0.1 / 0.005` 展开系数；不混用公开代码的 FM=10。
- 采用公开代码的路径终点辅助训练方式，但取消训练 clamp；论文没有唯一给出对应的可执行端点训练算法。
- 固定选用两个已提供标注的视网膜范围，两个投影取均值；这些范围、reduce 方式不是声称从公开代码恢复出的作者参数。
- 保留背景权重 0.1，避免完全放弃背景抑制；阈值、频域占比、按有效权重归一化都是公开记录的实现选择，不声称论文提供了这些完整细节。
- RDCA 采用论文的预测→真实深度分类，而不是公开代码对两组特征构造的双向 2N 对比；非空深度的选择也显式记录。
- 标准组仍是均匀 t；`08_full_half_start` 的一半 t=0 是根据前期诊断另设的实验组，不属于 FlowOCT 原贡献。

## 小样本实验

实验目录：`experiments/flowoct3d_full_20261022/`。入口 `run.py` 使用目录内冻结的 `source/`，后续修改工作区不会改变正在运行的模型。复用 2026-10-21 已冻结输入，16 个 NPZ 文件 SHA256 逐个匹配。

| 疾病 | 训练病例 | 验证病例 |
|---|---|---|
| NORMAL | 10301 | 10319 |
| DR | 10304 | 10447 |
| AMD | 10310 | 10456 |
| CNV | 10439 | 10302 |

每例体块 `[256,64,64]`，固定起点 `(0,64,64)`、`(0,120,120)`；不根据目标信号挑位置。全部病例沿用既有划分，没有读取测试病例图像。

| 组 | 设置 | 单病例更新 | 四病例更新 |
|---|---|---:|---:|
| 01_voxel_fm | 体素 FM | 400 | 800 |
| 02_haar_fm | 三维 Haar FM | 400 | 800 |
| 03_full | 完整模块 | 400 | 800 |
| 04_no_rdca | 完整模型去 RDCA | — | 800 |
| 05_no_projection | 完整模型去分层投影 | — | 800 |
| 06_no_structure | 完整模型去结构加权 | — | 800 |
| 07_no_reconstruction | 完整模型去体素重建 | — | 800 |
| 08_full_half_start | 完整模块，恰好一半更新 t=0 | 400 | 800 |

合计 12 个独立运行、8,000 次成功优化更新。两阶段从头独立初始化；单病例只训练 10301 的第一块。相同表示的所有组参数初始化完全相同；各组使用相同病例次序、相同随机时间序列。起点组只替换预先冻结的一半时间。四病例中每块恰好 100 次曝光。AMP 跳步重试同一病例与时间，不消耗成功更新预算。

AdamW lr2e-4、weight_decay0、梯度裁剪1、batch1、AMP，所有组一致。每 200 次更新评价，按验证患者宏平均 ILM–OPL SSIM 保存 `best_validation.pt`；主表统一使用 400/800 次的 `latest.pt`。患者内先平均体块，再患者等权平均。不能把相关体块当独立患者，也不能把短程预算下的差异当成收敛后的优劣。

```powershell
# 项目根目录；先激活已安装 PyTorch 的环境
python -m unittest flowoct3d.test_core flowoct3d.test_cli -v
python experiments/flowoct3d_full_20261022/run.py --resume
python -m flowoct3d.verify --root experiments/flowoct3d_full_20261022
python -m flowoct3d.report --root experiments/flowoct3d_full_20261022
```

重新独立运行可复制原病例缓存生成新的实验目录，再启动；不要覆盖已有结果：

```bash
python -m flowoct3d.small_data --previous experiments/ablation_validation_20261021 --output experiments/my_full3d
python -m flowoct3d.study --root experiments/my_full3d --benchmark
python -m flowoct3d.study --root experiments/my_full3d
```

## 全量训练、恢复与推理

本机原公共数据路径为 `data/OCTA500_common_v1`。可移植包不包含全数据，服务器使用实际 CSV 和标注路径。保持原有 3M 140/10/50 或 6M 180/20/100 划分。

```bash
python -m flowoct3d.train \
  --manifest /data/OCTA500_common_v1/split_3M_z256.csv --fov 3M \
  --layers-dir /data/OCTA500_common_v1/labels_native/GT_Layers \
  --output runs/flowoct3d_3M_full --device cuda:0

# 重复原训练参数并加 --resume，可增大 --epochs。
# 恢复会验证模型、数据清单、层标注哈希与训练源码，保留 AdamW/AMP 状态。
# 每个 epoch 200 次成功更新；200 epochs 共 40,000 次。仅在完整 epoch 边界恢复。
```

更换 FOV 使用对应 CSV 和 `--fov 6M`；层目录不变。可用 `--loss-config path.json` 传入 `LossConfig` 字段开关模块；完整默认参数见 `configs/full.json`。以 `--start-fraction 0.5` 启动起点组时使用新输出目录。正式训练默认没有自动应用起点修改。

```bash
# 只有 OCT 的部署入口。输入 NPZ 只需要 oct 数组与可选 axis_order=ZYX。
python -m flowoct3d.predict --checkpoint runs/flowoct3d_3M_full/best.pt \
  --input /data/new_oct.npz --output predictions/new_oct --device cuda:0

# 完整测试：先仅加载 OCT 生成，之后才读取 OCTA/层标注计算指标。
python -m flowoct3d.predict --checkpoint runs/flowoct3d_3M_full/best.pt \
  --manifest /data/OCTA500_common_v1/split_3M_z256.csv --fov 3M --split test \
  --layers-dir /data/OCTA500_common_v1/labels_native/GT_Layers \
  --output predictions/test3M --device cuda:0
```

默认强度 255、Z256、patch256×64×64、Euler8、50% 滑窗重叠。输入单位不同必须传 `--intensity-scale`；模型训练几何与推理需一致。推理只最终 clamp，整卷仅做一次小波分解，每个 ODE 步在统一系数网格拼接速度，避免不同窗口的小波相位冲突。

## 验证范围和文件

核心测试覆盖三维变换、各模块独立反传、batch1 对比、投影轴/病例隔离、空区域、频域权重、同初始化、时间序列、滑窗和保存恢复。独立 CLI 测试使用合成 200 例小体积，验证正式训练、断点恢复逐参数/优化器一致、仅 OCT 推理及验证集评价；合成测试不能当作真实数据性能。

`verify.py` 在实际小样本实验结束后独立读取预测，重算体素和 ILM–OPL 指标，核对真实更新数、优化器计数、对比头是否实际变化、输入/权重/预测哈希与患者汇总。`report.py` 生成中文报告、CSV 和固定窗对比图。

完整数据集 200 轮的入口已经实现，当前请求运行的是小样本验证；没有自动启动 200 轮全量训练。

