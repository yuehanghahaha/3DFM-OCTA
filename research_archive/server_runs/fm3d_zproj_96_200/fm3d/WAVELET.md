# 3D FM + 三维 Haar 小波：200 轮实验

这是在现有 `fm3d` 基线上增加的**一级三维 Haar 小波域条件 Flow Matching**。复用数据读取、病例划分、时间条件 U-Net、优化器和完整体积评价；训练与 ODE 状态改为 8 个小波子带。新入口为 `python -m fm3d.train_wavelet`，默认从头训练 200 轮。

## 1. 服务器直接运行

将 `fm3d_wavelet_ubuntu.zip` 解压到一个新代码目录，进入包含 `fm3d/` 的目录，激活**原来基线已能运行的 Python 环境**。代码包不含数据与权重，继续使用服务器原有数据；不需要再次预处理、划分病例或安装小波库。保留压缩包中的 `tools/prepare_common_octa500.py`，它供一项原有数据协议测试使用。

以下 `/data/OCTA500_common_v1` 是示例数据位置，换成服务器上基线使用的实际目录。CSV 必须与基线相同。

```bash
# 在包含 fm3d/ 的代码根目录执行
python -m unittest fm3d.test_core fm3d.test_wavelet -v

# 先测服务器自身的显存和优化更新；--patch 始终填写原始体素尺寸
python -m fm3d.benchmark --representation haar3d --device cuda \
  --patch 256 64 64 --base-channels 16 --batch-size 1 \
  --output runs/haar3d_gpu_check.json

# 3M：新实验，从头训练到 200 轮
bash fm3d/scripts/train_wavelet_200.sh \
  /data/OCTA500_common_v1/split_3M_z256.csv \
  3M runs/fm3d_haar3d_3M_200
```

6M 使用另一个输出目录：

```bash
bash fm3d/scripts/train_wavelet_200.sh \
  /data/OCTA500_common_v1/split_6M_z256.csv \
  6M runs/fm3d_haar3d_6M_200
```

后台运行并保存日志：

```bash
mkdir -p logs
nohup bash fm3d/scripts/train_wavelet_200.sh \
  /data/OCTA500_common_v1/split_3M_z256.csv \
  3M runs/fm3d_haar3d_3M_200 \
  > logs/fm3d_haar3d_3M_200.log 2>&1 &

tail -f logs/fm3d_haar3d_3M_200.log
```

日志放在 `logs/`，不要预先放进训练输出目录，否则防覆盖检查会拒绝启动。正式训练目录须为空或不存在。原有基线权重留在原运行目录。

也可先做真实数据流程短跑；它会进行 2 个训练批次和全部验证病例的 1 步 ODE，仍可能需要数分钟：

```bash
bash fm3d/scripts/train_wavelet_200.sh \
  /data/OCTA500_common_v1/split_3M_z256.csv \
  3M runs/fm3d_haar3d_pipeline_check \
  --epochs 1 --steps-per-epoch 2 --val-every 1 --ode-steps 1
```

正式实验用上面的新目录从头训练，不从短跑续训。

## 2. 具体改了什么

轴序仍为 `[B,C,Z,Y,X]`；子带字母依次表示 **Z、Y、X** 上的低通或高通：

```text
W = 一级正交 3D Haar DWT
子带顺序 = LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
低通 = (偶数位置 + 奇数位置)/sqrt(2)
高通 = (偶数位置 - 奇数位置)/sqrt(2)

c_w = W(OCT)                         [B,8,128,32,32]
y_w = W(OCTA)                        [B,8,128,32,32]
t ~ Uniform(0,1)                     [B]
x_t = (1-t)c_w + t*y_w
u = y_w - c_w
v = UNet3D(concat(x_t,c_w), t)        输入16通道，输出8通道
L_FM = mean((v-u)^2)                 所有子带等权

推理：从 x_0=c_w 开始，Euler 积分8步
输出：clamp(W^{-1}(x_1), 0, 1)        [B,1,256,64,64]（单块示意）
```

8 个子带全部保留；不做系数阈值、截断、额外归一化或高频加权。高频系数可为负；LLL 也可超过 1，不能对系数使用 `[0,1]` 裁剪。正交变换在偶数网格上保持能量，全部子带等权 MSE 与逆变换后的体素速度 MSE 等价；该实验改变的是网络表示，不是额外添加一种血管损失。

三维单级变换与逆变换的定义可参照 [PyWavelets 官方 nD DWT 文档](https://pywavelets.readthedocs.io/en/latest/ref/nd-dwt-and-idwt.html)。本实现使用可求导的 PyTorch 运算，并用 PyWavelets 作可选的独立数值核对；运行实验不依赖 PyWavelets。

空间尺寸各减半，U-Net 的内部特征图体素数减少。8 个子带的**总元素数没有减少**，因此这是可逆表示变换，不能声称压缩丢弃了 7/8 信息，也不能保证显存或时间恰好降为 1/8。相同网络层数在原始体素上的感受野也会变化，应将本实验描述为“小波域建模方案”的对比。

训练仍对原来的随机配对体块做 DWT。全卷推理先在 CPU 上对整卷补边并做一次 DWT，再在统一的小波网格上滑窗，逐步拼接速度、更新全卷状态。不会让不同窗口使用不同的全卷小波原点；最后做一次 IDWT、去除补边、裁剪最终强度。奇数体积尺寸在末端复制补到偶数；训练 patch 各维必须是至少 16 的偶数。命令中的 patch 总是**体素空间**大小，无需手动填写减半尺寸。

## 3. 固定实验配置

| 项目 | 小波实验默认值 |
|---|---|
| 数据 | 同基线 CSV；3M/6M 分别训练 |
| 输入协议 | ZYX，固定除以 255，深度 256 |
| 原始体块 | `256×64×64` |
| 小波体块 | `8×128×32×32` |
| 网络 | 原残差 3D U-Net，base=16，时间嵌入，显式 OCT 条件 |
| 优化器 | AdamW，lr=`2e-4`，weight_decay=0，梯度范数上限 1 |
| batch / AMP | 1 / CUDA AMP |
| 轮数 | 200 |
| 每轮训练批次 | 200，随机有放回取病例与体块 |
| 总训练预算 | 40,000 批次；AMP 跳过的更新单独记录 |
| 验证 | 每 5 轮及最后一轮，完整验证体积 |
| 推理 | Euler 8 步、50% 滑窗重叠 |
| 模型选择 | 最低验证集患者宏平均 MAE |
| 种子 / workers | 2026 / 0 |

沿用固定学习率，没有额外引入调度器、EMA、拓扑损失或增强，以便解释第一轮消融结果。`workers=0` 沿用原来的稳妥设置；若服务器读取成为瓶颈，可追加 `--workers 2`，并注意每个 worker 会读取完整体积占用主机内存。

200 轮是本次预算，不是已证明最优的轮数。现有 100 轮日志中，第 90 轮验证 MAE 为 0.01369975，第 100 轮为 0.01766786；后者更差，所以应使用验证最优权重判断效果，而非默认取最后一轮。此前日志不能预测新小波实验的收敛位置。

## 4. 断点、快照和测试

中断后，使用同一条命令追加 `--resume`：

```bash
bash fm3d/scripts/train_wavelet_200.sh \
  /data/OCTA500_common_v1/split_3M_z256.csv \
  3M runs/fm3d_haar3d_3M_200 --resume
```

恢复模型、优化器、AMP 状态和已记录的最佳 MAE，从上一个完成的 epoch 后继续。`--epochs 200` 表示**总计 200 轮**，不表示再追加 200 轮；未完成的 epoch 会重跑。数据、模型表示和主要训练配置变化会被拒绝。checkpoint 记录数据绝对路径，续训时保持服务器数据与输出位置不变。

| 文件 | 用途 |
|---|---|
| `latest.pt` | 最近完成轮次，用于续训 |
| `best.pt` | 全程验证 MAE 最低的权重，用于最终评价 |
| `epoch_0005.pt`、`epoch_0010.pt`、… | 每 5 轮的当前权重 |
| `best_through_0005.pt`、`best_through_0010.pt`、… | 每 5 轮截至当时验证 MAE 最低的权重；没有改善时相邻文件的模型可能相同 |
| `best_through_0100.pt` | 前 100 轮内验证最优权重，用于与已有 100 轮基线比较 |
| `best_through_0200.pt` | 前 200 轮内验证最优权重 |
| `history.jsonl`、`validation_*.json` | 每轮训练记录与逐病例验证分数 |
| `config.json`、`environment.json` | 超参数、表示类型、网络尺寸、设备与参数量 |
| `manifest_snapshot.json` | 本次使用的固定病例清单 |

`best_through_0100.pt` 内部的 `epoch` 可能是 90 等较早轮次，这是正常的验证模型选择，不是保存错误。恢复到超过 200 轮时，原 100/200 轮快照仍保留。

评价入口会自动根据 checkpoint 选择体素模型或小波模型：

```bash
# 与已有 100 轮基线比较时用前 100 轮内的最佳模型
python -m fm3d.evaluate \
  --checkpoint runs/fm3d_haar3d_3M_200/best_through_0100.pt \
  --split test --device cuda --output runs/fm3d_haar3d_3M_test100

# 200 轮预算的最终结果
python -m fm3d.evaluate \
  --checkpoint runs/fm3d_haar3d_3M_200/best_through_0200.pt \
  --split test --device cuda --output runs/fm3d_haar3d_3M_test200
```

测试集仅用于预先确定方案的最终报告；不要依据这两次测试结果再选择超参数。输出仍为全卷 NPZ（`octa`、`octa_gt`、`axis_order=ZYX`）与 `metrics.json`。评价协议保持 MAE/PSNR，与原基线相同。

## 5. 怎样比较才公平

优先比较“基线 100 轮”与“小波前 100 轮最佳”。如要把 200 轮结果作为主对比，基线也应在原运行目录续训至 200 轮：原命令保持主要配置，改为 `--epochs 200 --resume --save-every 5`；延长前另存其当前 `best.pt`，保留原 100 轮结果。旧 checkpoint 没有 `representation` 字段时会按 `voxel` 处理。

小波实验的首尾卷积从 2→1 通道改成 16→8，不能直接把体素基线的 `latest.pt` 当作小波续训文件。这里“在基线上添加小波”指复用基线方法与训练协议，默认重新初始化训练；没有进行未经定义的部分权重迁移。参数量会因首尾通道数略有增加，具体值写入 `environment.json`。

保持数据、patch、每轮批次、推理步数、评价协议一致。当前代码保留原随机裁块规则；不因显存减少立即扩大 patch，以免第一次对比同时改变多个因素。单次实验有随机性，正式结论仍需重复种子验证。

## 6. 代码入口与检查范围

| 文件 | 职责 |
|---|---|
| `wavelets.py` | 三维正交 Haar DWT / IDWT，固定子带轴序 |
| `model.py` | 复用 U-Net，按表示选择 1 或 8 个状态通道，小波域 FM 损失 |
| `inference.py` | 在统一体素/小波网格上进行完整体积 ODE 积分 |
| `train_wavelet.py` | 默认 200 轮的小波实验入口 |
| `scripts/train_wavelet_200.sh` | Ubuntu/服务器启动脚本 |
| `train.py`、`evaluate.py`、`benchmark.py` | 共享训练、评价、显存检查；旧入口默认仍为体素基线 |
| `test_wavelet.py` | 可逆性、能量、8 子带轴序、参考数值、梯度、滑窗、断点恢复、测试集导出 |

2026-10-20 本地验证使用 PyTorch 2.5.1+cu121、RTX 3060 Laptop：

- 原基线与小波共 17 项自动测试通过；另补测了 ODE 中间状态越界后返回的情况，确认不会逐步裁剪。
- 使用改动前代码生成的旧 checkpoint，成功用新入口评价、续训；CPU 参数与旧代码连续训练的结果逐项完全一致。
- 默认 `256×64×64`、base16、batch1、AMP 的合成训练完成 1 次预热与 3 次更新，无 AMP 跳过。峰值 allocated 约 **0.302 GiB**，reserved 约 **0.348 GiB**；网络参数 2,106,600。
- 真实 3M 训练病例完成 2 次体块更新与 **1 步完整 `256×304×304` 体积推理**，输出尺寸和数值检查通过；Haar 往返最大误差约 `2.98e-7`。这只是流程验证，没有使用测试病例调参，也未形成训练完成后的效果结果。

具体记录随代码包放在 `verification/`。这些检查不代表服务器 3090 的速度或正式医学图像质量；200 轮正式训练需要在服务器启动后完成。
