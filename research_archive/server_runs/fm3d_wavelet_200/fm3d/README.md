# OCT → OCTA：体素空间 3D Flow Matching 起步实现

**2026-10-20 新实验：** 已增加可独立启动的三维 Haar 小波域 FM，默认 200 轮。
服务器启动、续训、100/200 轮预算对比见 [WAVELET.md](WAVELET.md)。
以下文档描述原体素基线；原入口的默认模型、损失与超参数保持不变。

本目录是独立的、可以训练和测试的 3D FM 基础实现。保留 FlowOCT 的配对直线路径思路，但使用新的三维网络与输入协议。不是作者官方实现，也未完成患者数据上的性能验证。`upstream/`、`working/` 和其他基线目录未被修改。

## 1. 方法具体是什么

设源 OCT 为 c，真实 OCTA 为 y，二者都为 `[B,1,Z,Y,X]`。

```text
t ~ Uniform(0,1)，每个体块一个时间值
x_t = (1-t)c + ty
目标速度 u = y-c
预测速度 v = 3D_UNet(concat(x_t,c), t)
L_FM = mean((v-u)^2)
```

训练时真实 OCTA 用于构造路径和监督；推理时只使用 OCT，从 `x_0=c` 开始积分：

```text
x_{k+1} = x_k + (1/K) * v(x_k, k/K, c)
```

- 默认是**显式 OCT 条件的确定性配对流**，不从高斯噪声开始，不声称生成多种随机 OCTA。
- 3D U-Net 输入2通道（当前状态与固定OCT），输出1通道（有正负的速度）。时间经正弦/余弦嵌入后注入残差块。
- 网络有3次空间下采样、跳跃连接、GroupNorm；默认基础通道16，约210万参数。没有全局注意力、2D/3D小波或VAE。
- `--no-condition` 将输入改为单通道 `x_t`，OCT仍然是积分起点。可用于显式条件消融。
- 速度输出不使用Sigmoid；训练不裁剪速度，ODE中间状态也不逐步clamp。仅最终图像裁到[0,1]，并明确作为评价协议。
- 第一版仅有FM速度MSE。没有cbDice、分割器、三轴投影损失、对比学习或分阶段预训练；后续可在此基础上做逐项消融。

## 2. BMP 与 NPZ：格式不是前处理算法

| 格式 | 保存什么 | 注意点 |
|---|---|---|
| 原始BMP | 一张B-scan一个文件 | 一例需要按数字编号堆叠成三维体；两模态切片必须对应，不能按字符串排序成1、10、100、2 |
| 配对NPZ | 同一例的`oct`、`octa`两个三维数组 | 只是容器；可能保存原始灰度，也可能被其他脚本归一化、裁剪或转轴。必须追查生成过程 |

本目录的转换器保存原始8-bit灰度值，NPZ采用无损压缩，只堆叠和转轴，不去噪、不裁视网膜、不展平、不归一化、不缩放。两者在此条件下信息相同。NPZ减少文件打开和配对管理，但整卷解压不保证更快；本版本每次样本仍加载完整体再取块，正式长训练可再优化缓存。

内部统一轴序：`Z=轴向深度，Y=扫描堆叠方向，X=B-scan横向`。张量统一为 `[B,C,Z,Y,X]`。文件名、shape及字母习惯不能替代实际轴方向核验。

## 3. 明确的数据协议

| 项目 | 本实现 |
|---|---|
| 3M划分 | 140训练／10验证／50测试；严格校验配对体数量 |
| 6M划分 | 180训练／20验证／100测试；严格校验配对体数量 |
| 病例名单 | 独立CSV；检查重复case、重复文件路径、患者跨集合。名单数量相同不代表论文官方病例名单相同 |
| 强度 | 必须显式传`--intensity-scale`；原生8-bit灰度用255，已核实的[0,1]数据用1。几何处理后、取块前固定缩放到[0,1]，无逐块min–max |
| 几何 | 默认仅将轴向Z缩放到256，三线性插值、align_corners=False；横向Y/X不变。`--depth 0`保留原始深度 |
| 训练取块 | 默认连续`[256,64,64]`，配对共享坐标，不拆成二维独立切片，无翻转增强 |
| 验证／测试 | 保留完整预处理体；确定性滑窗覆盖全视野，默认50%重叠；每个ODE步先加权拼接速度，再统一更新全卷状态 |
| 输出网格 | `[256,Y,X]`，若原始为640深度则不声称恢复640的原始细节；`--depth 0`时才保留原深度 |
| 基础指标 | 完整预处理体MAE、PSNR（data_range=1，MSE下限1e-12）；每病例计算、同患者多扫描先平均，再患者宏平均 |
| 模型选择 | 最低验证集MAE保存best.pt；不使用测试集选择权重 |

这是**你自己的初始实验协议**。默认深度256、基础通道16、200训练批次/epoch、8步Euler等是起步设置，不是已验证最优参数。它改变了FlowOCT原代码的裁剪、归一化、维度、条件与损失，不能把性能差异全部归因于“2D变3D”。主干维度的受控对照需要另做。

PSNR和MAE只是起步运行指标。正式论文还需冻结3D/投影SSIM等指标定义、投影轴与区域、病例汇总、置信区间。现有`tools/evaluate_results.py`可读取本实现导出的`octa`与`octa_gt`，但其中前景阈值是示例协议，不是血管真值，也不是原论文协议。

## 4. 文件入口

| 文件 | 职责 |
|---|---|
| `model.py` | 时间条件3D U-Net、直线路径、速度MSE |
| `data.py` | BMP打包、CSV划分、协议校验、轴变换、强度/几何处理、配对随机块 |
| `inference.py` | CPU保存全卷、GPU处理窗口，每步拼接速度，边缘正权重覆盖 |
| `train.py` | 随机取样训练、混合精度、全卷验证、best/latest权重、续训 |
| `evaluate.py` | 根据checkpoint冻结的协议评价验证／测试集，并导出完整体 |
| `benchmark.py` | 合成体前向+反向+Adam更新显存测试，可直接在3090运行 |
| `test_core.py` | 数学、梯度、轴与配对、BMP顺序、划分、滑窗和端到端续训测试 |

## 5. 操作顺序

所有命令从项目根目录运行。可复用本项目现有`flowoct`环境；此实现只依赖PyTorch、NumPy、Pillow。需要安装时，先按训练机器安装CUDA版PyTorch，再安装`fm3d/requirements.txt`，不要在已有可用环境中盲目替换CUDA组件。

```powershell
conda activate flowoct
Set-Location '/workspace/FlowOCT'
python -m unittest fm3d.test_core -v
python -m fm3d.benchmark --patch 256 64 64 --base-channels 16 --output runs/3090_memory.json
```

建议先batch=1；3090上的实测有余量后，再尝试`--patch 256 96 96`或`256 128 128`。扩大patch时要同时修改训练参数；峰值显存不等于显卡全部使用量。不要先强行整卷256×400×400训练。

### A. 如果当前只有BMP

先对一例核验：配对目录中的1.bmp等编号是否表示同一扫描位置；单张图的行是否为轴向深度；文件数量与体尺寸是否符合元数据。下列`YZX`仅适用于确认“堆叠轴=Y、图像行=Z、图像列=X”的情况。

```powershell
python -m fm3d.data pack --oct-dir PATH_TO_OCT_CASE --octa-dir PATH_TO_OCTA_CASE --output data/fm3d/3M/CASE_ID.npz --axis-order YZX
```

将`PATH_TO_*`和`CASE_ID`替换为真实路径/病例编号，逐例打包；两组BMP名称必须完全对应、数字编号连续。若命名不同，应先明确配对关系，转换器不会自行猜测。输出NPZ自带`axis_order=ZYX`。3M、6M分别放置。

### B. 如果已有配对NPZ

不用再转BMP。先确认：

1. 数组名是否为`oct`、`octa`，两个shape是否完全相同。
2. 轴序是否为ZYX；如果是作者示例的`[Y,X,Z]`，训练传`--axis-order YXZ`。
3. 灰度是否仍为0..255，还是已经归一化到0..1；不能仅依据float32判断。
4. 是否做过裁剪/缩放；已经裁掉的视野不能由本加载器恢复。

同一运行中所有NPZ必须有相同轴和强度约定；带有轴元数据的NPZ若与命令冲突会报错。错误强度上界会报错，但0..1数据误填255仍落在有效区间，无法自动识别，必须核对来源。

### C. 固定病例名单

建立`data/fm3d/3M_inventory.csv`（完整200行）或6M的300行：

```csv
case_id,patient_id,path
10301,VERIFIED_PATIENT_ID,3M/10301.npz
```

此行是格式示意，不是已核实的患者映射。`path`相对CSV目录；`patient_id`必须来自患者元数据，同一患者多次扫描/双眼使用相同ID，不能默认为扫描编号。

若确认每例均来自不同患者、并接受自己生成的随机协议，可以运行：

```powershell
python -m fm3d.data split --inventory data/fm3d/3M_inventory.csv --fov 3M --seed 2026 --output data/fm3d/split_3M.csv
python -m fm3d.data split --inventory data/fm3d/6M_inventory.csv --fov 6M --seed 2026 --output data/fm3d/split_6M.csv
```

划分器只产生自己的固定随机名单，不宣称沿用官方具体ID，不做疾病分层，也不覆盖已有划分。如果存在重复患者，它会拒绝随机按扫描分配：应自行按患者组制定满足数量约束的名单。疾病分层也应依据元数据单独制定。

已有人工/官方病例名单时，直接使用四列`case_id,patient_id,path,split`，split取`train/validation/test`，不用运行split。只要相同患者不跨集合，手工名单可以包含同患者多次扫描；数量仍按配对体计算。若两FOV合并训练，要另行审计跨FOV患者重叠；本实现按独立3M/6M实验校验。

读取全部体核验形状、灰度、轴元数据（不会训练）：

```powershell
python -m fm3d.data check --manifest data/fm3d/split_3M.csv --fov 3M --axis-order ZYX --intensity-scale 255
```

检查依赖你提供的身份信息，不能证明未知患者映射正确或自动验证OCT/OCTA解剖配准。重复文件路径会被拒绝，但复制到另一文件名的相同体内容没有自动去重。

### D. 训练基础3DFM

```powershell
python -m fm3d.train --manifest data/fm3d/split_3M.csv --fov 3M --axis-order ZYX --intensity-scale 255 --depth 256 --patch 256 64 64 --base-channels 16 --batch-size 1 --epochs 100 --steps-per-epoch 200 --val-every 5 --ode-steps 8 --output runs/fm3d_3M
python -m fm3d.train --manifest data/fm3d/split_6M.csv --fov 6M --axis-order ZYX --intensity-scale 255 --depth 256 --patch 256 64 64 --base-channels 16 --batch-size 1 --epochs 100 --steps-per-epoch 200 --val-every 5 --ode-steps 8 --output runs/fm3d_6M
```

这里一个epoch定义为200个训练批次，随机有放回抽病例并取块，不是每病例恰好遍历一次。100 epochs=20,000个批次；若AMP检测到梯度溢出，会跳过该批次更新并降低缩放系数，实际更新次数记录为`optimizer_updates`，跳过次数为`amp_skipped_updates`。训练默认开启CUDA混合精度，初始缩放系数1024；`--no-amp`关闭。默认worker=0方便Windows调试，可按CPU/RAM和I/O调整。

输出：`config.json`、病例清单快照、环境信息、逐epoch日志、每次验证逐例结果、`latest.pt`和`best.pt`。配置包含病例列表哈希，但不包含NPZ内容哈希，运行期间不要替换数据文件。

续训：重复原训练命令，增加`--resume`，可将`--epochs`增加到更大值。支持epoch边界恢复模型、优化器和AMP状态；逐epoch固定随机种子，不保存未完成epoch内的进度。关键预处理、名单、patch、模型和训练预算设置变化会拒绝续训，消融请使用新目录。checkpoint含优化器等Python对象，只加载自己生成或可信来源文件。

### E. 测试并导出

先在验证集比较ODE步数，例如3/8/16，再固定选定值用于测试，不能用测试分数选择步数。

```powershell
python -m fm3d.evaluate --checkpoint runs/fm3d_3M/best.pt --split validation --ode-steps 8 --output runs/fm3d_3M_val8
python -m fm3d.evaluate --checkpoint runs/fm3d_3M/best.pt --split test --ode-steps 8 --output runs/fm3d_3M_test
```

每例输出一个NPZ：`octa`是预测、`octa_gt`是同网格目标、`axis_order=ZYX`。报告记录实际推理步数和checkpoint epoch。生成函数只需要OCT；评价入口为了算指标才读取OCTA真值。

全卷状态和累加器在CPU，仅当前patch送GPU。每个ODE步的所有窗口读取同一个当前状态，拼接后一起更新；不会让前一个窗口的更新污染后一个窗口。滑窗是对完整视野的近似计算，依然可能有边界效应，应检查血管连续性；它不是“拓扑正确”的保证。

## 6. 已完成验证及边界

2026-10-19，本机PyTorch 2.5.1+cu121：

- 9项自动测试覆盖真值速度零误差、batch=2梯度、条件开关、不同轴序、配对裁剪、BMP无损数字排序、两个FOV数量、患者与重复影像泄漏拒绝、重复组调整保持配额、AMP溢出跳过后恢复、滑窗覆盖与补边，以及native在线重采样与z256离线缓存的输入逐元素一致性。最新结果见`artifacts/fm3d_implementation_20261019/unit_tests_after_real_data.txt`。
- 合成200例按3M协议完成训练、验证、保存、续训、50例测试输出。CPU上“连续两epoch”和“第一epoch后恢复”的模型参数逐项完全一致。
- 实际本机GPU为RTX3060 Laptop，不是用户的3090。默认256×64×64、batch1、base16、AMP完成1次预热+3次优化更新，峰值allocated=1,949,082,112 bytes（约1.82GiB），reserved=2,413,821,952 bytes（约2.25GiB）。报告在`artifacts/fm3d_implementation_20261019/local_gpu_benchmark.json`。
- 上述均为合成数据验证，没有运行正式OCTA-500训练，没有确定真实病例名单，没有声称PSNR/SSIM提升或3090性能。具体BMP路径和元数据确认后，才能固定真实数据协议。

## 7. 后续公共数据准备

2026-10-19新增批量准备入口`tools/prepare_common_octa500.py`，从本地实际的`E:/研一/OCTA-500`读取原始BMP。公共数据目录为`data/OCTA500_common_v1`，具体协议与完成状态查看该目录的`README_统一数据协议.md`和`audit.json`。这一数据准备步骤与前面的合成代码测试分开记录；不要在`audit.json`尚未标记complete时启动完整训练。

公共数据完成后直接使用`split_3M_z256.csv`／`split_6M_z256.csv`，无需再次随机划分。该版本保留0..255灰度，训练仍传`--intensity-scale 255`。记录的数组轴序为ZYX，预处理深度已经为256。
