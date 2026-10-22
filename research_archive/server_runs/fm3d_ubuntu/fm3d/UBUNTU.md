# Ubuntu / RTX 3090 运行说明

代码未使用Windows专用接口，可部署到Ubuntu。此前测试发生在Windows/RTX3060 Laptop；Ubuntu/3090是否运行正常，要在服务器上通过下面的检查确认。

## 1. 上传哪些文件

只需上传整个`fm3d/`目录及你自己的数据/病例CSV。基础实现不依赖`working/`、`upstream/`、MONAI或TorchIO，也不用上传本地Conda环境和`__pycache__`。

压缩包`fm3d_ubuntu.zip`只包含源代码与说明，不包含医学数据、训练权重或Python环境。示例目录：

```text
~/projects/oct_fm/
├── fm3d/
├── data/fm3d/
│   ├── 3M/                 每例一个NPZ
│   ├── 6M/
│   ├── split_3M.csv
│   └── split_6M.csv
└── runs/
```

所有`python -m fm3d...`命令从包含`fm3d/`的父目录运行，不能在`fm3d/`目录内执行。

## 2. 建立环境

以下假设服务器已有Conda，且NVIDIA驱动支持所选CUDA版PyTorch。先查看GPU：

```bash
nvidia-smi
conda create -n octfm3d python=3.10 -y
conda activate octfm3d
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 'Pillow>=10.0'
```

这里固定PyTorch 2.5.1以匹配此前测试环境，并非要求安装最新版。CUDA 12.1安装源依据[PyTorch官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v251)。若驱动与此构建不兼容，先核对驱动再选择官方提供的兼容构建，不要只看服务器上`nvcc`的版本。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable')"
```

确认CUDA为True且设备为预期的3090。训练时显式加`--device cuda`，避免GPU不可用时自动落到CPU。

## 3. 检查与测显存

假设代码解压到了`~/projects/oct_fm`：

```bash
cd ~/projects/oct_fm
python -m unittest fm3d.test_core -v
python -m fm3d.benchmark --device cuda --patch 256 64 64 --base-channels 16 --output runs/3090_memory.json
```

单元测试会生成临时合成数据，不需要真实OCTA-500。第二条命令会进行合成体的前向、反向和优化器更新，报告服务器自己的显存占用。

## 4. 数据与路径

- BMP/NPZ无需因操作系统变化而转换格式。Linux区分文件名大小写；转换器当前匹配小写`.bmp`。
- CSV的`path`推荐使用相对CSV所在目录的路径，如`3M/10301.npz`，统一用正斜杠`/`。不要保留`D:\...`、`C:\...`或Windows反斜杠路径。
- 迁移已固定的划分时，保留`case_id/patient_id/split`，仅修改`path`；不要重新随机分配病例。
- 新实验在服务器上创建输出目录和checkpoint。当前checkpoint记录绝对数据路径，Windows训练得到的checkpoint不能直接作为Linux上的无修改续训配置。权重本身可跨平台加载，但本版本尚未提供移动checkpoint的数据路径重映射入口。
- 如使用本地已准备的`OCTA500_common_v1`，上传整个数据目录并使用其`split_3M_z256.csv`／`split_6M_z256.csv`；不要重复打包或重新划分。完成状态及核验命令见数据包的`README_统一数据协议.md`。体数据仍为0..255，因此仍指定`--intensity-scale 255`。
- 真实数据清单含`leakage_group`：3M的10333和10388影像完全重复，已共同放入训练集。保留此字段并使用最终`split_*.csv`；不要使用`assignment_provisional.json`。训练140个编号对应139份不同影像，细节必须随统一实验协议披露。
- 每例NPZ须含`oct`/`octa`，两者三维shape一致。下例假设数组轴序ZYX，灰度仍为0..255；其他数据必须相应调整。

```bash
python -m fm3d.data check --manifest data/fm3d/split_3M.csv --fov 3M --axis-order ZYX --intensity-scale 255
```

病例清单格式、BMP打包命令和固定划分方法见`README.md`。

## 5. 正式训练和测试

```bash
python -m fm3d.train \
  --manifest data/fm3d/split_3M.csv --fov 3M \
  --axis-order ZYX --intensity-scale 255 \
  --depth 256 --patch 256 64 64 --base-channels 16 \
  --batch-size 1 --epochs 100 --steps-per-epoch 200 \
  --val-every 5 --ode-steps 8 --device cuda --workers 0 \
  --output runs/fm3d_3M
```

6M实验将`split_3M.csv`、`--fov 3M`和输出目录分别换为6M；不要共用输出目录。`workers=0`用于先验证数据读取，之后可按服务器RAM与磁盘性能增加。

```bash
python -m fm3d.evaluate --checkpoint runs/fm3d_3M/best.pt --split test --output runs/fm3d_3M_test --device cuda
```

正式长训练可放在服务器的tmux会话内执行，以便SSH断开后继续。推理步数和其他超参数应在验证集选择后固定。

## 6. 已准备数据的直接入口

代码包只含代码、协议说明和固定清单，不含约67.8 GiB的体数据与附件。另行上传`OCTA500_common_v1`；若只做3M，可先上传`z256/3M/`（约8.89 GiB）及`split_3M_z256.csv`，保持目录结构。

例如数据在`/data/OCTA500_common_v1`时，先运行一个epoch、2个训练批次和1步ODE的完整10例验证，验证服务器读取、训练、全体积拼接和保存流程。这个短跑仅检查流程，不能用来判断方法效果：

```bash
python -m fm3d.train \
  --manifest /data/OCTA500_common_v1/split_3M_z256.csv --fov 3M \
  --axis-order ZYX --intensity-scale 255 --depth 256 \
  --patch 256 64 64 --base-channels 16 --batch-size 1 \
  --epochs 1 --steps-per-epoch 2 --val-every 1 --ode-steps 1 \
  --device cuda --workers 0 --output runs/fm3d_3M_pipeline_check
```

此命令会覆盖验证集完整视野，耗时高于只做一次体块前向。完成后应有`latest.pt`、`best.pt`、`validation_0001.json`和`history.jsonl`。正式实验另外新建输出目录，将epochs设为100、steps-per-epoch设为200、val-every设为5、ode-steps设为8，不从短跑权重续训。6M改用6M清单和`--fov 6M`。

本地已完成真实训练病例的体块读入、梯度更新和局部推理检查；尚未在3090服务器执行上述完整验证短跑，也未完成正式性能实验。
