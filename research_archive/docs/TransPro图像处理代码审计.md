# TransPro 图像处理代码审计

审计对象：当前项目根目录下的 `TransPro` 官方仓库。

- 远程仓库：`https://github.com/ustlsh/TransPro.git`
- 当前提交：`dc748eac63ebffadfe191e96ef85e45e5d3539a0`
- XOCT 已下载到同级目录 `XOCT`，当前提交为 `7d5596b4422e303e64616e8ce85c7620c4e3a133`。

## 一、官方 3D 训练/测试的实际数据路径

README 的 3D 训练命令使用 `--dataset_mode alignedoct2octa3d`、`--load_size 304`、默认 `crop_size=256`、`--no_flip`：

- `TransPro/README.md:55-58`
- `TransPro/options/base_options.py:41-57`

数据集工厂根据 `dataset_mode` 动态加载 `data/alignedoct2octa3d_dataset.py`：

- `TransPro/data/__init__.py:18-38, 47-79`

## 二、输入读取与配对

文件：`TransPro/data/alignedoct2octa3d_dataset.py`

- `:29-33`：在 `phase/A` 和 `phase/B` 中枚举 OCT、OCTA 文件，并分别排序。
- `:36-39`：只检查 A/B 文件数量相等，未检查每个 NPY 的空间尺寸、体素间距或病人 ID 内容。
- `:54-57`：使用 `np.load` 读取配对 NPY。仓库在线代码没有实现原始 BMP 到 NPY 的打包。
- `:58-59`：在最前面增加一个通道维，输入从 `[D,H,W]` 变为 `[1,D,H,W]`。
- `:61-68`：为 A、B 生成同一份 `transform_params`，再分别应用同一空间变换，保证 OCT 和 OCTA 的裁剪区域一致。

需要注意：`data/image_folder.py:12-32` 将扩展名 `npy` 纳入可枚举文件，但 `default_loader` (`:35-36`) 是给普通图像用的 PIL 读取器；3D 数据集实际绕过它，用 `np.load` 读取 NPY。

## 三、3D 在线预处理：缩放、裁剪、翻转

文件：`TransPro/data/base_dataset3d.py`

### 1. 裁剪参数

`get_params` 位于 `:66-83`：

```python
_, d, w, h = size
if opt.preprocess == 'resize_and_crop':
    new_d = new_h = opt.load_size
y_ini = random.randint(0, max(0, new_h - opt.crop_size))
d_ini = random.randint(0, max(0, new_d - opt.crop_size))
return {'crop_pos': (d_ini, d_end, x_ini, x_end, y_ini, y_end), 'flip': flip}
```

训练命令中 `load_size=304`、`crop_size=256`，因此两个平面方向的裁剪起点分别在 `0..48` 随机选择；不是固定中心起点 `24`。中间方向的 `x_ini=x_end=0`，因此该方向不额外裁掉边缘。

### 2. 线性插值缩放

`get_transform` 位于 `:86-107`，默认 `preprocess='resize_and_crop'` 时执行：

```python
osize = (opt.load_size, opt.crop_size, opt.load_size)
transform_list.append(tio.Resize(osize, "linear"))
```

所以，如果输入 NPY 是原始体积排列 `[304,640,304]`，3M 训练在线处理是：

```text
[304,640,304] --线性插值--> [304,256,304]
                  --随机裁剪--> [256,256,256]
```

6M 若输入为 `[400,640,400]`，且用户把训练参数设为 `load_size=400`，对应流程是：

```text
[400,640,400] --线性插值--> [400,256,400]
                  --随机裁剪--> [256,256,256]
```

公开 README 只明确给出了 3M 训练命令，因此不能把 `load_size=400` 当成作者公开 6M 命令的已证实配置。

### 3. 裁剪实现

- `:93-97`：有参数时使用 `tio.Crop`，按前面生成的六个边距裁剪；A/B 共享参数。
- `:94-95`：没有参数时会尝试 `transforms.RandomCrop(opt.crop_size)`。这属于备用分支，官方配对 3D 数据集总是传入 `params`。

### 4. 翻转

- `:81` 随机生成 `flip` 标志。
- `:102-106`：只有 `no_flip=False` 且 `params['flip']` 为真时才执行 `tio.RandomFlip(axes=1, ...)`。
- README 训练命令包含 `--no_flip`；`test3d.py:13-16` 也强制 `opt.no_flip=True`，因此官方示例训练/测试均关闭翻转。

## 四、测试时的特殊尺寸设置

文件：`TransPro/options/test_options.py:18-22`

```python
parser.set_defaults(load_size=parser.get_default('crop_size'))
```

默认测试时 `load_size=crop_size=256`，所以如果输入保留完整扫描范围，在线变换为：

```text
3M: [304,640,304] --> Resize [256,256,256] --> Crop 边距全为 0
6M: [400,640,400] --> Resize [256,256,256] --> Crop 边距全为 0
```

这表示默认测试不是把训练阶段的某个随机 `256³` patch 拼回整卷；它是把当前输入 NPY 整体重采样到 `256³` 后直接生成。

## 五、归一化与张量布局

文件：`TransPro/data/alignedoct2octa3d_dataset.py:69-75`

```python
A = (A-A.min())/(A.max()-A.min())
B = (B-B.min())/(B.max()-B.min())
A = 2*A - 1
B = 2*B - 1
A = torch.from_numpy(A)
B = torch.from_numpy(B)
```

含义：A、B 各自独立做当前体积的 min-max 归一化，再映射到 `[-1,1]`。这不是使用整个训练集统一的固定均值/标准差；如果体积是常数，分母为零。

文件：`TransPro/models/TransPro_model.py:81-89`

```python
self.real_A = input[...].permute(1,0,2,3,4)
```

DataLoader 后的形状通常是 `[B,C,D,H,W]`；这里交换前两个维度，在单通道、batch=1 时数值形状仍表现为 `[1,1,256,256,256]`，但写法不适用于一般 batch 或多通道情况。

## 六、网络内部的 3D 图像变换

真正被 `TransProModel` 实例化的生成器是：

- `TransPro/models/TransPro_model.py:26-29`
- `TransPro/models/networks_2g_st.py:94-170`

它是 3D U-Net 风格编码器-解码器：

- `:99-112`：八层 `Conv3d`，步长为 2，连续下采样。
- `:116-130`：八层 `ConvTranspose3d`，连续上采样。
- `:150-166`：编码器和解码器特征通过 `torch.cat` 跳跃连接。
- `:167-170`：输出经过 `torch.tanh`，保持在 `[-1,1]`。

这里没有第二次图像插值缩放；网络中的尺寸变化来自卷积步长 2 和转置卷积步长 2。

README 命令中的 `--netG unet_256` 不能用来判断实际 3D 生成器，因为 `TransProModel` 没有依据这个参数选择生成器，而是直接调用 `networks_2g_st.generator3D(opt.ngf)`。

## 七、投影、辅助 2D 分支和损失中的图像处理

文件：`TransPro/models/TransPro_model.py`

- `:85-88`：对 OCT 和真实 OCTA 沿第 3 个空间维度做 `torch.mean(..., dim=3)`，形成 2D 平均投影，并用 `Norm` 映射到 `[-1,1]`。
- `:91-95`：3D 生成结果 `fake_B` 同样沿第 3 个空间维度做平均投影；输入 OCT 投影送入冻结的 2D HCG 生成器 `netG_t`。
- `:31-48`：训练时加载预训练 HCG 和 VPG 权重并冻结参数；这两个辅助模块不是测试时的 3D 主生成器。
- `:111-135`：3D 判别器和 2D 投影判别器分别处理体积和投影。
- `:145-160`：投影 L1、HCG 投影一致性、VPG 分割 logits 一致性和体积 L1 共同组成生成器损失。
- `:208-212`：`Norm` 用当前张量 min-max 归一化到 `[-1,1]`，同样存在常数张量除零风险。

训练脚本中 `TransPro/train3d.py:14-18, 80-91` 还定义了验证用的 `Norm` 和体积 MAE；`real_A_proj` 在验证代码中被计算但没有参与验证 MAE。

## 八、输出保存与可视化

文件：`TransPro/models/TransPro_model.py:191-201`：将三维张量取出，交给 `tensor2im3d`。

文件：`TransPro/util/util3d.py:18-20`：`tensor2im3d` 取 batch 0、channel 0，返回三维 NumPy 体，不恢复原始空间尺寸。

文件：`TransPro/util/util3d.py:40-50`：

- 遍历三维数组的第 1 维；
- 将每一张二维切片复制为三通道；
- 将 `[-1,1]` 线性映射到 `[0,255]`；
- 转为 `uint8`，逐张保存为 PNG。

文件：`TransPro/util/visualizer3d.py:179-198` 调用上述函数。这里保存的是切片 PNG，不是恢复后的 3D NPY/NIfTI；没有拼接、反缩放或原始坐标记录。

## 九、仓库中存在但官方 3D 主流程不使用的处理代码

不要把下列代码误认为 TransPro 3D 主路径：

- `TransPro/data/base_dataset.py`：普通 2D CycleGAN 数据处理，包括 PIL 的 BICUBIC resize、2D crop 和 horizontal flip。
- `TransPro/data/aligned_dataset.py`、`single_dataset.py`、`unaligned_dataset.py`、`colorization_dataset.py`：普通 2D 数据集。
- `TransPro/models/networks.py`：2D 生成器/判别器。
- `TransPro/models/networks3d.py:87-105`：通用 3D 生成器工厂和其他 3D 网络定义；当前 `TransProModel` 的 `netG` 没有调用这里的生成器，只用它的 3D 判别器工厂。
- `TransPro/data/base_dataset3d.py:91-100, 110-137`：`scale_width`、`none`、`__make_power_2` 等继承自通用 CycleGAN 的备用分支；README 的官方 3D 命令使用 `resize_and_crop`，不会进入这些分支。它们的实现仍按普通 2D `img.size`/PIL 逻辑编写，不能自动视为已验证的 3D 处理路径。

## 十、结论

TransPro 3D 主流程中真正改变医学体数据的代码集中在：

1. `data/alignedoct2octa3d_dataset.py:54-75`：读取、加通道、共享变换、独立 min-max 归一化。
2. `data/base_dataset3d.py:66-107`：线性插值、随机裁剪、可选翻转。
3. `models/TransPro_model.py:81-95`：张量布局和 3D→2D 平均投影。
4. `models/networks_2g_st.py:94-170`：3D U-Net 的卷积下采样、转置卷积上采样和跳跃连接。
5. `util/util3d.py:18-50`：取出三维结果并逐切片保存为 PNG。

当前官方代码没有提供“训练随机 patch → 测试滑窗覆盖整卷 → 重叠区域融合 → 恢复原始 640 深度”的完整逆处理流程。
