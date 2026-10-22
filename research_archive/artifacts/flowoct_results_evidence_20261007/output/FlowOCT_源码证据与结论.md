# FlowOCT 实验与公开实现核对

核对日期：2026-10-07。本轮只生成汇报材料，没有修改模型或运行训练。结论以论文正文、原始数表和公开代码为依据；未加载作者权重执行推理，不能追溯未公开实现。

## 核对版本

作者仓库：[cnu-medilab/FlowOCT](https://github.com/cnu-medilab/FlowOCT)。本地 upstream 工作树无修改，其 HEAD 与检索时远端 main 均为 **2309126956d2b80a95f2d307d85a50799cc658a6**。下面链接固定到这一提交，避免以后代码更新影响证据。

论文：[FlowOCT: Wavelet Flow Matching for OCT-to-OCTA Translation](https://www.ijcai.org/proceedings/2026/213)，IJCAI 2026，DOI 10.24963/ijcai.2026/213。本地正式版本为 `papers/FlowOCT_IJCAI2026.pdf`。

## 实验结果

用户截图为 Table 2，即 OCTA-6M。各格为全体积/前景区域，不能混为单个数值，也不能混用 Table 1 的 OCTA-3M 数值。

- FlowOCT 的五项前景指标在该表均排名第一。
- 前景 MAE 从最佳其他方法 Flow Matching 的 3.52 降为 3.30，相对减少 `(3.52−3.30)/3.52=6.25%`。
- 前景 SSIM 从 55.83 提升到 58.50，相对增幅约 4.782%。原表缩放标度为 ×10⁻²，实际 SSIM 从 0.5583 到 0.5850，绝对差为 0.0267。
- 前景 PSNR 相对最佳其他方法提高 0.48 dB。SSIM 相对增幅不能表述为增加4.8个百分点。
- 全体积 FID 的 FlowOCT 值为19.38，高于 Adjacent GAN 的14.88和 Pix2pix 2D 的15.23。这里不能用“所有指标都最佳”。
- Table 4 去除 RDCA 后前景 SSIM 为54.88，去除投影约束后为53.66，完整模型为58.50。这支持这些组件在本文设置下的贡献，不是保证任何实现都有效。
- 表2未报告方差和显著性检验，由均值排名不能推出统计显著。

## 二维生成主干的实际证据

[模型配置 L29–38](https://github.com/cnu-medilab/FlowOCT/blob/2309126956d2b80a95f2d307d85a50799cc658a6/models/FlowOCT_model.py#L29-L38)：

```python
self.netG = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=4,
    out_channels=4,
).to(self.device)
self.dwt_2d = DWT_2D('haar')
self.idwt_2d = IDWT_2D('haar')
```

[训练的切片展平 L113–124](https://github.com/cnu-medilab/FlowOCT/blob/2309126956d2b80a95f2d307d85a50799cc658a6/models/FlowOCT_model.py#L113-L124)：

```python
B, C, D, H, W = self.real_A.shape
num_slices = max(1, D // 4)
indices = torch.randperm(D, device=self.device)[:num_slices]
oct_slices = self.real_A[:, :, indices, :, :]
oct_flat = oct_slices.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
```

[二维小波定义 L237–244](https://github.com/cnu-medilab/FlowOCT/blob/2309126956d2b80a95f2d307d85a50799cc658a6/util/DWT_IDWT_layer.py#L237-L244)说明输入 `(N,C,H,W)`，四子带各为 `(N,C,H/2,W/2)`，拼接后网络输入 `(N,4,H/2,W/2)`。4通道是子带，并非相邻切片。

[推理拆片回拼 L245–258](https://github.com/cnu-medilab/FlowOCT/blob/2309126956d2b80a95f2d307d85a50799cc658a6/models/FlowOCT_model.py#L245-L258)：

```python
flat = condition.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
for start in range(0, flat.shape[0], chunk_size):
    chunk = flat[start:start + chunk_size]
    outputs.append(self.generate(chunk))
fake_flat = torch.cat(outputs, dim=0)
return fake_flat.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)
```

**可证实的结论：论文面向体数据转换，但已核对公开实现是二维切片速度网络，输出重新组成体数据，不是完整三维生成主干。** 论文 §3.2 写的是3D DWT，而这一公开生成代码调用2D DWT，两者存在差异。库文件里有DWT_3D类，不代表模型实际使用它。

不要扩大成“完全没有三维信息”：[对比与投影损失 L154–176](https://github.com/cnu-medilab/FlowOCT/blob/2309126956d2b80a95f2d307d85a50799cc658a6/models/FlowOCT_model.py#L154-L176)确实联合统计切片；不过损失层面耦合样本不等于每次前向已接收相邻切片上下文。公开实现投影对展平的随机切片batch做 `max(dim=0)`，也不等同于论文按预定义视网膜层范围的投影。

## OCT 条件：论文与代码不同

论文式(6)、式(12)速度网络为：

\[
v_\theta(w_t,w_0,t).
\]

固定源系数 `w_0` 明确作为条件。按数学定义，每次速度预测都可访问源 OCT。因此“论文未使用持续源条件”的说法不准确。

但[代码推理 L81–107](https://github.com/cnu-medilab/FlowOCT/blob/2309126956d2b80a95f2d307d85a50799cc658a6/models/FlowOCT_model.py#L81-L107)实际为：

```python
x = condition
for i in range(gen_steps):
    x = self.euler_step(x, t_start, t_end)

# euler_step:
LL, LH, HL, HH = self.dwt_2d(x_t)
v_dwt = self.netG(torch.cat([LL, LH, HL, HH], dim=1), t_start)
```

[训练 L130–138](https://github.com/cnu-medilab/FlowOCT/blob/2309126956d2b80a95f2d307d85a50799cc658a6/models/FlowOCT_model.py#L130-L138)同样调用 `self.netG(x_t_dwt,t)`，没有将固定原始 OCT 作为独立拼接或编码条件输入。

**准确结论：公开代码把 OCT 用作初始状态，后续没有持续显式输入固定原始 OCT；这与论文包含 w_0 的速度场接口不同。** 当前状态由 OCT 演化而来，仍可能保留源信息，不能说后续完全不用 OCT，也不能未经实验声称必然遗忘。

## 持续条件是否可行

可行。保持源起点和线性训练路径：

\[
c=\mathcal W(\mathrm{OCT}),\quad y=\mathcal W(\mathrm{OCTA}),
\quad w_t=(1-t)c+ty.
\]

每次调用 `v_theta(cat(w_t,c),t)`，监督目标仍为 `y−c`。二维单通道四子带实现可用8输入通道、4输出通道。训练与推理必须都采用新接口并重新训练，不能直接给原有4通道权重增加输入。

源条件 `c` 在积分全过程保持固定，不把更新后的 `w_t` 当成新的源条件。固定源编码特征可在推理中缓存，训练时应保持需要学习的梯度通路。

一手先例：[Flow Matching for Conditional MRI-CT and CBCT-CT Image Synthesis，2025预印本 §2.2](https://arxiv.org/html/2510.04823v1#S2.SS2)，将固定MRI/CBCT条件编码并与当前状态拼接，其推理ODE为 `v_theta(x_t,t|c)`。这个先例从高斯噪声开始，证明的是固定条件接口可行，不是已证明它比FlowOCT的源起点更好。

建议2×2对照：2D/3D patch主干 × 仅OCT起点/起点加持续条件。固定患者级划分、损失、采样步数和训练预算，记录参数量、显存、耗时及多个随机种子。比较前景MAE/PSNR/SSIM及3D血管连通性，条件分支与3D网络带来的参数增加需要说明或控制。持续条件可能提供稳定参照，也可能过度复制OCT纹理，收益不能预先保证。

因为论文公式本来就写了固定条件，将它补进公开实现首先是实现一致性与消融问题，不能直接宣称新的理论创新。持续源条件也不能代替联合三维上下文。
