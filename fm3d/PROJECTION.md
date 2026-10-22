# 体素 3D FM：Z 均值投影约束与 96×96 横向块

新入口为 `python -m fm3d.train_projection`：默认从头训练 200 轮，
块尺寸 **[Z,Y,X]=[256,96,96]**，base16、batch1、学习率 2e-4、AMP，
每轮 200 个随机训练批次。原 `fm3d.train` / `fm3d.train_wavelet` 的默认
块尺寸和损失保持兼容；`--lambda-projection 0` 可关闭新项。

## 损失定义

```text
c = OCT，y = OCTA；[B,1,Z,Y,X]，固定归一化到 [0,1]
P_z(a) = mean(a, dim=2)                       # [B,1,Y,X]
t ~ Uniform(0,1)
L_FM = MSE(v((1-t)c+t*y, t, c), y-c)          # 原始 FM 不变
y_start = c + v(c, 0, c)                     # 额外一次仅输入 OCT 的前向
L_projection = L1(P_z(y_start), P_z(y))
L_total = L_FM + 0.1 * L_projection
```

投影使用全深度均值，FP32 累加，不逐图 min-max，不读取分层标签或官方投影 BMP。
[TransPro 源码](https://github.com/ustlsh/TransPro/blob/main/models/TransPro_model.py)
也采用均值投影；这里仅借鉴投影方式，并非复现其完整损失。
本项目 Z 位于 `dim=2`，不能直接照抄其他轴序代码中的 `dim=3`。

使用起点估计是因为当前基线曾出现“真实插值路径拟合好、纯 OCT 起步生成差”的诊断。
投影分支网络输入没有 OCTA，真值仅用于计算监督误差。
训练的起点估计保持未裁剪，避免超出 [0,1] 后梯度被截断。
这是一阶终点估计的辅助监督，**不等同于对 8 步自由积分结果直接反传**；
正式验证仍从 OCT 独立进行原来的 8 步全卷滑窗推理，仅最终图像裁剪。
0.1 是本次初始实验权重，未宣称已经调优。均值投影不能唯一确定三维血管结构。

## 启动、检查和恢复

从包含 `fm3d/` 的代码根目录运行：

```bash
python -m unittest fm3d.test_core fm3d.test_wavelet fm3d.test_projection -v
python -m fm3d.benchmark --patch 256 96 96 --lambda-projection 0.1 --device cuda --output verification/zproj96_memory.json
bash fm3d/scripts/train_projection_200.sh /ABS/split_3M_z256.csv 3M runs/fm3d_zproj_3M_96_200
```

服务器可通过环境变量 `PYTHON` 指定已有训练环境，通过 `CUDA_VISIBLE_DEVICES` 选择已分配 GPU。
恢复时使用相同命令并加 `--resume`；仍是总计 200 轮。
改变 patch 或投影权重会拒绝续训，需使用独立实验目录。

脚本每轮完整验证，记录三维 MAE/PSNR 和全 Z 均值投影 MAE/PSNR/PCC，
均先按患者汇总再宏平均；恒定投影的 PCC 记 null，汇总排除该值。
记录 `fm_loss`、`z_projection_loss`、总 `loss`、实际优化更新数与 AMP 跳过数。
仍按最低三维验证 MAE 保存 `best.pt`，每轮保存 `latest.pt`，
每 5 轮额外保存 `epoch_XXXX.pt` 和 `best_through_XXXX.pt`，以保留后续分析所需权重。
测试集不参与训练或选模。

本次同时改变投影监督与横向块尺寸，结果属于组合实验；
若要归因，需要补做 96 块但投影权重为 0 的对照。
