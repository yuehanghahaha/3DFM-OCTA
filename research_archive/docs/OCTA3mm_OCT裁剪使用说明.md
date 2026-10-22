# OCTA-3mm OCT/OCTA 配对图像裁剪

程序：`tools/crop_oct_3mm.py`。

输入目录：`E:\研一\OCTA-500数据\OCTA_3mm\OCT`。

层标注：`E:\研一\OCTA-500数据\Label\GT_Layers`。

OCT 输出：`E:\研一\OCTA-500数据\OCTA_3mm_OCT_cropped`。

OCTA 输出：`E:\研一\OCTA-500数据\OCTA_3mm_OCTA_cropped`。

## 实际裁剪方式

- 为每个病例读取 `GT_Layers/<ID>.mat` 中的 `Layer`，形状为 `[6,304,304]`。
- 按 MATLAB 行号减 1 得到 Python 坐标，取整卷 ILM 的最小坐标和 BM 的最大坐标。
- 默认在 ILM 上方保留 30 像素，在 BM 下方保留 120 像素。
- 每例的全部 304 张切片使用同一上下范围；保留全部横向列。
- 保存独立的 BMP 图像，保持原病例编号和 `1.bmp` 至 `304.bmp` 的文件名。
- 不缩放、不归一化、不补边；图像高度随病例变化。
- OCT 使用层标注确定坐标；OCTA 直接复用已导出的 OCT 坐标表。原始 OCT/OCTA 和标注文件不修改。

本次定位依据已有层标注，不是一个仅输入 OCT 的自动定位模型。30/120 是基于之前图像检查选出的保守候选余量；裁剪的像素一致性验证不等同于已证明全部生物学目标都被保留。

## 输出结构

```text
OCTA_3mm_OCT_cropped/
  10301/
    1.bmp
    ...
    304.bmp
    crop_metadata.json
  ...
  10500/
    ...
  crop_config.json
  crop_coordinates.csv
  crop_summary.json
  README_裁剪说明.txt
```

`crop_coordinates.csv` 记录每例的 `z0`、`z1_exclusive`、原始及裁剪尺寸和成功保存数量。裁剪区间为 Python 0 基 `[z0,z1_exclusive)`；OCTA 导出直接复用同一坐标，不在 OCTA 上另找边界，确保两种数据的配对位置一致。OCTA 输出目录采用与上述 OCT 相同的结构。

程序保存后会逐张重新读取 BMP，检查模式、尺寸，并逐像素比较原图裁剪区域。`crop_summary.json` 中 `failed_images=0` 且 `saved_verified_images=expected_images` 表示此次导出和像素验证全部完成。读取或验证异常会被记录，并令程序以非零状态结束。

## 重新运行

依赖：Python 3.9 或更高、numpy、scipy、Pillow。本次使用本地 Anaconda 的 Python；依赖已经可用。

在项目目录执行，注意输出必须是新的空目录：

```powershell
python tools/crop_oct_3mm.py --dataset-root "E:\研一\OCTA-500数据" --output "E:\研一\OCTA-500数据\OCTA_3mm_OCT_cropped_new"
```

只处理指定病例：

```powershell
python tools/crop_oct_3mm.py --dataset-root "E:\研一\OCTA-500数据" --output "E:\研一\OCTA-500数据\OCT_crop_subset" --cases 10301 10337
```

修改保留余量：

```powershell
python tools/crop_oct_3mm.py --dataset-root "E:\研一\OCTA-500数据" --output "E:\研一\OCTA-500数据\OCT_crop_other_margin" --margin-top 30 --margin-bottom 90
```

已有内容的输出目录会被拒绝，程序不会覆盖之前导出的数据。

## 导出配对 OCTA

先完成 OCT 导出，再指定其坐标表。输出仍须是新的空目录：

```powershell
python tools/crop_oct_3mm.py --dataset-root "E:\研一\OCTA-500数据" --output "E:\研一\OCTA-500数据\OCTA_3mm_OCTA_cropped_new" --modality OCTA --coordinates "E:\研一\OCTA-500数据\OCTA_3mm_OCT_cropped\crop_coordinates.csv"
```

程序会检查参考 OCT 导出是否完成、两种原图的尺寸是否一致、病例切片文件名是否配对，以及裁剪坐标是否有效。保留余量以参考 OCT 导出的配置为准，不重新计算。保存后的每张 OCTA BMP 同样逐像素验证；既有 OCT 裁剪目录不会修改。
