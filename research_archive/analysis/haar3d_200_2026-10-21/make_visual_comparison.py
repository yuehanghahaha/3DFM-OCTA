"""Make paired OCTA projection galleries from the saved 50-case test outputs.

The fixed-window panels preserve relative brightness. The enhanced panels use a
separate P99 window for each image and are for morphology inspection only.
"""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SLABS = (
    ("FULL", "Full-depth mean", 80 / 255, 0.05),
    ("ILM_OPL", "ILM-OPL maximum", 1.0, 0.2),
    ("OPL_BM", "OPL-BM maximum", 1.0, 0.2),
    ("ILM_OPL", "ILM-OPL center 128x128", 1.0, 0.2),
)
TILE = 248
GAP = 14


def font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def crop_center(array):
    height, width = array.shape
    return array[height // 2 - 64:height // 2 + 64,
                 width // 2 - 64:width // 2 + 64]


def gray(array, maximum):
    values = np.clip(array / max(maximum, 1e-6), 0, 1)
    image = Image.fromarray(np.rint(values * 255).astype(np.uint8), mode="L")
    return image.resize((TILE, TILE), Image.Resampling.NEAREST).convert("RGB")


def error_image(array, maximum):
    values = np.clip(array / maximum, 0, 1)
    rgb = np.stack((values, 0.35 * values, 0.05 * values), axis=-1)
    image = Image.fromarray(np.rint(rgb * 255).astype(np.uint8), mode="RGB")
    return image.resize((TILE, TILE), Image.Resampling.NEAREST)


def read_projection(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    for name, _, _, _ in SLABS[:3]:
        for prefix in ("prediction", "reference_z256"):
            key = f"{prefix}_{name}"
            if key not in result or result[key].ndim != 2 or not np.isfinite(result[key]).all():
                raise ValueError(f"Invalid {key} in {path}")
    return result


def render_case(case_id, baseline, wavelet, baseline_row, wavelet_row, output, enhanced):
    columns = ["Reference", "Voxel baseline", "3D Haar wavelet"]
    if not enhanced:
        columns += ["Baseline absolute error", "Wavelet absolute error"]
    width = 20 + len(columns) * (TILE + GAP)
    height = 170 + len(SLABS) * (TILE + 48)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), f"Case {case_id} | baseline epoch 90 vs Haar epoch 195", font=font(24), fill="black")
    draw.text((20, 47),
              f"MAE: {baseline_row['mae']:.5f} vs {wavelet_row['mae']:.5f}; "
              f"PSNR: {baseline_row['psnr']:.2f} vs {wavelet_row['psnr']:.2f} dB",
              font=font(19), fill="black")
    explanation = ("Independent 0-to-P99 contrast per image: shape only, not brightness"
                   if enhanced else "Fixed windows shared by both models and reference; errors use raw intensities")
    draw.text((20, 78), explanation, font=font(17), fill="#444444")
    for col, title in enumerate(columns):
        draw.text((20 + col * (TILE + GAP), 112), title, font=font(17), fill="black")
    for row_index, (name, label, maximum, error_maximum) in enumerate(SLABS):
        y = 150 + row_index * (TILE + 48)
        suffix = " [0,P99 separately]" if enhanced else f" [0,{maximum:.3f}]"
        draw.text((20, y - 18), label + suffix, font=font(17), fill="#222222")
        reference = baseline[f"reference_z256_{name}"]
        voxel = baseline[f"prediction_{name}"]
        haar = wavelet[f"prediction_{name}"]
        if row_index == 3:
            reference, voxel, haar = map(crop_center, (reference, voxel, haar))
        arrays = (reference, voxel, haar)
        for col, arr in enumerate(arrays):
            upper = float(np.percentile(arr, 99)) if enhanced else maximum
            canvas.paste(gray(arr, upper), (20 + col * (TILE + GAP), y))
        if not enhanced:
            for col, pred in enumerate((voxel, haar), start=3):
                canvas.paste(error_image(np.abs(pred - reference), error_maximum),
                             (20 + col * (TILE + GAP), y))
    draw.text((20, height - 24),
              "All projections use the same Z256 reference grid. Enhanced contrast is display-only.",
              font=font(15), fill="#555555")
    canvas.save(output)


def overview_page(rows, projections, output):
    thumb = 142
    cell_w, cell_h = 3 * (thumb + 5) + 22, thumb + 54
    page = Image.new("RGB", (2 * cell_w + 20, 5 * cell_h + 28), "white")
    draw = ImageDraw.Draw(page)
    draw.text((10, 3), "ILM-OPL projection; each panel independently enhanced to P99 (shape only)",
              font=font(17), fill="black")
    for index, row in enumerate(rows):
        case_id = row["case_id"]
        x, y = 10 + (index % 2) * cell_w, 32 + (index // 2) * cell_h
        delta = row["mae_wavelet"] - row["mae_baseline"]
        draw.text((x, y), f"{case_id} | MAE Haar - baseline {delta:+.5f}",
                  font=font(15), fill="black")
        b, w = projections[case_id]
        images = (b["reference_z256_ILM_OPL"],
                  b["prediction_ILM_OPL"], w["prediction_ILM_OPL"])
        for col, array in enumerate(images):
            upper = float(np.percentile(array, 99))
            image = gray(array, upper).resize((thumb, thumb), Image.Resampling.NEAREST)
            page.paste(image, (x + col * (thumb + 5), y + 24))
            draw.text((x + col * (thumb + 5), y + thumb + 26),
                      ("Reference", "Baseline", "Haar3D")[col], font=font(14), fill="black")
    page.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-projections", type=Path, required=True)
    parser.add_argument("--wavelet-projections", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--wavelet-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    b_report = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    w_report = json.loads(args.wavelet_metrics.read_text(encoding="utf-8"))
    b_cases = {str(row["case_id"]): row for row in b_report["cases"]}
    w_cases = {str(row["case_id"]): row for row in w_report["cases"]}
    if len(b_cases) != 50 or b_cases.keys() != w_cases.keys():
        raise ValueError("Expected the same 50 test cases for both models")
    if (b_report["split"], w_report["split"]) != ("test", "test"):
        raise ValueError("Both reports must evaluate the test split")
    if b_report["config"]["manifest_sha256"] != w_report["config"]["manifest_sha256"]:
        raise ValueError("Manifest hashes differ")
    ids = list(b_cases)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    projections = {}
    for index, case_id in enumerate(ids, start=1):
        b = read_projection(args.baseline_projections / f"{case_id}_projections.npz")
        w = read_projection(args.wavelet_projections / f"{case_id}_projections.npz")
        for name, _, _, _ in SLABS[:3]:
            key = f"reference_z256_{name}"
            if b[key].shape != w[key].shape or not np.allclose(b[key], w[key], atol=1e-7, rtol=0):
                raise ValueError(f"Reference projection mismatch for {case_id} {name}")
        projections[case_id] = b, w
        render_case(case_id, b, w, b_cases[case_id], w_cases[case_id],
                    args.output / f"{case_id}_fixed.png", enhanced=False)
        render_case(case_id, b, w, b_cases[case_id], w_cases[case_id],
                    args.output / f"{case_id}_enhanced.png", enhanced=True)
        rows.append({"case_id": case_id, "mae_baseline": b_cases[case_id]["mae"],
                     "mae_wavelet": w_cases[case_id]["mae"],
                     "psnr_baseline": b_cases[case_id]["psnr"],
                     "psnr_wavelet": w_cases[case_id]["psnr"]})
        print(f"Rendered {index}/50: {case_id}", flush=True)
    for start in range(0, len(rows), 10):
        overview_page(rows[start:start + 10], projections,
                      args.output / f"overview_{start // 10 + 1}.png")
    with (args.output / "per_case_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cards = "".join(
        f'<section id="case-{html.escape(row["case_id"])}"><h2>{html.escape(row["case_id"])}</h2>'
        f'<p>MAE: {row["mae_baseline"]:.5f} → {row["mae_wavelet"]:.5f}; '
        f'PSNR: {row["psnr_baseline"]:.2f} → {row["psnr_wavelet"]:.2f} dB</p>'
        f'<a href="{row["case_id"]}_fixed.png"><img loading="lazy" '
        f'src="{row["case_id"]}_fixed.png" alt="{row["case_id"]} fixed-window comparison"></a>'
        f'<a href="{row["case_id"]}_enhanced.png">查看形态增强图</a></section>'
        for row in rows)
    overviews = "".join(f'<a href="overview_{i}.png"><img loading="lazy" '
                        f'src="overview_{i}.png" alt="overview {i}"></a>' for i in range(1, 6))
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>50例 3D FM 对比图集</title>
<style>body{{font:16px system-ui;max-width:1450px;margin:24px auto;padding:0 16px;color:#192435}}
img{{max-width:100%;height:auto}}section{{margin:55px 0;border-top:1px solid #ccc;padding-top:18px}}
section img{{width:100%}}a{{color:#1d588a}}</style>
<h1>50例测试：3D FM 基线与 3D Haar 小波</h1>
<p>相同测试集、相同 Z256 网格、相同参考图。基线使用 100 轮训练中第
{b_report["checkpoint_epoch"]} 轮最佳权重；小波使用 200 轮训练中第
{w_report["checkpoint_epoch"]} 轮最佳权重。基线续训至 200 轮的测试尚未包含在本图集。</p>
<p>逐例固定窗图：真实图、基线预测、小波预测、两组绝对误差并排。亮度窗统一，适合比较强度和误差；
全层均值为 [0,80/255]，两层最大值为 [0,1]，误差窗全层 [0,0.05]、分层 [0,0.2]。
形态增强图对每张图分别使用 [0,P99]，仅帮助观察血管形态，不能用来比较绝对亮度。
末行是预先固定的 ILM–OPL 中心 128×128 区域，没有挑选局部最佳区域。
所有投影使用相同的真实层边界作测试后评价，层边界没有输入生成模型。</p>
<p><a href="per_case_metrics.csv">下载逐例指标</a></p>
<h2>全部50例速览（每页10例）</h2>{overviews}{cards}</html>'''
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(f"Report: {args.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
