"""OCTA-500-style slab projections, with explicit native/Z256/display distinctions."""
import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .visualize import font


def project(volume, layers):
    """Layer is [6,Y,X], MATLAB 1-based native 640-depth boundary coordinates.

    Select resampled voxel centers inside closed layer intervals. Use the same
    supplied annotation for prediction and reference, for post-hoc evaluation only.
    Returned projection orientation is [X,Y], matching the distributed BMPs.
    """
    if volume.ndim != 3 or layers.shape != (6, *volume.shape[1:]):
        raise ValueError('Expected volume ZYX and layers 6YX')
    if not np.isfinite(volume).all() or not np.isfinite(layers).all():
        raise ValueError('Nonfinite input')
    if (layers < 1).any() or (layers > 640).any() or (np.diff(layers.astype(float), axis=0) < 0).any():
        raise ValueError('Invalid native layer boundaries')
    centers = ((np.arange(volume.shape[0]) + 0.5) * 640 / volume.shape[0] - 0.5)[:, None, None]
    maps, empty = {'FULL': volume.mean(0, dtype=np.float64).T.astype(np.float32)}, {}
    for name, lower, upper in (('ILM_OPL', 0, 2), ('OPL_BM', 2, 5)):
        mask = (centers >= layers[lower].astype(float) - 1) & (
            centers <= layers[upper].astype(float) - 1)
        count = mask.sum(0)
        empty[name] = int((count == 0).sum())
        # Zero for a ray with no samples in the interval; disclose its count.
        maps[name] = np.where(mask, volume, 0).max(0).T
    return maps, empty


def display(values, limits=None):
    lo, hi = (float(values.min()), float(np.percentile(values, 99))) if limits is None else limits
    scaled = np.zeros_like(values) if hi <= lo else np.clip((values - lo) / (hi - lo), 0, 1)
    return Image.fromarray(np.rint(scaled * 255).astype(np.uint8)), [lo, hi]


def scores(a, b):
    error = a.astype(np.float64) - b
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return {'mae': float(np.abs(error).mean()),
            'psnr': float(-10 * np.log10(max(float((error * error).mean()), 1e-12))),
            'correlation': corr}


def comparison(case, sets, row, output, stretched):
    names = ['FULL', 'ILM_OPL', 'OPL_BM', 'ILM_OPL']
    labels = ['FULL mean', 'ILM-OPL maximum', 'OPL-BM maximum', 'ILM-OPL central 128x128 zoom']
    canvas = Image.new('RGB', (1340, 1535), 'white')
    draw = ImageDraw.Draw(canvas)
    mode = 'Independent min-to-P99 display (NOT used for metrics)' if stretched else 'Fixed display windows (same for prediction and references)'
    draw.text((15, 10), f'{case} | 3D MAE {row["mae"]:.6f} | PSNR {row["psnr"]:.3f} dB',
              font=font(23), fill='black')
    draw.text((15, 45), mode, font=font(19), fill='black')
    for col, label in enumerate(['Prediction Z256', 'Reference Z256', 'Reference native640', 'Official supplied BMP']):
        draw.text((15 + 330 * col, 80), label, font=font(19), fill='black')
    windows = {}
    for r, (name, label) in enumerate(zip(names, labels)):
        top = 120 + 345 * r
        draw.text((15, top), label, font=font(20), fill='black')
        for col, maps in enumerate(sets):
            arr = maps[name]
            # Official BMP is already contrast mapped; never stretch it again.
            limits = (0, 1) if col == 3 else (None if stretched else (0, 80 / 255 if name == 'FULL' else 1))
            im, window = display(arr, limits)
            windows[f'{name}_{col}'] = window
            if r == 3:
                h, w = arr.shape
                im = im.crop((w // 2 - 64, h // 2 - 64, w // 2 + 64, h // 2 + 64))
            canvas.paste(im.resize((304, 304), Image.Resampling.NEAREST), (15 + col * 330, top + 30))
    draw.text((15, 1505), 'Rows=X, columns=Y. GT layers used only for post-hoc projections; generation uses OCT only.',
              font=font(17), fill='black')
    canvas.save(output)
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    source = json.loads((args.input / 'metrics.json').read_text(encoding='utf-8'))
    provenance = json.loads((args.assets / 'manifest.json').read_text(encoding='utf-8'))
    ids = [str(row['case_id']) for row in source['cases']]
    if set(ids) != set(provenance['case_ids']):
        raise ValueError('Projection asset IDs must match the evaluated split')
    args.output.mkdir(parents=True, exist_ok=True)
    records, overview, windows = [], [], {}
    for row in source['cases']:
        case = str(row['case_id'])
        with np.load(args.input / f'{case}.npz', allow_pickle=False) as data:
            if data['axis_order'].item() != 'ZYX':
                raise ValueError('Expected ZYX')
            pred, target = data['octa'], data['octa_gt']
        with np.load(args.assets / f'{case}.npz', allow_pickle=False) as assets:
            layers = assets['layers']
            native = {key: assets[f'native_{key}'] for key in ('FULL', 'ILM_OPL', 'OPL_BM')}
            official = {key: assets[f'official_{key}'].astype(np.float32) / 255
                        for key in ('FULL', 'ILM_OPL', 'OPL_BM')}
        a, empty = project(pred, layers)
        b, _ = project(target, layers)
        record = dict(case_id=case, patient_id=row['patient_id'], volume_mae=row['mae'],
                      volume_psnr=row['psnr'], empty_rays=empty,
                      projections={key: scores(a[key], b[key]) for key in a},
                      resampling_reference={key: scores(b[key], native[key]) for key in a})
        records.append(record)
        arrays = {f'{prefix}_{key}': maps[key] for prefix, maps in (
            ('prediction', a), ('reference_z256', b), ('reference_native', native)) for key in a}
        np.savez_compressed(args.output / f'{case}_projections.npz', **arrays)
        sets = [a, b, native, official]
        windows[case] = {
            'fixed': comparison(case, sets, row, args.output / f'{case}_fixed.png', False),
            'stretched': comparison(case, sets, row, args.output / f'{case}_stretched.png', True)}
        overview.append((case, [display(m['ILM_OPL'])[0] for m in (a, b)]))
        print(f'Projection report {len(records)}/{len(ids)}: {case}', flush=True)
    patients = defaultdict(list)
    for r in records:
        patients[r['patient_id']].append(r)
    aggregate = {}
    for key in ('FULL', 'ILM_OPL', 'OPL_BM'):
        aggregate[key] = {}
        for metric in ('mae', 'psnr', 'correlation'):
            values = [[r['projections'][key][metric] for r in group
                       if r['projections'][key][metric] is not None] for group in patients.values()]
            aggregate[key][metric] = float(np.mean([np.mean(v) for v in values if v]))
    report = {'volume_evaluation': source, 'projection_metrics_patient_macro': aggregate,
              'cases': records, 'assets': provenance, 'display_windows': windows,
              'protocol': 'Source paper section 3.3: FULL mean, ILM-OPL max, OPL-BM max. '
              'Layer intervals closed; native center z=(z256+0.5)*640/256-0.5; MATLAB boundaries minus 1; '
              'projection orientation XY. Raw [0,1] projections used for scores. '
              'GT layer annotations used only for post-hoc evaluation, not generation. '
              'Official BMP is a display reference, not the target for raw projection metrics. '
              'Native reconstructed OPL-BM is not pixel-identical to official BMP.'}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    with (args.output / 'per_case_metrics.csv').open('w', newline='', encoding='utf-8-sig') as file:
        fields = ['case_id', 'volume_mae', 'volume_psnr'] + [
            f'{key}_{metric}' for key in aggregate for metric in ('mae', 'psnr', 'correlation')]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for r in records:
            flat = {key: r[key] for key in fields[:3]}
            flat.update({f'{key}_{metric}': r['projections'][key][metric]
                         for key in aggregate for metric in ('mae', 'psnr', 'correlation')})
            writer.writerow(flat)
    # Paginated contact sheets: all cases, in manifest order, no selection by quality.
    for start in range(0, len(overview), 10):
        sheet = Image.new('RGB', (960, 5 * 260), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (case, pair) in enumerate(overview[start:start + 10]):
            x, y = i % 2 * 480, i // 2 * 260
            draw.text((x + 8, y + 3), f'{case}: prediction | reference (P99 display)',
                      font=font(14), fill='black')
            for col, im in enumerate(pair):
                sheet.paste(im.resize((230, 230), Image.Resampling.NEAREST), (x + 8 + col * 235, y + 25))
        sheet.save(args.output / f'overview_{start // 10 + 1}.png')
    sections = ''.join(
        f'<section id="case-{case}"><h2>{case}</h2><p>上：统一显示窗；下：每图独立拉伸，仅查看形态。</p>'
        f'<a href="{case}_fixed.png"><img loading="lazy" src="{case}_fixed.png"></a>'
        f'<a href="{case}_stretched.png"><img loading="lazy" src="{case}_stretched.png"></a></section>' for case in ids)
    table = ''.join(f'<tr><td>{key}</td><td>{v["mae"]:.6f}</td><td>{v["psnr"]:.3f}</td>'
                    f'<td>{v["correlation"]:.3f}</td></tr>' for key, v in aggregate.items())
    links = ' '.join(f'<a href="#case-{case}">{case}</a>' for case in ids)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>3DFM 测试集分层投影</title>
<style>body{{font:17px system-ui;max-width:1340px;margin:30px auto;padding:0 20px;color:#172536}}
img{{width:100%}}td,th{{padding:8px 22px;border-bottom:1px solid #ccc}}section{{margin-top:50px}}
a{{margin-right:8px}}</style><h1>3DFM：{html.escape(source['split'])}，{len(ids)}例</h1>
<p>固定第{source['checkpoint_epoch']}轮模型，{source['config']['ode_steps']}步Euler。
完整三维体MAE：{source['metrics']['mae']:.6f}；PSNR：{source['metrics']['psnr']:.3f} dB。</p>
<p>依据<a href="https://arxiv.org/pdf/2012.07261">OCTA-500论文§3.3</a>：
FULL为全深度均值；ILM–OPL、OPL–BM为分层最大值。GT层标注仅用于推理后的投影评价，不输入生成网络。
原始640层边界通过采样中心坐标映射到256层。横向方向与官方BMP一致（行X，列Y）。</p>
<p>每图四列：预测256层、真实256层、真实640层重算、原始官方BMP。
固定窗：FULL为[0,80/255]，两种最大值投影为[0,1]。
拉伸图：每张重算投影独立用最小值到第99百分位显示；只用于看形态，会掩盖绝对亮度差，不能据此报指标。
官方BMP保持原样；不保证重算图与官方BMP逐像素一致，尤其OPL–BM。
最后一行是同位置中心128×128区域放大，没有挑选最好区域。</p>
<p>以下投影指标比较同一256层网格的预测与真实投影，使用未拉伸的[0,1]数值。
它们是新增的补充评价，未用于选择模型。病例相同患者先平均、再患者平均。</p>
<table><tr><th>投影</th><th>MAE↓</th><th>PSNR↑</th><th>Pearson相关↑</th></tr>{table}</table>
<p><a href="per_case_metrics.csv">逐病例指标</a><a href="metrics.json">完整配置与显示窗口</a></p>
<p>{links}</p>{sections}</html>"""
    (args.output / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
