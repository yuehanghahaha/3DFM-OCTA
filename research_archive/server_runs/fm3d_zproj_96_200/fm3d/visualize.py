"""Create fixed-window comparison PNGs and an HTML report from evaluate outputs."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for name in ('DejaVuSans.ttf', 'Arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.0 has no scalable bundled default font.
        return ImageFont.load_default()


def tile(values, maximum, error=False):
    scaled = np.clip(values / maximum, 0, 1)
    if error:
        rgb = np.stack([scaled, scaled * 0.25, np.zeros_like(scaled)], axis=-1)
    else:
        rgb = np.repeat(scaled[..., None], 3, axis=-1)
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).resize(
        (360, 300), Image.Resampling.NEAREST)


def render_case(path, destination, record, projection_max, error_max):
    with np.load(path, allow_pickle=False) as data:
        if str(data['axis_order'].item()) != 'ZYX':
            raise ValueError(f'{path}: expected ZYX')
        pred, gt = data['octa'], data['octa_gt']
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(f'{path}: invalid paired 3D shapes')
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError(f'{path}: nonfinite values')
    # All cases and both modalities share the same display windows.
    planes = [('Full-depth Z mean (not layer segmented)', pred.mean(0), gt.mean(0),
               projection_max)]
    for fraction in (0.25, 0.5, 0.75):
        y = min(int(pred.shape[1] * fraction), pred.shape[1] - 1)
        planes.append((f'B-scan Y={y} (Z vertical, X horizontal)',
                       pred[:, y, :], gt[:, y, :], 1.0))
    canvas = Image.new('RGB', (1140, 1550), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), f'Case {record["case_id"]} | MAE {record["mae"]:.6f}'
              f' | PSNR {record["psnr"]:.3f} dB', font=font(23), fill='black')
    draw.text((20, 45), f'ZYX={pred.shape}; panels resized for display, not physical aspect ratio',
              font=font(16), fill='black')
    for x, label in zip((20, 400, 780), ('Predicted OCTA', 'Reference OCTA', 'Absolute difference')):
        draw.text((x, 75), label, font=font(22), fill='black')
    for i, (label, a, b, maximum) in enumerate(planes):
        top = 115 + i * 350
        draw.text((20, top), f'{label}; grayscale 0..{maximum:.4f}',
                  font=font(18), fill='black')
        for x, image in zip((20, 400, 780),
                            (tile(a, maximum), tile(b, maximum),
                             tile(np.abs(a - b), error_max, error=True))):
            canvas.paste(image, (x, top + 30))
    draw.text((20, 1520), f'Error: black=0, orange>={error_max:g}; '
              'projection error = abs(mean(pred)-mean(reference)).', font=font(15), fill='black')
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Directory with metrics.json and NPZs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--projection-max', type=float, default=80 / 255,
                        help='Fixed Z-mean display upper bound; never changes metrics')
    parser.add_argument('--error-max', type=float, default=0.2)
    args = parser.parse_args()
    if min(args.projection_max, args.error_max) <= 0:
        raise ValueError('Display bounds must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    report = json.loads((args.input / 'metrics.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    rows, entries = report['cases'], []
    for row in rows:
        case = str(row['case_id'])
        if Path(case).name != case:
            raise ValueError('case_id must be a filename')
        render_case(args.input / f'{case}.npz', args.output / f'{case}.png',
                    row, args.projection_max, args.error_max)
        entries.append(f'<section><h2>{html.escape(case)}</h2>'
                       f'<a href="{html.escape(case)}.png"><img loading="lazy" '
                       f'src="{html.escape(case)}.png"></a></section>')
        print(f'Rendered {case}', flush=True)
    with (args.output / 'per_case_metrics.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=['case_id', 'patient_id', 'mae', 'psnr'])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    table = ''.join(f'<tr><td>{html.escape(str(r["case_id"]))}</td>'
                    f'<td>{r["mae"]:.6f}</td><td>{r["psnr"]:.3f}</td></tr>' for r in rows)
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>3DFM 推理结果</title><style>body{{font:17px system-ui;max-width:1140px;margin:30px auto;
padding:0 20px;color:#172536}}img{{width:100%}}td,th{{padding:8px 25px;border-bottom:1px solid #ddd}}
section{{margin-top:40px}}code{{overflow-wrap:anywhere}}</style>
<h1>3DFM 推理结果</h1>
<p>集合：{html.escape(report['split'])}；模型轮次：{report['checkpoint_epoch']}；
病例数：{len(rows)}；患者数：{report['patients']}；Euler步数：{report['config']['ode_steps']}。</p>
<p>患者宏平均 MAE：<b>{report['metrics']['mae']:.6f}</b>；
PSNR：<b>{report['metrics']['psnr']:.3f} dB</b>。</p>
<p>指标在完整预处理三维体上计算，强度范围为[0,1]。图像仅用于查看，显示窗口不改变指标。
左列为预测、中列为真实OCTA、右列为绝对差值。第一行是全深度平均投影，并非分层血管投影；
其余三行是固定25%、50%、75%位置的B-scan。所有病例使用相同显示窗口，未逐图调整对比度。
图像面板按固定大小显示，不代表真实物理比例。点击图片可放大。</p>
<p><a href="per_case_metrics.csv">下载逐病例指标</a> · <a href="metrics.json">完整评价配置和指标</a></p>
<table><tr><th>病例</th><th>MAE ↓</th><th>PSNR ↑</th></tr>{table}</table>
{''.join(entries)}</html>"""
    (args.output / 'index.html').write_text(content, encoding='utf-8')
    print(f'Report: {args.output / "index.html"}')


if __name__ == '__main__':
    main()
