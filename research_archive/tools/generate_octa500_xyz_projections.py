"""Generate real X/Y/Z OCTA projections from ordered OCTA-500 3-mm BMPs.

No training files are changed. Arrays retain raw intensity units. Figure axes
use pixel coordinates; panel aspect ratios are independent display choices.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat

plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                     'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 12})
ORDER = ('Z', 'X', 'Y')
AXES = {'Z': 0, 'Y': 1, 'X': 2}  # volume[Z,Y,X]
TITLES = {'Z': '沿 Z 投影：XY 平面（en-face）',
          'X': '沿 X 投影：YZ 平面', 'Y': '沿 Y 投影：XZ 平面'}


def project(volume):
    return ({a: volume.max(axis=i) for a, i in AXES.items()},
            {a: volume.mean(axis=i, dtype=np.float32) for a, i in AXES.items()})


def panel(ax, array, axis, z0=0, vmax=255):
    height, width = array.shape
    top = 0 if axis == 'Z' else z0
    ax.imshow(array, cmap='gray', vmin=0, vmax=vmax, interpolation='nearest',
              extent=(-0.5, width-0.5, top+height-0.5, top-0.5), aspect='auto')
    ax.set_title(TITLES[axis], pad=10)
    ax.set_xlabel(('Y' if axis == 'X' else 'X') + '（像素）')
    ax.set_ylabel(('Y' if axis == 'Z' else 'Z') + '（像素）')
    ax.tick_params(labelsize=10)


def triplets(rows, title, footer, output):
    fig, axs = plt.subplots(len(rows), 3, figsize=(15, 4.2*len(rows)+0.6), squeeze=False)
    for r, (label, arrays, z0, vmax) in enumerate(rows):
        for c, axis in enumerate(ORDER):
            panel(axs[r, c], arrays[axis], axis, z0, vmax)
        axs[r, 0].text(0, 1.17, label, transform=axs[r, 0].transAxes,
                       fontsize=16, fontweight='bold', va='bottom')
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.992)
    fig.text(0.5, 0.015, footer, ha='center', va='bottom', fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88 if len(rows)==2 else 0.91,
                        bottom=0.09 if len(rows)==2 else 0.065, wspace=0.22, hspace=0.63)
    fig.savefig(output.with_suffix('.png'), dpi=180, facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)


def load_volume(folder):
    files = sorted(folder.glob('*.bmp'), key=lambda p: int(p.stem))
    if [int(p.stem) for p in files] != list(range(1, 305)):
        raise ValueError(f'{folder}: expected BMP slices 1..304')
    digest = hashlib.sha256()
    slices = []
    for f in files:
        digest.update(f.name.encode('utf-8')); digest.update(f.read_bytes())
        with Image.open(f) as im:
            if im.mode != 'L' or im.size != (304, 640):
                raise ValueError(f'{f}: unexpected size/mode {im.size}/{im.mode}')
            slices.append(np.array(im))
    # Stack the B-scan number on Y, not on the axial depth Z.
    return np.stack(slices, axis=1), files, digest.hexdigest()


def verify_full_max(files, projs):
    # Independently recompute all axes directly from the source BMP slices.
    pz = np.zeros((304, 304), dtype=np.uint8)
    px = np.zeros((640, 304), dtype=np.uint8)
    py = np.zeros((640, 304), dtype=np.uint8)
    for y, f in enumerate(files):
        with Image.open(f) as im:
            scan = np.array(im)
        pz[y] = scan.max(axis=0)
        px[:, y] = scan.max(axis=1)
        np.maximum(py, scan, out=py)
    for a, reference in {'Z': pz, 'X': px, 'Y': py}.items():
        np.testing.assert_array_equal(projs[a], reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['10301', '10401', '10500'])
    args = parser.parse_args()
    source = args.dataset_root.resolve()
    out = args.output.resolve()
    if out == source or source in out.parents:
        parser.error('Output must be outside the source dataset')
    if out.exists() and any(out.iterdir()):
        parser.error('Use a new empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    overview_full, overview_roi, records = [], [], []
    for case in args.cases:
        dest = out / case; dest.mkdir()
        volume, files, source_hash = load_volume(source / 'OCTA_3mm' / 'OCTA' / case)
        full_max, full_mean = project(volume)
        verify_full_max(files, full_max)
        layers_path = source / 'Label' / 'GT_Layers' / f'{case}.mat'
        layers = loadmat(layers_path)['Layer']
        if layers.shape != (6, 304, 304):
            raise ValueError(f'{layers_path}: unexpected Layer shape')
        # A rectangular Z crop, not a per-ray layer mask or retinal flattening.
        z0 = max(0, int(layers[0].min()) - 1 - 10)
        z1 = min(640, int(layers[5].max()) + 10)
        roi_max, roi_mean = project(volume[z0:z1])
        for a in ('X', 'Y'):
            np.testing.assert_array_equal(roi_max[a], full_max[a][z0:z1])
            np.testing.assert_allclose(roi_mean[a], full_mean[a][z0:z1], rtol=0, atol=1e-5)
        arrays = {}
        for prefix, projs in [('full_mip', full_max), ('full_mean', full_mean),
                              ('roi_mip', roi_max), ('roi_mean', roi_mean)]:
            for axis, arr in projs.items():
                arrays[f'{prefix}_{axis}'] = arr
                if 'mip' in prefix:
                    path = dest / f'{prefix}_{axis}.png'
                    Image.fromarray(arr).save(path)
                    with Image.open(path) as reread:
                        np.testing.assert_array_equal(np.array(reread), arr)
        np.savez_compressed(dest / 'projections_raw.npz', **arrays)
        with np.load(dest/'projections_raw.npz') as saved:
            for key, arr in arrays.items():
                np.testing.assert_array_equal(saved[key], arr)
        mean_vmax = float(np.percentile(np.concatenate([p[p>0] for p in roi_mean.values()]), 99.5))
        triplets([('全体数据 MIP', full_max, 0, 255),
                  (f'深度 ROI MIP：Z ∈ [{z0}, {z1})', roi_max, z0, 255)],
                 f'OCTA-500 / OCTA-3mm / 样本 {case}：真实三轴投影',
                 'MIP = 沿对应轴取最大值，统一灰度窗 0–255。坐标为像素，各面板独立缩放。',
                 out/f'{case}_三轴投影_全体与ROI')
        triplets([('ROI 最大值投影（MIP）', roi_max, z0, 255),
                  (f'ROI 均值投影（AIP），显示窗 0–{mean_vmax:.1f}', roi_mean, z0, mean_vmax)],
                 f'样本 {case}：最大值与均值投影对照',
                 f'ROI 为矩形深度裁剪 [{z0},{z1})，未做逐点层掩膜或展平。均值图仅线性调整显示窗，NPZ 保留原数值。',
                 out/f'{case}_三轴投影_MIP与均值')
        record = {'case': case, 'source_folder': str(files[0].parent), 'source_bmp_sha256': source_hash,
                  'source_volume_order': 'Z,Y,X', 'volume_shape': list(volume.shape),
                  'axis_semantics': {'X':'BMP column / fast lateral scan', 'Y':'numeric B-scan index / slow scan', 'Z':'BMP row / axial depth'},
                  'roi_z0': z0, 'roi_z1_exclusive': z1, 'roi_definition': 'global minimum ILM minus 10 pixels to global maximum BM plus 10 pixels',
                  'roi_layer_source': str(layers_path), 'layer_coordinates': 'MATLAB 1-based converted to Python 0-based',
                  'roi_is_rectangular': True, 'per_ray_layer_mask': False, 'flattened': False, 'spatial_resampling': False,
                  'mip_display_window': [0,255], 'roi_mean_display_window': [0,mean_vmax],
                  'projection_shapes': {key: list(arr.shape) for key,arr in arrays.items()},
                  'independent_source_bmp_mip_check': 'pass', 'png_pixel_roundtrip': 'pass', 'npz_roundtrip': 'pass'}
        (dest/'metadata.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        records.append(record)
        overview_full.append((f'样本 {case} / 全体深度',full_max,0,255))
        overview_roi.append((f'样本 {case} / Z ∈ [{z0}, {z1})',roi_max,z0,255))
        print(json.dumps({'case':case,'shape':list(volume.shape),'roi':[z0,z1],'checks':'pass'},ensure_ascii=False),flush=True)
    triplets(overview_full,'三个真实样本：全体数据三轴最大值投影',
             '原始 OCTA 体：640×304×304（Z,Y,X）。无层掩膜、无裁剪、無空间重采样。灰度窗统一为 0–255。',out/'三例_全体三轴MIP')
    triplets(overview_roi,'三个真实样本：视网膜深度 ROI 的三轴最大值投影',
             'ROI 根据 GT_Layers 的 ILM/BM 全局上下界各扩展 10 像素，保留完整 X、Y 范围。像素坐标，各面板独立缩放。',out/'三例_ROI三轴MIP')
    (out/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(out),flush=True)


if __name__ == '__main__':
    main()
