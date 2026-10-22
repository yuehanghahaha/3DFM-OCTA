"""Recompute three-axis means using the display rule matching official FULL maps."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat

plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
ORDER = ('Z', 'X', 'Y')
LABELS = {'Z': ('Y（B-scan 序号）', 'X（BMP 列）'),
          'X': ('Y（B-scan 序号）', 'Z（深度）'),
          'Y': ('X（BMP 列）', 'Z（深度）')}


def stretch(a):
    lo, hi = float(a.min()), float(np.percentile(a, 99))
    b = np.zeros(a.shape, dtype=np.uint8) if hi <= lo else np.rint(
        np.clip((a-lo)/(hi-lo), 0, 1)*255).astype(np.uint8)
    return b, [lo, hi]


def plot_cases(cases, output, zoom):
    fig, axs = plt.subplots(len(cases), 3, figsize=(14, (4.5 if zoom else 7.7)*len(cases)+1), squeeze=False)
    for row, case in enumerate(cases):
        for col, axis in enumerate(ORDER):
            a = case['images'][axis]
            z0 = 0
            if zoom and axis != 'Z':
                z0, z1 = case['roi']
                a = a[z0:z1]
            ax = axs[row,col]
            ax.imshow(a, cmap='gray', vmin=0, vmax=255, interpolation='nearest', aspect='equal',
                      extent=(-.5,a.shape[1]-.5,z0+a.shape[0]-.5,z0-.5))
            plane = {'Z':'XY 平面（官方显示方向）','X':'YZ 平面','Y':'XZ 平面'}[axis]
            ax.set_title(f"{case['case']} / 沿 {axis} 均值投影\n{plane}", fontsize=13, pad=9)
            ax.set_xlabel(LABELS[axis][0], fontsize=11)
            ax.set_ylabel(LABELS[axis][1], fontsize=11)
            ax.tick_params(labelsize=9)
    fig.suptitle('OCTA-500 三轴均值投影'+('：侧向视网膜区域放大' if zoom else '：完整空间范围'),
                 fontsize=20, fontweight='bold', y=.985)
    note = '各方向沿完整轴取均值，各图独立以自身最小值至第 99 百分位拉伸。像素等比例，非物理长度等比例。'
    if zoom:
        note += '\n侧向仅裁剪显示深度范围；未重新计算投影、未重新拉伸。Z 投影仍使用全深度。'
    fig.text(.5,.014,note,ha='center',va='bottom',fontsize=10)
    height = fig.get_size_inches()[1]
    fig.subplots_adjust(left=.075,right=.98,bottom=1.25/height,top=1-1.1/height,
                        wspace=.28,hspace=.35)
    fig.savefig(output.with_suffix('.png'),dpi=160,facecolor='white')
    fig.savefig(output.with_suffix('.pdf'),facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=Path('E:/研一/OCTA-500数据'))
    p.add_argument('--output',type=Path,default=Path('output/figures/OCTA500_三轴均值投影_20261015'))
    p.add_argument('--cases',nargs='+',default=['10301','10401','10500'])
    args=p.parse_args()
    root,out=args.dataset_root.resolve(),args.output.resolve()
    if root==out or root in out.parents:
        p.error('Output must be outside the source dataset.')
    if out.exists() and any(out.iterdir()):
        p.error('Use a new empty output directory.')
    out.mkdir(parents=True,exist_ok=True)
    cases,records=[],[]
    for case in args.cases:
        folder=root/'OCTA_3mm'/'OCTA'/case
        files=sorted(folder.glob('*.bmp'),key=lambda f:int(f.stem))
        if [int(f.stem) for f in files]!=list(range(1,305)):
            raise ValueError(f'Expected scans 1..304: {folder}')
        scans=[]
        digest=hashlib.sha256()
        for file in files:
            digest.update(file.name.encode());digest.update(file.read_bytes())
            with Image.open(file) as im:
                if im.mode!='L' or im.size!=(304,640):
                    raise ValueError(f'Unexpected BMP format: {file}')
                scans.append(np.array(im))
        volume=np.stack(scans,axis=1)  # [Z,Y,X]
        native={'Z':volume.mean(axis=0),'X':volume.mean(axis=2),'Y':volume.mean(axis=1)}
        # Independent calculations from individual scans, avoiding volume axis assumptions.
        np.testing.assert_allclose(native['Z'],np.stack([s.mean(axis=0) for s in scans]),rtol=0,atol=1e-12)
        np.testing.assert_allclose(native['X'],np.stack([s.mean(axis=1) for s in scans],axis=1),rtol=0,atol=1e-12)
        np.testing.assert_allclose(native['Y'],sum(s.astype(np.float64) for s in scans)/len(scans),rtol=0,atol=1e-12)
        display={'Z':native['Z'].T,'X':native['X'],'Y':native['Y']}
        dest=out/case;dest.mkdir()
        images,windows={},{}
        for axis in ORDER:
            images[axis],windows[axis]=stretch(display[axis])
            target=dest/f'{case}_沿{axis}_全轴均值_显示拉伸.png'
            Image.fromarray(images[axis]).save(target)
            np.testing.assert_array_equal(np.asarray(Image.open(target)),images[axis])
        arrays={f'mean_{a}_native':native[a] for a in ORDER}
        arrays.update({f'mean_{a}_display_orientation':display[a] for a in ORDER})
        np.savez_compressed(dest/'三轴均值_原始数值.npz',**arrays)
        with np.load(dest/'三轴均值_原始数值.npz') as saved:
            for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value)
        layers=loadmat(root/'Label'/'GT_Layers'/f'{case}.mat')['Layer']
        roi=[max(0,int(layers[0].min())-1-10),min(640,int(layers[5].max())+10)]
        reference=np.asarray(Image.open(root/'OCTA_3mm'/'Projection Maps'/'OCTA(FULL)'/f'{case}.bmp'))
        delta=np.abs(reference.astype(float)-images['Z'].astype(float))
        rec={'case':case,'source_folder':str(folder),'source_bmp_sha256':digest.hexdigest(),
             'volume_order':'Z,Y,X','volume_shape':list(volume.shape),'operator':'arithmetic mean over the entire axis',
             'native_array_orders':{'Z':'Y,X','X':'Z,Y','Y':'Z,X'},
             'display_array_orders':{'Z':'X,Y','X':'Z,Y','Y':'Z,X'},
             'display_rule':'independently min to percentile 99; clip to [0,1]; round *255',
             'display_windows':windows,'zoom_depth_range':roi,'zoom_is_display_crop_only':True,
             'official_z_exact_pixel_fraction':float((delta==0).mean()),'official_z_max_error':float(delta.max()),
             'official_z_mae':float(delta.mean()),'independent_mean_checks':'pass','saved_array_and_png_checks':'pass'}
        (dest/'metadata.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
        records.append(rec)
        item={'case':case,'images':images,'roi':roi};cases.append(item)
        plot_cases([item],out/f'{case}_三轴均值_完整范围',False)
        plot_cases([item],out/f'{case}_三轴均值_侧向放大',True)
        print(json.dumps(rec,ensure_ascii=False),flush=True)
    plot_cases(cases,out/'三例_三轴均值_完整范围',False)
    plot_cases(cases,out/'三例_三轴均值_侧向放大',True)
    (out/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'说明.md').write_text('''# 三轴均值投影

全部投影由原始 OCTA B-scan 重新计算。沿 X/Y/Z 完整轴取算术均值，不做深度 ROI 投影、不做层掩膜、展平、去噪或锐化。源数据未修改。

体数组 V[Z,Y,X]。Z=mean(axis=0)，X=mean(axis=2)，Y=mean(axis=1)。Z 图转置为行 X、列 Y，以对齐官方 FULL 图；X 图为行 Z、列 Y；Y 图为行 Z、列 X。坐标约定沿用此前分析，并不定义鼻颞侧等解剖方向。

每个方向分别使用该均值图自身最小值和第 99 百分位线性拉伸至 0–255，截断后四舍五入。Z 的方法在三个样本上与官方 FULL 图近乎或完全一致；X/Y 是相同算子与显示规则的扩展，并非数据集提供的官方侧向投影。不同方向分别拉伸，因此显示亮度不能直接用于比较绝对信号强弱。

“完整范围”展示所有深度；“侧向放大”仅截取 X/Y 图的视网膜深度显示区（ILM 全局最小至 BM 全局最大，上下扩展 10 像素），不重算、不重新拉伸；Z 始终为全深度投影。所有面板像素等比例，不代表真实物理长度等比例。

每例文件夹保存三个独立显示 PNG、六个浮点均值数组组成的 NPZ（原始轴序和显示轴序）、metadata.json。原始均值保留输入 BMP 的强度单位，不做独立归一化。训练投影损失应比较统一强度规则下的预测与真实均值，不能直接用分别拉伸后的显示 PNG。

检查：三轴均值分别用原始切片独立重算验证；PNG/NPZ 保存后回读一致；Z 与官方 FULL 的逐像素误差见 manifest.json。原始体为 640×304×304。

复现：在 FlowOCT 根目录执行 tools/generate_octa500_xyz_mean_projections.py。默认处理 10301、10401、10500；重新运行请用 --output 指定新空目录。支持 --dataset-root 和 --cases 参数。
''',encoding='utf-8')
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP validation failed')
    print(f'Output: {out}\nZIP: {archive}',flush=True)


if __name__=='__main__':
    main()
