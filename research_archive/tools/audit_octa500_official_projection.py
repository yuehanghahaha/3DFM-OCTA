"""Audit official FULL projection against original B-scans, without editing inputs."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('E:/研一/OCTA-500数据/OCTA_3mm')
OLD = Path('output/figures/OCTA500_三轴投影_真实样例_20261014')
OUT = Path('output/figures/OCTA500_官方投影核对_20261015')
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
records = []
fig, axes = plt.subplots(3, 4, figsize=(16, 13))
for row, case in enumerate(['10301', '10401', '10500']):
    files = sorted((ROOT/'OCTA'/case).glob('*.bmp'), key=lambda p: int(p.stem))
    assert [int(p.stem) for p in files] == list(range(1, 305))
    # Independently calculate a row per B-scan: output [Y,X].
    mean_yx = np.stack([np.asarray(Image.open(f)).mean(axis=0) for f in files])
    old = np.load(OLD/case/'projections_raw.npz')
    np.testing.assert_allclose(mean_yx, old['full_mean_Z'], atol=1e-6, rtol=0)
    ref = np.asarray(Image.open(ROOT/'Projection Maps'/'OCTA(FULL)'/f'{case}.bmp'))
    mean_xy = mean_yx.T
    lo, hi = float(mean_xy.min()), float(np.percentile(mean_xy, 99))
    rebuilt = np.rint(np.clip((mean_xy-lo)/(hi-lo), 0, 1)*255).astype(np.uint8)
    Image.fromarray(rebuilt).save(OUT/f'{case}_全深度均值_官方方向_显示拉伸.png')
    Image.fromarray(ref).save(OUT/f'{case}_官方FULL_原像素.png')
    np.save(OUT/f'{case}_全深度均值_官方方向_原始数值.npy', mean_xy)
    # Diagnostic fit only; do not use reference-fitted windows for reconstructed output.
    mask = (ref>0)&(ref<255)
    slope, offset = np.polyfit(mean_xy[mask], ref[mask], 1)
    fit = np.clip(mean_xy*slope+offset, 0, 255)
    error = np.abs(rebuilt.astype(float)-ref.astype(float))
    rec = dict(case=case, reference=str(ROOT/'Projection Maps'/'OCTA(FULL)'/f'{case}.bmp'),
               source_count=len(files), native_mean_order='Y,X', official_display_order='X,Y',
               display_low=lo, display_high=hi, display_rule='min to 99th percentile; clip; round uint8',
               raw_mean_vs_official_correlation=float(np.corrcoef(mean_xy.ravel(), ref.ravel())[0,1]),
               rebuilt_mae=float(error.mean()), rebuilt_max_error=float(error.max()),
               exact_pixel_fraction=float((error==0).mean()), within_one_fraction=float((error<=1).mean()),
               diagnostic_fit_slope=float(slope), diagnostic_fit_offset=float(offset),
               diagnostic_fit_mae=float(np.abs(fit-ref).mean()),
               diagnostic_fit_max_error=float(np.abs(fit-ref).max()))
    records.append(rec)
    images=[old['roi_mip_Z'],old['roi_mip_Z'].T,rebuilt,ref]
    titles=['此前：ROI 最大值投影', '仅转置：ROI 最大值投影',
            '全深度均值＋转置＋显示拉伸', '数据集官方 OCTA(FULL)']
    for col,(im,title) in enumerate(zip(images,titles)):
        ax=axes[row,col]
        ax.imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest',aspect='equal')
        ax.set_title(title,fontsize=12,pad=9)
        ax.set_xlabel('X（BMP 列）' if col==0 else 'Y（B-scan 序号）',fontsize=10)
        ax.set_ylabel(f'{case}\n'+('Y（B-scan 序号）' if col==0 else 'X（BMP 列）'),fontsize=10)
        ax.tick_params(labelsize=8)
    print(json.dumps(rec,ensure_ascii=False),flush=True)
fig.suptitle('OCTA-500 官方投影核对：方向、投影算子和灰度显示',fontsize=20,fontweight='bold',y=.985)
fig.text(.5,.015,'均值图使用自身最小值至第 99 百分位线性拉伸；未去噪、锐化或按官方图拟合。原始均值另存 NPY。',ha='center',fontsize=11)
fig.subplots_adjust(left=.055,right=.985,bottom=.065,top=.93,wspace=.28,hspace=.32)
fig.savefig(OUT/'三例_官方投影核对.png',dpi=180,facecolor='white')
fig.savefig(OUT/'三例_官方投影核对.pdf',facecolor='white')
plt.close(fig)
(OUT/'audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'核对说明.md').write_text('''# 官方投影核对

原始数据没有修改。此前图为 ROI 最大值投影，图像数组 [Y,X]；官方 FULL 图与原始 B-scan 全深度均值的转置 [X,Y] 高度吻合。

本次从原始 304 张 B-scan 重新计算全深度均值，逐例转置，并使用各图自身最小值和第 99 百分位进行线性灰度拉伸、截断和四舍五入。该显示规则没有使用官方图的像素拟合，数值误差记录见 audit.json。不能仅据匹配结果断言官方源码采用完全相同的百分位算法。

对照图第一列保留原先 ROI MIP，第二列仅转置，第三列为全深度均值与显示拉伸，第四列是官方原图。第三列同时改变了深度范围、投影算子、灰度显示；第二列单独展示方向差异。

原始均值数值另存 NPY。显示用拉伸 PNG 不宜直接用作训练时的投影损失：预测和真实体应在统一方向、深度范围及强度规则下比较，避免各自独立拉伸掩盖强度误差。

此前 MIP 的数组计算检查通过，但没有核对官方图的显示方向和投影生成方法，因此不能当作官方 FULL 投影的复现。侧向投影的带状外观仍需在均值、最大值及局部范围之间进一步比较，不能仅凭此前 MIP 图评价三轴监督效果。
''',encoding='utf-8')
