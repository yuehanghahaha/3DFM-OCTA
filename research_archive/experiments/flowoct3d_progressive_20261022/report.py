"""Cumulative-removal comparisons, original FlowOCT controls, and mechanism evidence."""
import csv
import html
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from flowoct3d.utils import read_json, write_json


ZH = {
    '00_flowoct_public': '公开代码 FlowOCT', '01_full3d': '完整三维模型',
    '02_minus_rdca': '累计去对比', '03_minus_rdca_projection': '再去投影',
    '04_minus_rdca_projection_structure': '再去结构加权', '05_haar_fm_only': '再去重建：小波 FM',
    '06_voxel_fm_only': '再去小波：体素 FM', '07_full_half_start': '完整三维＋50%起点',
}
SHORT = {'00_flowoct_public': 'Public FlowOCT', '01_full3d': 'Full 3D', '02_minus_rdca': '- RDCA',
    '03_minus_rdca_projection': '- projection', '04_minus_rdca_projection_structure': '- structure',
    '05_haar_fm_only': '- voxel L1', '06_voxel_fm_only': '- Haar', '07_full_half_start': 'Full + 50% t=0'}


def images(names, config, update, split, output):
    selection = read_json(ROOT/'data/selection.json')
    samples = [r for r in selection['patches'] if r['split'] == split and r['patch_id'] == 'y064_x064']
    if not all((ROOT/name/f'predictions_{update:05d}').exists() for name in names):
        return
    fig, axs = plt.subplots(len(samples), len(names)+1, figsize=(2.05*(len(names)+1), 8.5), constrained_layout=True)
    for i, sample in enumerate(samples):
        with np.load(ROOT/sample['file']) as data:
            target, layers = data['octa'], data['layers']
        z = ((np.arange(len(target))+.5)*640/len(target)-.5)[:, None, None]
        mask = (z >= layers[0]-1) & (z <= layers[2]-1)
        arrays = [np.where(mask, target, 0).max(0)]
        for name in names:
            with np.load(ROOT/name/f'predictions_{update:05d}'/(sample['case_id']+'_'+sample['patch_id']+'.npz')) as data:
                arrays.append(np.where(mask, data['octa'], 0).max(0))
        for j, a in enumerate(arrays):
            axs[i, j].imshow(a, cmap='gray', vmin=0, vmax=.5, interpolation='nearest')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if i == 0:
                axs[i, j].set_title(['Target', *[SHORT[n] for n in names]][j], fontsize=10)
            if j == 0:
                axs[i, j].set_ylabel(sample['case_id']+' '+sample['disease'])
    fig.suptitle(f'{split} patients | {update} successful updates | fixed first crop, ILM-OPL max, display [0,0.5]')
    fig.savefig(output/f'{split}_{update:05d}.png', dpi=160)
    plt.close(fig)


def diagnosis_text():
    path = ROOT/'verification/diagnosis_prior800.json'
    if not path.exists():
        return []
    records = [r for r in read_json(path)['records'] if r['model'] == 'prior_full_800' and r['split'] == 'train']
    lines = ['## 为什么此前生成效果差', '',
        '前一轮的“完整模型”是三维研究扩展，包含自定结构权重、层范围和辅助终点实现；并非作者原代码。'
        '新实验把公开代码 FlowOCT 单独加入，避免把三维扩展的结果当成作者方法的结果。', '',
        '**最直接的现象是：含真值的训练状态估计明显好于纯 OCT 生成，连训练块的血管映射也未充分学好。**', '',
        '以下使用上轮 800 次完整三维模型，每位训练患者固定第一块；不是新模型的测试成绩。', '',
        '| 训练病例 | 含50%真值的状态估计 PCC | 纯OCT起点单步 PCC | 纯OCT八步生成 PCC |',
        '|---|---:|---:|---:|']
    for r in records:
        q = {v['mode']: v for v in r['quality']}
        lines.append(f'| {r["case_id"]} | {q["target_assisted_t05_endpoint"]["ilm_opl_pcc"]:.4f} | '
                     f'{q["pure_oct_t0_endpoint"]["ilm_opl_pcc"]:.4f} | {q["pure_oct_euler8"]["ilm_opl_pcc"]:.4f} |')
    lines += ['', '三维网络同时接收固定 OCT c 和插值状态 x_t=(1−t)c+t*y。对于 t>0，'
        '代数上存在 y−c=(x_t−c)/t 的通道，而 t=0 时必须真正学会由 OCT 推断 OCTA。'
        '这说明一个可能的捷径；当前证据没有证明网络内部精确实现了该公式，也不证明流匹配理论本身无效。', '',
        '**损失组合也有需要排查的局部失衡和冲突。** 下表的梯度已乘实际损失权重，均只对生成主干计算。'
        '使用 t=0 探针；它反映起点处的局部行为，不代表所有训练时间的平均。', '',
        '| 病例 | FM梯度范数 | 重建梯度范数 | 投影梯度范数 | 对比梯度范数 | FM与投影梯度余弦 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        g = r['weighted_gradients'][0]
        norms = [g['terms'][k]['generator_gradient_norm'] for k in ('fm', 'reconstruction', 'projection', 'rdca')]
        lines.append(f'| {r["case_id"]} | '+' | '.join(f'{v:.4f}' for v in norms)+f' | {g["gradient_cosines"]["fm__projection"]:.4f} |')
    lines += ['', '负余弦表示该点上两项梯度方向相反；较大的对比梯度会影响共同更新。'
        '这不能单独证明某模块导致失败，因此还需要下面的真实训练删减对照。'
        '也不能用标量 loss 的大小直接代替梯度影响。', '',
        '预算短是限制之一，但不是唯一解释：此前相同三维主干的纯起点监督可记住单个训练块，'
        '普通随机时间 FM 延长到 7,200 次后仍未得到相同拟合效果。该证据来自单块实验，不能直接外推四病例泛化。'
        '本轮把所有方法统一延长到 2,400 次，并保留中间节点，检验短程排序是否持续。', '',
        '目标图在相同 Z256 网格仍有清晰血管；固定窗、轴序和保存目标的核验也通过。'
        '当前证据不支持仅把失败归因于显示窗、积分步数或完全没有网络容量。'
        '小体块上下文、有限病例数及目标本身的不确定性仍可能影响泛化，本轮没有逐一证明它们的贡献。']
    return lines


def main():
    config = read_json(ROOT/'protocol.json')
    output = ROOT/'reports'
    output.mkdir(exist_ok=True)
    results, rows = {}, []
    for name in config['arms']:
        for update in config['milestones']:
            path = ROOT/name/f'metrics_{update:05d}.json'
            if not path.exists():
                continue
            result = read_json(path)
            results[name, update] = result
            for split, summary in result['summary'].items():
                rows.append(dict(arm=name, updates=update, split=split, **summary['metrics']))
    if rows:
        keys = ['arm', 'updates', 'split']+sorted(set().union(*(r.keys() for r in rows))-{'arm', 'updates', 'split'})
        with (output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    write_json(output/'summary.json', rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.facecolor': 'white'})
    names = [n for n in config['arms'] if (n, config['updates']) in results]
    if names:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
        for ax, metric in zip(axes, ('ilm_opl_pcc', 'ilm_opl_ssim')):
            x = np.arange(len(names))
            for i, (split, color) in enumerate((('train', '#3676A4'), ('validation', '#C77B3D'))):
                values = [results[n, config['updates']]['summary'][split]['metrics'][metric] for n in names]
                ax.bar(x+(i-.5)*.36, values, .34, label=split, color=color)
            ax.set_xticks(x, [SHORT[n] for n in names], rotation=35, ha='right')
            ax.set_ylim(0, 1)
            ax.set_title(metric.replace('_', ' ').upper())
            ax.grid(axis='y', alpha=.2)
            ax.legend(frameon=False)
        fig.suptitle(f'Final equal budget: {config["updates"]} successful updates | cumulative 3D removal + method controls')
        fig.savefig(output/'final_comparison.png', dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for name in config['arms']:
        path = ROOT/name/'history.jsonl'
        if not path.exists():
            continue
        import json
        history = [json.loads(line) for line in path.read_text().splitlines()]
        for ax, split in zip(axes, ('train', 'validation')):
            ax.plot([r['update'] for r in history], [r['summary'][split]['metrics']['ilm_opl_ssim'] for r in history],
                marker='o', markersize=3, label=SHORT[name])
            ax.set_title(split+' ILM-OPL SSIM')
            ax.set_xlabel('Successful updates')
            ax.grid(alpha=.2)
            ax.set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=8, loc='best')
    fig.savefig(output/'budget_curves.png', dpi=160)
    plt.close(fig)
    core = ['00_flowoct_public', '01_full3d', '05_haar_fm_only', '06_voxel_fm_only', '07_full_half_start']
    for split in ('train', 'validation'):
        images(core, config, config['updates'], split, output)
    for update in config['milestones']:
        images(list(config['arms']), config, update, 'validation', output)
    done = len([1 for n in config['arms'] if (n, config['updates']) in results])
    lines = ['# 原版 FlowOCT 与三维模块累计删减：对照报告', '',
        f'日期：2026-10-22。计划 {len(config["arms"])} 组，每组 {config["updates"]:,} 次成功更新；当前完成 {done} 组。', '',
        '固定 4 位训练患者、4 位验证患者，每例两个全 Z 体块，输入与前轮逐元素一致。'
        '所有主表都是纯 OCT 起步、Euler8、仅最终裁剪后的患者宏平均指标。测试集图像不参与本实验。', '',
        '## 比较设置', '',
        '| 组 | 小波 | 对比 | 投影 | 结构加权 | 重建 |', '|---|---|---|---|---|---|']
    for name, arm in config['arms'].items():
        if arm['kind'] == 'public':
            lines.append('| 公开代码 FlowOCT | 二维 | 原切片InfoNCE | 原抽片max | 原ROI裁框 | L1 |')
        else:
            loss = arm['losses']
            flags = [arm['representation'], '有' if loss['rdca_weight'] else '无', '有' if loss['projection_weight'] else '无',
                     '有' if loss['structure'] else '无', '有' if loss['reconstruction_weight'] else '无']
            lines.append('| '+ZH[name]+' | '+' | '.join(flags)+' |')
    lines += ['', '三维组 01→06 每一步仅在前一步上移除一个模块。该路径说明特定剩余模块组合下的条件贡献，'
        '不能当成各模块独立主效应或排除交互。50% 起点组单列，完整模块保持一致，只改变训练时间采样。', '',
        '公开代码 FlowOCT 保留作者二维 MONAI 网络、二维 Haar、Adam(beta1=0.5)、'
        'FP32、无梯度裁剪及 10/0.5/0.005/0.1 权重。三维组保留原 AdamW、AMP、clip1 和 1/0.5/0.1/0.005 权重。'
        '因此原版与三维组是同数据、同更新数的方法对照；逐项因果解释只用于三维累计删减路径。', '',
        '公开代码每次抽 16 张 B-scan，每张单独采样时间；三维组使用完整 256×64×64 体块和每块一个时间。'
        '每次体素曝光和计算量不同，不能说是等 FLOPs 或仅改变网络维度。'
        '公开代码中切片对比/投影也不等同于论文描述的物理轴向 RDCA/分层投影。', '',
        '3D 第一段 800 次复用前轮的精确病例顺序和时间序列；后两段在训练前冻结。'
        '每个训练体块最终恰好曝光 300 次。全部组从头训练；主表取固定终点，不用验证集挑选对自己有利的预算。']
    for update in config['milestones']:
        lines += ['', f'## 同预算 {update:,} 次', '', '| 方法 | 训练PCC | 训练SSIM | 验证PCC | 验证SSIM | 验证体素PSNR |',
                  '|---|---:|---:|---:|---:|---:|']
        for name in config['arms']:
            if (name, update) not in results:
                lines.append(f'| {ZH[name]} | 待完成 | — | — | — | — |')
                continue
            a, b = [results[name, update]['summary'][s]['metrics'] for s in ('train', 'validation')]
            lines.append(f'| {ZH[name]} | {a["ilm_opl_pcc"]:.4f} | {a["ilm_opl_ssim"]:.4f} | '
                f'{b["ilm_opl_pcc"]:.4f} | {b["ilm_opl_ssim"]:.4f} | {b["volume_psnr"]:.3f} |')
    lines += ['', '![预算曲线](budget_curves.png)', '']
    if (output/'final_comparison.png').exists():
        lines += ['![最终同预算对比](final_comparison.png)', '']
    native_path = ROOT/'00_flowoct_public/final_native3.json'
    if native_path.exists():
        native = read_json(native_path)['summary']['validation']['metrics']
        shared = results['00_flowoct_public', config['updates']]['summary']['validation']['metrics']
        lines += ['## 原版求解器补充', '', f'同一份公开代码 FlowOCT 权重：原生三步/逐步裁剪的验证 PCC/SSIM 为 '
            f'{native["ilm_opl_pcc"]:.4f}/{native["ilm_opl_ssim"]:.4f}；统一八步/最终裁剪为 '
            f'{shared["ilm_opl_pcc"]:.4f}/{shared["ilm_opl_ssim"]:.4f}。'
            '两者仅作为求解设置对照，三步分数不参与主表的统一八步排名。', '']
    lines += diagnosis_text()
    lines += ['', '## 固定病例的图像对比', '',
        '每张图使用全部四位患者的固定第一体块，所有面板共用显示窗 [0,0.5]。不按模型分数挑病例。']
    for path in (output/f'train_{config["updates"]:05d}.png', output/f'validation_{config["updates"]:05d}.png'):
        if path.exists():
            lines += ['', f'![{path.stem}]({path.name})']
    for update in config['milestones']:
        path = output/f'validation_{update:05d}.png'
        if path.exists() and update != config['updates']:
            lines += ['', f'![{update}次全部累计组]({path.name})']
    audit_path = ROOT/'verification/results_audit.json'
    if audit_path.exists():
        audit = read_json(audit_path)
        lines += ['', f'独立复核：{audit["successful_updates"]:,} 次成功更新，{audit["predictions"]} 份预测；'
            f'主要指标最大重算差异 {audit["maximum_metric_error"]:.3g}。', '']
    lines += ['## 结论边界与文件', '',
        '这是固定小样本、单种子的机制对照。训练高分说明拟合，不等于新患者泛化。'
        '不能把任何一条累计路径、一个预算节点或一组局部梯度，单独视为最终方法排名或医学有效性证明。', '',
        '- `summary.csv` 保存全部同预算患者宏平均结果；各组 `metrics_*.json` 包含逐病例和逐患者数据。',
        '- 各组 `update_00800.pt`、`update_01600.pt`、`update_02400.pt` 保存模型、优化器和 AMP 状态。',
        '- `predictions_*/` 保存原始预测及目标、固定窗 B-scan 和分层投影；原版另存 `predictions_native3/`。',
        '- `../verification/diagnosis_prior800.json` 和 `diagnosis_final.json` 保存辅助终点、自由生成与梯度诊断。',
        '- `../verification/results_audit.json` 保存源码、输入、输出哈希、成功更新、优化器计数和指标独立重算。']
    (output/'实验报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    table = []
    for name in names:
        m = results[name, config['updates']]['summary']['validation']['metrics']
        table.append(f'<tr><td>{html.escape(ZH[name])}</td><td>{m["ilm_opl_pcc"]:.4f}</td><td>{m["ilm_opl_ssim"]:.4f}</td><td>{m["volume_psnr"]:.3f}</td></tr>')
    figures = [p for p in (output/'final_comparison.png', output/'budget_curves.png', output/f'train_{config["updates"]:05d}.png', output/f'validation_{config["updates"]:05d}.png') if p.exists()]
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>FlowOCT 累计删减实验</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 24px;line-height:1.6;color:#18252e}'
        'table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}'
        'td:first-child,th:first-child{text-align:left}img{max-width:100%}</style>'
        '<h1>原版 FlowOCT 与三维模块累计删减</h1>'
        f'<p>{done}/{len(config["arms"])} 组完成 · 每组 {config["updates"]:,} 次成功更新 · 4位训练/4位验证患者</p>'
        '<p><a href="实验报告.md">完整中文报告与原因诊断</a> · <a href="summary.csv">原始汇总表</a></p>'
        '<table><tr><th>方法</th><th>验证 PCC</th><th>验证 SSIM</th><th>验证 PSNR</th></tr>'+''.join(table)+'</table>'
        +''.join(f'<img src="{p.name}" alt="comparison">' for p in figures)+'</html>')
    (output/'index.html').write_text(page, encoding='utf-8')
    print('Report generated:', done, '/', len(config['arms']), flush=True)


if __name__ == '__main__':
    main()

