"""Create a Chinese experiment report and fixed-window scientific figures."""
import argparse
import csv
import html
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .utils import read_json, write_json


LABELS = {
    '01_voxel_fm': '3D FM', '02_haar_fm': '3D Haar FM', '03_full': 'Full model',
    '04_no_rdca': 'No RDCA', '05_no_projection': 'No layer projection',
    '06_no_structure': 'No structure weighting', '07_no_reconstruction': 'No voxel L1',
    '08_full_half_start': 'Full + 50% t=0',
}
ZH = {
    '01_voxel_fm': '体素 3D FM', '02_haar_fm': '3D 小波 FM', '03_full': '完整模型',
    '04_no_rdca': '去除 RDCA', '05_no_projection': '去除分层投影',
    '06_no_structure': '去除结构加权', '07_no_reconstruction': '去除体素重建',
    '08_full_half_start': '完整模型＋50% 起点',
}


def report(root):
    root = Path(root)
    config = read_json(root/'suite.json')
    records, finals = [], {}
    for stage in config['stages']:
        for name in stage['arms']:
            path = root/name/stage['name']/'final_metrics.json'
            if not path.exists():
                raise FileNotFoundError(f'Experiment is not complete: {path}')
            final = read_json(path)
            finals[stage['name'], name] = final
            for split, values in final['summary'].items():
                records.append(dict(stage=stage['name'], arm=name, split=split, updates=final['updates'],
                                    patients=values['patients'], patches=values['patches'], **values['metrics']))
    output = root/'reports'
    output.mkdir(exist_ok=True)
    total_updates = sum(v['updates'] for v in finals.values())
    few_budget = config['stages'][1]['updates']
    keys = sorted(set().union(*(r.keys() for r in records)))
    with (output/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    write_json(output/'summary.json', records)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    names = config['stages'][1]['arms']
    x = np.arange(len(names))
    for axis, key, title in zip(axs, ['ilm_opl_pcc', 'ilm_opl_ssim'], ['ILM-OPL projection PCC', 'ILM-OPL projection SSIM']):
        for i, (split, color) in enumerate([('train', '#3676A4'), ('validation', '#C77B3D')]):
            values = [finals['few_case', n]['summary'][split]['metrics'][key] for n in names]
            axis.bar(x+(i-.5)*.37, values, width=.35, color=color, label=split)
        axis.set_xticks(x, [LABELS[n] for n in names], rotation=35, ha='right')
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.legend(frameon=False)
        axis.grid(axis='y', alpha=.2)
    fig.suptitle(f'Four training + four validation patients | {few_budget} successful updates per arm | final checkpoints')
    fig.savefig(output/'comparison.png', dpi=170)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name in ('01_voxel_fm', '02_haar_fm', '03_full', '08_full_half_start'):
        path = root/name/'few_case/history.jsonl'
        history = [__import__('json').loads(line) for line in path.read_text().splitlines()]
        for ax, split in zip(axs, ('train', 'validation')):
            ax.plot([r['update'] for r in history],
                    [r['summary'][split]['metrics']['ilm_opl_ssim'] for r in history],
                    marker='o', label=LABELS[name])
            ax.set_title(split+' ILM-OPL SSIM')
            ax.set_xlabel('Successful updates')
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            ax.legend(frameon=False)
    fig.savefig(output/'training_curves.png', dpi=170)
    plt.close(fig)
    selection = read_json(root/'data/selection.json')
    # Show all four held-out patients, not a case chosen by model scores.
    samples = [r for r in selection['patches'] if r['split'] == 'validation' and r['patch_id'] == 'y064_x064']
    columns = ('02_haar_fm', '03_full', '08_full_half_start')
    fig, axs = plt.subplots(len(samples), 4, figsize=(10, 10), constrained_layout=True)
    for i, sample in enumerate(samples):
        with np.load(root/sample['file'], allow_pickle=False) as data:
            target, layers = data['octa'], data['layers']
        centers = ((np.arange(len(target))+.5)*640/len(target)-.5)[:, None, None]
        mask = (centers >= layers[0]-1) & (centers <= layers[2]-1)
        arrays = [np.where(mask, target, 0).max(0)]
        for arm in columns:
            with np.load(root/arm/'few_case/predictions'/(sample['case_id']+'_'+sample['patch_id']+'.npz')) as data:
                arrays.append(np.where(mask, data['octa'], 0).max(0))
        for j, a in enumerate(arrays):
            axs[i, j].imshow(a, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if i == 0:
                axs[i, j].set_title(['Target', *[LABELS[n] for n in columns]][j])
            if j == 0:
                axs[i, j].set_ylabel(sample['case_id']+' '+sample['disease'])
    fig.suptitle('Held-out patients, fixed first crop | ILM-OPL max | identical display window [0,1]')
    fig.savefig(output/'heldout_predictions.png', dpi=170)
    plt.close(fig)
    # Keep a second common narrow window for details; never independently stretch each panel.
    for ax in axs.flat:
        ax.images[0].set_clim(0, .5)
    fig.suptitle('Held-out patients, fixed first crop | ILM-OPL max | identical display window [0,0.5]')
    fig.savefig(output/'heldout_predictions_detail.png', dpi=170)
    plt.close(fig)
    lines = ['# 3D 小波 FlowOCT 完整模块：小样本验证报告', '',
             f'日期：2026-10-22。共 {len(finals)} 个独立运行、{total_updates:,} 次成功优化更新，使用同一已冻结数据协议。', '',
             '这是短程可行性和模块对照实验；单随机种子、训练/验证各四位患者，每例两个固定全 Z 体块。'
             '单病例阶段为 400 次更新；四病例阶段为 800 次更新。未读取测试病例图像，也未进行全数据 200 轮训练。', '',
             '**所有表格使用仅从 OCT 自由积分的最终同预算检查点。路径中含真值的训练辅助终点没有用于生成指标。**', '',
             '## 四病例实验', '',
             '| 方法 | 训练 PCC | 训练 SSIM | 验证 PCC | 验证 SSIM | 验证体素 PSNR |',
             '|---|---:|---:|---:|---:|---:|']
    for name in names:
        a, b = [finals['few_case', name]['summary'][s]['metrics'] for s in ('train', 'validation')]
        lines.append(f'| {ZH[name]} | {a["ilm_opl_pcc"]:.4f} | {a["ilm_opl_ssim"]:.4f} | '
                     f'{b["ilm_opl_pcc"]:.4f} | {b["ilm_opl_ssim"]:.4f} | {b["volume_psnr"]:.3f} |')
    lines += ['', 'PCC 和 SSIM 均为 ILM–OPL 最大投影指标；先在患者内平均两个体块，再对患者等权平均。', '',
              '![同预算比较](comparison.png)', '', '![训练曲线](training_curves.png)', '',
              '## 完整模块的增量与边界', '']
    a, b = [finals['few_case', n]['summary']['validation']['metrics'] for n in ('02_haar_fm', '03_full')]
    lines.append(f'完整模型相对小波 FM：验证 PCC 变化 **{b["ilm_opl_pcc"]-a["ilm_opl_pcc"]:+.4f}**，'
                 f'SSIM 变化 **{b["ilm_opl_ssim"]-a["ilm_opl_ssim"]:+.4f}**。')
    no_structure = finals['few_case', '06_no_structure']['summary']['validation']['metrics']
    start = finals['few_case', '08_full_half_start']['summary']['validation']['metrics']
    lines += ['', '**本次结果不支持“把模块全部叠加后就优于小波基线”。** 完整模型的 SSIM 略有变化，PCC 和体素 PSNR 反而下降。'
              '统一显示窗下，预测仍偏模糊，未重建出真值中的清晰血管网络。', '',
              f'去除结构加权的验证 PCC 为 {no_structure["ilm_opl_pcc"]:.4f}，完整模型为 {b["ilm_opl_pcc"]:.4f}；'
              '这提示当前阈值、背景权重和频域权重组合需要进一步受控验证，不能直接解释为论文中的结构聚焦模块普遍无效。', '',
              f'50% 起点组的验证 SSIM 为 {start["ilm_opl_ssim"]:.4f}、PCC 为 {start["ilm_opl_pcc"]:.4f}、'
              f'体素 PSNR 为 {start["volume_psnr"]:.3f} dB。相对完整均匀时间组有所改善，'
              '但视觉细节仍不足，不能宣称起点问题已经解决。']
    lines += ['', '去模块组只改变对应损失开关；同一表示的初始化、病例顺序、随机时间、优化器和更新预算均一致。'
              '完整模型与去投影组使用的额外训练标注不同：分层投影需要训练病例层标注。'
              '体素与小波组在网络输入通道及有效空间网格上不同，不能宣称参数逐个一致。', '',
              '完整模型的标准组仍使用均匀时间。50% 起点组属于前期起点问题诊断的扩展，单列，不能当作 FlowOCT 原有贡献。', '',
              '## 单病例拟合', '', '| 方法 | 训练 PCC | 训练 SSIM | 验证 PCC | 验证 SSIM |', '|---|---:|---:|---:|---:|']
    for name in config['stages'][0]['arms']:
        a, b = [finals['one_case', name]['summary'][s]['metrics'] for s in ('train', 'validation')]
        lines.append(f'| {ZH[name]} | {a["ilm_opl_pcc"]:.4f} | {a["ilm_opl_ssim"]:.4f} | {b["ilm_opl_pcc"]:.4f} | {b["ilm_opl_ssim"]:.4f} |')
    lines += ['', '## 起点检查', '',
              '下表直接检查同一训练数据上的速度 MSE。t=0.5 的状态含真值，只用于机制诊断，不是可部署输入。', '',
              '| 方法 | 训练 t=0 MSE | 训练 t=0.5 MSE | 起点/中间误差比 |', '|---|---:|---:|---:|']
    for name in ('01_voxel_fm', '02_haar_fm', '03_full', '08_full_half_start'):
        m = finals['few_case', name]['summary']['train']['metrics']
        t0, t05 = m['t0_velocity_mse'], m['target_assisted_t05_velocity_mse']
        lines.append(f'| {ZH[name]} | {t0:.8f} | {t05:.8f} | {t0/max(t05,1e-12):.2f} |')
    lines += ['', '## 固定病例与固定显示窗', '', '下图涵盖全部四位验证患者的第一体块，不按模型分数挑样例。', '',
              '![验证病例](heldout_predictions.png)', '',
              '![统一较窄显示窗](heldout_predictions_detail.png)', '',
              '逐例 B-scan、均值/分层投影、误差和原始预测位于每组 `predictions/`。'
              '显示窗只影响图片，不改变指标。预测保留 Z256、64×64 体块网格，不声称恢复原始 Z640。', '',
              '## 可复核性与文件', '',
              '- `../verification/results_audit.json`：独立重算指标、输入/权重/输出 SHA256、更新数和对比头变化。',
              '- `summary.csv`：全部逐组汇总；各运行 `final_metrics.json` 包含逐病例和逐患者数据。',
              '- `../data/selection.json`：冻结病例、坐标、原输入指纹与标注用途。',
              '- `../source/`：实际训练源码快照；`../run.py --resume` 使用该快照恢复。',
              '- `latest.pt` 是固定终点主比较；`best_validation.pt` 仅根据验证集选择。', '',
              '当前样本量和预算不能给出稳定的方法排名、统计显著性、全数据泛化或临床有效性结论。'
              '完整模块的实现选择（背景权重、频域 mask、层范围及辅助终点）见工程 README。']
    audit_path = root/'verification/results_audit.json'
    if audit_path.exists():
        audit = read_json(audit_path)
        if audit.get('passed') and audit['successful_updates'] == total_updates and audit['runs'] == len(finals):
            lines += ['', f'独立复核通过：{audit["successful_updates"]:,} 次成功更新，{audit["predictions"]} 份体块预测，'
                      '重算主要指标与保存值一致。']
    else:
        lines += ['', '本目录尚无独立复核记录，请运行 `python -m flowoct3d.verify --root 实验目录`。']
    test_path = root/'verification/unit_tests.txt'
    if test_path.exists() and 'Ran 12 tests' in test_path.read_text(encoding='utf-8') and 'OK' in test_path.read_text(encoding='utf-8'):
        lines += ['', '12 项核心及端到端测试通过，记录见 `../verification/unit_tests.txt`。']
    if (root/'verification/full_volume_smoke.json').exists():
        lines += ['', '另外完成 10319 的完整 256×304×304 验证体推理，见 `../verification/full_volume_smoke.json`。'
                  '这是一例额外的整卷流程验证，不并入体块对照表。']
    (output/'实验报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    rows = ''.join('<tr><td>'+html.escape(ZH[n])+'</td>'+''.join(f'<td>{finals["few_case",n]["summary"][s]["metrics"][k]:.4f}</td>'
                   for s in ('train','validation') for k in ('ilm_opl_pcc','ilm_opl_ssim'))+'</tr>' for n in names)
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>3D FlowOCT 实验报告</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;line-height:1.6;color:#18252e}'
            'table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}'
            'td:first-child,th:first-child{text-align:left}img{max-width:100%}</style>'
            f'<h1>3D 小波 FlowOCT：完整模块验证</h1><p>{len(finals)} 个运行 · {total_updates:,} 次成功更新 · 四位训练/四位验证患者 · 固定预算</p>'
            '<p>比较只使用从 OCT 自由积分的预测。短程单种子结果不能替代完整测试。</p>'
            '<table><tr><th>方法</th><th>训练 PCC</th><th>训练 SSIM</th><th>验证 PCC</th><th>验证 SSIM</th></tr>'+rows+'</table>'
            '<p><a href="实验报告.md">完整中文报告</a> · <a href="summary.csv">原始汇总表</a></p>'
            +''.join(f'<img src="{name}" alt="experiment results">' for name in ('comparison.png','training_curves.png','heldout_predictions.png','heldout_predictions_detail.png'))+'</html>')
    (output/'index.html').write_text(page, encoding='utf-8')
    print(output/'实验报告.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    report(parser.parse_args().root)

