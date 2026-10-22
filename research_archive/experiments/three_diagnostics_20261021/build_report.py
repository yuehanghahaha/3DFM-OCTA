"""Build an auditable local report from the downloaded server results."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                     'figure.facecolor': '#f7f8fa', 'axes.facecolor': '#ffffff',
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
COLORS = {'voxel': '#2764ad', 'haar': '#c47920', 'supervised': '#21856f'}
NAMES = {'voxel': '3D 基线', 'haar': '3D + Haar 小波'}


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def macro(records, extractor):
    patients = defaultdict(list)
    for record in records:
        value = extractor(record)
        if value is not None:
            patients[record['patient_id']].append(value)
    return float(np.mean([np.mean(values) for values in patients.values()]))


def teacher(record, t):
    return next(row for row in record['teacher'] if row['time'] == t)


def solver(record, steps, method='euler'):
    return next(row for row in record['solvers'] if row['steps'] == steps and row['method'] == method)


def main():
    protocol = read('protocol.json')
    exp1 = read('experiment1_supervised/result.json')
    curve = read('experiment1_supervised/history.json')
    matched = read('experiment1_same_backbone/result.json')
    matched_curve = read('experiment1_same_backbone/history.json')
    exp2 = read('experiment2_velocity/result.json')
    exp3 = read('experiment3_formulas/result.json')
    audit = read('final_audit.json')
    layers = read('posthoc_layers/result.json')
    assert exp1['complete'] and matched['complete'] and exp2['complete'] and exp3['complete']
    assert exp3['all_passed'] and all(audit[key] for key in ('snapshot_hashes_unchanged', 'copied_production_source_hashes_unchanged', 'no_test_cases'))
    cases = exp2['cases']
    summary = dict(checkpoints=protocol['checkpoints'], supervised=exp1, capacity_matched_supervised=matched,
                   formula_checks=len(exp3['checks']), posthoc_layers=layers, groups={})
    csv_rows = []
    for arm in ('voxel', 'haar'):
        summary['groups'][arm] = {}
        for split in ('train', 'validation'):
            records = [r for r in cases if r['arm']==arm and r['split']==split]
            group = dict(patients=len({r['patient_id'] for r in records}), patches=len(records), teacher={}, solvers={}, noise_ood={})
            for t in protocol['times']:
                group['teacher'][str(t)] = {
                    key: macro(records, lambda r: teacher(r,t)[key])
                    for key in ('velocity_mse', 'zero_velocity_mse', 'relative_rmse', 'velocity_pcc', 'high_signal_velocity_mse')}
                group['teacher'][str(t)]['endpoint_pcc'] = macro(records, lambda r: teacher(r,t)['endpoint_clamp']['max_projection_pcc'])
            for steps in (1, 8, 32):
                group['solvers'][str(steps)] = {key: macro(records, lambda r: solver(r,steps)['final_clamp'][key])
                    for key in ('mae', 'psnr', 'max_projection_pcc', 'max_projection_ssim')}
            first = next(r for r in records if r['origin_ZYX'][1] == 64)
            group['first_patch_extra_solvers'] = [r for r in first['solvers'] if r['steps']==128 or r['method']=='heun']
            for t in (.1, .5, .9):
                group['noise_ood'][str(t)] = {key: macro(records, lambda r: np.mean([v[key] for v in r['noise_ood'] if v['time']==t]))
                                            for key in ('velocity_mse', 'relative_rmse', 'velocity_pcc')}
            group['t0_to_t05_velocity_mse_ratio'] = group['teacher']['0.0']['velocity_mse']/group['teacher']['0.5']['velocity_mse']
            summary['groups'][arm][split] = group
            for record in records:
                for row in record['teacher']:
                    csv_rows.append(dict(arm=arm, split=split, case_id=record['case_id'],
                        origin_y=record['origin_ZYX'][1], **{key: row[key] for key in ('time', 'velocity_mse', 'relative_rmse', 'velocity_pcc')},
                        endpoint_mae=row['endpoint_clamp']['mae'], endpoint_pcc=row['endpoint_clamp']['max_projection_pcc']))
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    with (ROOT / 'velocity_per_patch.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), layout='constrained')
    steps = [r['update'] for r in curve]
    axes[0,0].plot(steps, [r['raw']['max_projection_pcc'] for r in curve], color='#7a8591', lw=1.6, label='小 U-Net：PCC')
    axes[0,0].plot(steps, [r['raw']['max_projection_ssim'] for r in curve], color='#7a8591', ls='--', lw=1.3, label='小 U-Net：SSIM')
    axes[0,0].plot([r['update'] for r in matched_curve], [r['raw']['max_projection_pcc'] for r in matched_curve], color=COLORS['supervised'], lw=2.4, label='原主干纯监督：PCC')
    axes[0,0].plot([r['update'] for r in matched_curve], [r['raw']['max_projection_ssim'] for r in matched_curve], color=COLORS['supervised'], ls='--', lw=1.7, label='原主干纯监督：SSIM')
    first_id = exp1['evaluation'][0]['id']
    for arm in ('voxel','haar'):
        case = next(r for r in cases if r['arm']==arm and r['id']==first_id)
        axes[0,0].axhline(solver(case,8)['final_clamp']['max_projection_pcc'], color=COLORS[arm], ls=':', label=NAMES[arm]+'：自由生成 PCC')
    axes[0,0].set(title='实验 1 | 固定训练块的记忆曲线', xlabel='成功优化更新次数', ylabel='全深度最大投影 PCC / SSIM', ylim=(-.05,1.02))
    axes[0,0].legend(fontsize=8)
    times = protocol['times']
    for arm in ('voxel','haar'):
        for split, ls, suffix in (('train','-','训练'),('validation','--','验证')):
            group=summary['groups'][arm][split]
            axes[0,1].plot(range(len(times)), [group['teacher'][str(t)]['velocity_mse'] for t in times], ls=ls, marker='o', ms=3, color=COLORS[arm], label=NAMES[arm]+' / '+suffix)
    axes[0,1].set(title='实验 2 | 真实训练路径上的速度误差', xlabel='时间 t；t>0 的输入含真实 OCTA', ylabel='速度 MSE（对数刻度）', yscale='log')
    axes[0,1].set_xticks(range(len(times)),[f'{t:g}' for t in times])
    axes[0,1].legend(fontsize=8)
    for arm in ('voxel','haar'):
        group=summary['groups'][arm]['validation']
        axes[1,0].plot([1,8,32],[group['solvers'][str(n)]['max_projection_pcc'] for n in (1,8,32)], marker='o', color=COLORS[arm], label=NAMES[arm])
    axes[1,0].set(title='补充对照 | 增加积分步数能否恢复结构', xlabel='Euler 步数（4 个验证块）', ylabel='最终最大投影 PCC', ylim=(-.05,1.02))
    axes[1,0].set_xticks([1,8,32])
    axes[1,0].legend()
    axes[1,1].axis('off')
    text = (f"实验 3 | 原生产代码公式检查\n\n{len(exp3['checks'])} / {len(exp3['checks'])} 项通过\n\n"
            '训练：状态 (1−t)·OCT + t·OCTA\n目标速度：OCTA − OCT\n采样：从 OCT 出发，正向积分\n小波逆变换、时间网格、重叠窗：通过\n\n'
            '图中的噪声路径另做分布外测试；\n现有权重没有按该路径训练。')
    axes[1,1].text(.05,.94,text,va='top',fontsize=12,linespacing=1.65)
    fig.suptitle('3D OCT → OCTA：三项故障诊断', fontsize=18, fontweight='bold')
    fig.savefig(ROOT / 'diagnostic_overview.png', dpi=170)
    plt.close(fig)

    fig, axes=plt.subplots(2,6,figsize=(16.8,6.3),layout='constrained')
    ids = [exp1['evaluation'][0]['id'], next(r['id'] for r in exp1['evaluation'] if r['split']=='validation')]
    for row, identity in enumerate(ids):
        paths=[ROOT/'reference_maps'/(identity+'_oct.npz'), ROOT/'reference_maps'/(identity+'_octa.npz'),
               ROOT/'experiment2_velocity'/('voxel_'+identity+'_euler8.npz'), ROOT/'experiment2_velocity'/('haar_'+identity+'_euler8.npz'),
               ROOT/'experiment1_supervised'/(identity+'.npz'),ROOT/'experiment1_same_backbone'/(identity+'.npz')]
        names=['OCT 输入','真实 OCTA','3D 基线 / Euler8','3D + Haar / Euler8','小 U-Net / 纯监督','原主干 / 纯监督']
        for col,(path,name) in enumerate(zip(paths,names)):
            with np.load(path) as data:
                image=data['max_z']
            axes[row,col].imshow(image,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            axes[row,col].set_xticks([])
            axes[row,col].set_yticks([])
            label=name
            if col in (2,3):
                arm=('voxel','haar')[col-2]
                record=next(r for r in cases if r['arm']==arm and r['id']==identity)
                label+=f"\nPCC {solver(record,8)['final_clamp']['max_projection_pcc']:.3f}"
            if col==4:
                record=next(r for r in exp1['evaluation'] if r['id']==identity)
                label+=f"\nPCC {record['final_clamp']['max_projection_pcc']:.3f}"
            if col==5:
                record=next(r for r in matched['evaluation'] if r['id']==identity)
                label+=f"\nPCC {record['final_clamp']['max_projection_pcc']:.3f}"
            axes[row,col].set_title(label,fontsize=10)
        axes[row,0].set_ylabel(('已训练的单块\n' if row==0 else '未训练的验证块\n')+identity.split('_')[1],fontsize=11)
    fig.suptitle('同一数据的全 Z 最大投影；所有面板统一灰度 0–1，无单图拉伸',fontsize=14)
    fig.savefig(ROOT/'projection_comparison.png',dpi=180)
    plt.close(fig)

    fig, axes=plt.subplots(2,6,figsize=(16.8,6.3),layout='constrained')
    for row,identity in enumerate(ids):
        record=next(r for r in layers['cases'] if r['id']==identity)
        assert record['cached_input_exact_match'] and record['empty_rays']==0
        with np.load(ROOT/'posthoc_layers'/(identity+'.npz')) as arrays:
            for col,(name,label) in enumerate(zip(('oct','truth','voxel','haar','small','matched'),names)):
                axes[row,col].imshow(arrays[name],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                axes[row,col].set_xticks([])
                axes[row,col].set_yticks([])
                if name in record['metrics']:
                    label+=f"\nPCC {record['metrics'][name]['pcc']:.3f}"
                axes[row,col].set_title(label,fontsize=10)
        axes[row,0].set_ylabel(('已训练的单块\n' if row==0 else '未训练的验证块\n')+record['case_id'],fontsize=11)
    fig.suptitle('ILM–OPL 分层最大投影；真实层边界仅用于推理后评价；灰度统一 0–1',fontsize=14)
    fig.savefig(ROOT/'ilm_opl_comparison.png',dpi=180)
    plt.close(fig)

    final=curve[-1]['raw']
    initial=curve[0]['raw']
    trained=exp1['evaluation'][0]
    mse_ratio=final['mse']/exp1['constant_mean_predictor']['mse']
    table=[]
    for arm in ('voxel','haar'):
        group=summary['groups'][arm]['validation']
        values=[group['teacher'][str(t)]['velocity_mse'] for t in (0.,.1,.5,.9)]
        table.append('| '+NAMES[arm]+' | '+' | '.join(f'{v:.8f}' for v in values)+f" | {group['t0_to_t05_velocity_mse_ratio']:.1f} 倍 |")
    solver_table=[]
    for arm in ('voxel','haar'):
        group=summary['groups'][arm]['validation']
        solver_table.append('| '+NAMES[arm]+' | '+' | '.join(f"{group['solvers'][str(n)]['max_projection_pcc']:.4f}" for n in (1,8,32))+' |')
    same_patch_table=[]
    for arm in ('voxel','haar'):
        record=next(r for r in cases if r['arm']==arm and r['id']==first_id)
        metric=solver(record,8)['final_clamp']
        same_patch_table.append(f"| {NAMES[arm]}，8 步自由生成 | {metric['mae']:.6f} | {metric['max_projection_pcc']:.4f} | {metric['max_projection_ssim']:.4f} |")
    same_patch_table.append(f"| 监督 U-Net，2400 次更新 | {trained['final_clamp']['mae']:.6f} | {trained['final_clamp']['max_projection_pcc']:.4f} | {trained['final_clamp']['max_projection_ssim']:.4f} |")
    matched_trained=matched['evaluation'][0]
    same_patch_table.append(f"| 原主干纯监督，2400 次更新 | {matched_trained['final_clamp']['mae']:.6f} | {matched_trained['final_clamp']['max_projection_pcc']:.4f} | {matched_trained['final_clamp']['max_projection_ssim']:.4f} |")
    extras=[]
    for arm in ('voxel','haar'):
        group=summary['groups'][arm]['validation']
        record=next(r for r in cases if r['arm']==arm and r['id']==ids[1])
        for method,n in [('euler',8),('euler',32),('euler',128),('heun',8),('heun',32)]:
            metric=solver(record,n,method)['final_clamp']
            extras.append(f"| {NAMES[arm]} | {method} {n} | {metric['mae']:.6f} | {metric['max_projection_pcc']:.4f} |")
    noise_rows=[]
    for arm in ('voxel','haar'):
        group=summary['groups'][arm]['validation']
        noise_rows.append('| '+NAMES[arm]+' | '+' | '.join(f"{group['noise_ood'][str(t)]['relative_rmse']:.3f}" for t in (.1,.5,.9))+' |')
    interpretation = ('与基线相同的 3D 主干在纯监督下通过了记忆阈值，而两组 FM 的纯 OCT 自由生成仍很差。当前证据最支持：原 FM 的起点映射没有学好，训练中间状态的低误差没有转化为从 OCT 出发的结构恢复。不能把失败笼统归因为“三维网络完全学不会”。'
                      if matched['memorization_threshold_passed'] else
                      '同主干监督控制未同时达到预设全部阈值，需结合实际结构改善继续排查优化与拟合能力；目前不能把 FM 结果单独归因为训练路径，也不能仅凭未达阈值认定数据有错。')
    layer_train=next(r for r in layers['cases'] if r['id']==first_id)
    layer_val=next(r for r in layers['cases'] if r['id']==ids[1])
    layer_rows=[]
    for name,label in [('voxel','3D 基线 / Euler8'),('haar','3D + Haar / Euler8'),('small','小 U-Net / 纯 MSE'),('matched','原主干 / 纯 MSE')]:
        a,b=layer_train['metrics'][name],layer_val['metrics'][name]
        layer_rows.append(f"| {label} | {a['pcc']:.4f} | {a['ssim']:.4f} | {b['pcc']:.4f} | {b['ssim']:.4f} |")
    report=f'''# 3D 基线与 3D 小波：三项服务器诊断

运行在服务器物理 GPU0，原有 GPU1 训练持续运行。全部实验与最终复核完成时间：{audit['completed']}。含三项诊断、同主干监督控制与推理后分层评价。

## 判断

{interpretation}

两组 FM 权重都应根据“纯 OCT 起点的速度误差”与“含真值插值状态的速度误差”的差异判断训练是否覆盖了实际生成需求。下表和图像是本次冻结快照的实测结果。公式检查全部通过，暂未发现训练/采样的符号、时间方向或小波接口错误。增加积分步数的实际收益见下表，不能仅因中间状态误差低就把失败归咎于 ODE。

![诊断总览]({(ROOT/'diagnostic_overview.png').as_posix()})

## 冻结对象与数据

- 3D 基线：第 **{protocol['checkpoints']['voxel']['epoch']}** 轮，SHA256 `{protocol['checkpoints']['voxel']['sha256']}`。
- 3D 小波：第 **{protocol['checkpoints']['haar']['epoch']}** 轮，SHA256 `{protocol['checkpoints']['haar']['sha256']}`。
- 固定选训练病例 10301、10303，验证病例 10302、10319；每例取 `[Z,Y,X]=[256,64,64]`、起点 `(0,64,64)` 与 `(0,120,120)` 两块。共 4 位患者、8 块，无测试病例。
- 每位患者先平均两块，再按患者等权平均。样本少、单随机种子，用于机制诊断，不能作为方法排名或泛化性能证明。
- 投影指标均为**全深度 Z 最大投影**，不是 ILM–OPL 指标，也不是血管分割/拓扑精度。
- 现有两组实际训练起点是 OCT，**不是高斯噪声**。图中噪声路径的额外检查见实验 2 的分布外结果。

## 实验 1：纯监督单样本记忆

从头训练独立的小型 3D U-Net：base16、两次降采样，{exp1['parameter_count']:,} 参数；只输入 OCT，直接预测 OCTA；无时间条件、无 FM、无小波、无投影或分割损失。只训练 10301 的 `(0,64,64)` 固定块，纯全体素 MSE，2400 次成功更新，FP32，Adam，无 weight decay。前 1600 次学习率 2e-4，后 800 次 5e-5。没有增强或换块。

训练块原始输出：MAE **{initial['mae']:.6f} → {final['mae']:.6f}**，最大投影 PCC **{initial['max_projection_pcc']:.4f} → {final['max_projection_pcc']:.4f}**，SSIM **{initial['max_projection_ssim']:.4f} → {final['max_projection_ssim']:.4f}**。最终 MSE 是该目标均值常数预测器的 **{mse_ratio*100:.2f}%**。预先记录的记忆阈值为 PCC≥0.90、SSIM≥0.80、MSE≤常数预测器的10%；是否全部通过：**{exp1['memorization_threshold_passed']}**。

同一训练块的最终裁剪后比较：

| 模型 | 体积 MAE ↓ | 最大投影 PCC ↑ | 最大投影 SSIM ↑ |
|---|---:|---:|---:|
{chr(10).join(same_patch_table)}

这是能力正对照，模型与训练目标不同，不能把差值归因为唯一的因素。新监督模型未在额外块上训练；其所有保留块结果见 `experiment1_supervised/result.json`。训练块拟合好不等于新患者泛化好。

### 补充：把主干容量固定的直接监督控制

小网络未达到全部阈值后，为避免把新换的网络结构/容量引入归因，补训了**与原基线完全相同的 {matched['parameter_count']:,} 参数主干**。仍从相同种子随机初始化，只输入固定训练块的 OCT，固定 t=0，以 `prediction=OCT+network(OCT,0,OCT)` 做直接回归，纯 MSE；不构造 FM 插值状态，也不做 ODE。AdamW、lr2e-4、weight_decay0、clip1、AMP，对齐原主干训练设置。2400 次成功更新，AMP 跳过 {matched['skipped_amp_updates']} 次，没有据验证集选轮。

其训练块 PCC **{matched_trained['raw']['max_projection_pcc']:.4f}**、SSIM **{matched_trained['raw']['max_projection_ssim']:.4f}**，MSE 是目标均值常数预测器的 **{100*matched_trained['raw']['mse']/matched['constant_mean_predictor']['mse']:.2f}%**；是否通过同一阈值：**{matched['memorization_threshold_passed']}**。这是同一网络在直接监督下的拟合能力，**不是训练好的多步 FM 模型**。不能把其 t=0 回归权重直接拿去做多步采样，也不能把其训练分数当作验证泛化分数。

附加控制的动机、配置、完整曲线和所有保留块结果均单独记录在 `experiment1_same_backbone/`；没有替换或丢弃小网络的失败结果。它与 FM 的训练历史、训练样本覆盖和预算不同，是机制正对照，并非单因素方法性能消融。

![统一灰度投影]({(ROOT/'projection_comparison.png').as_posix()})

### 推理后补充：血管层的投影视图

全 Z 最大投影会混入其他层信号，因此另对预先选定的第一位训练/验证患者各两块，使用已有官方 ILM–OPL 层边界做推理后评价。先核对了其 OCT 和 OCTA 与既有冻结数据逐元素完全一致，4 块均通过；层边界从未输入生成模型，也未参与本次监督训练。没有按结果挑图或调整阈值。

以下是 10301 与 10302 的 `(0,64,64)` 两块；它们是两个具体样例，不是全部验证患者的平均：

| 模型 | 训练 ILM–OPL PCC | 训练 SSIM | 验证 ILM–OPL PCC | 验证 SSIM |
|---|---:|---:|---:|---:|
{chr(10).join(layer_rows)}

![ILM–OPL 分层投影]({(ROOT/'ilm_opl_comparison.png').as_posix()})

这组图也显示必须区分训练块的记忆和新患者的生成。高训练 PCC/SSIM 是正对照通过的依据，不是恢复了泛化的证明。完整4块指标及精确数据一致性检查见 `posthoc_layers/result.json`。

## 实验 2：速度场是否学好

先严格使用生产训练路径 `x_t=(1−t)c+tY`、`u=Y−c`，其中 `c=OCT`。小波模型在线性 Haar 域计算同一条路径，再逆变换评价图像。所有探针 FP32，无 ODE 采样参与速度误差。时间取 0、0.001、0.01、0.05、0.1、0.5、0.9、1。

验证患者宏平均的速度 MSE：

| 模型 | t=0 | t=0.1 | t=0.5 | t=0.9 | t=0 / t=0.5 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

`velocity_per_patch.csv` 保留每块数值；`summary.json` 同时给出相对于零速度预测的 RMSE、速度 PCC 和真实强度>0.1 区域的误差，避免仅看绝对 MSE。强度阈值不等于血管标注。

**所有 t>0 的真实路径输入已经含真实 OCTA；其终点图像不能当作有效生成。** 尤其 t 接近1时，输入本身已接近真值。起点 t=0 与纯 OCT 自由积分才检查实际生成能力。

当前路径加固定 OCT 条件，在 t>0 时满足 `Y−c=(x_t−c)/t`。这提供可利用的代数通道，说明低中间状态 loss 不足以保证学好了起点；本实验并不证明网络精确实现了该表达式，也不说明所有 FM 方法都存在同样失败。

### 积分步数与求解器对照

4 个验证块的自由生成最大投影 PCC：

| 模型 | Euler 1 | Euler 8 | Euler 32 |
|---|---:|---:|---:|
{chr(10).join(solver_table)}

另外固定第一位验证患者 10302、`(0,64,64)` 单块，检查更密步数与二阶 Heun：

| 模型 | 求解器与步数 | MAE ↓ | 最大投影 PCC ↑ |
|---|---|---:|---:|
{chr(10).join(extras)}

这部分是诊断采样敏感性，没有根据测试集选求解器。单块额外对照不代表4块的宏平均，也不代表整个体积的所有拼接设置。

### 图中随机噪声路径的补充检查

还使用三组固定随机种子的高斯噪声 ε，构造 `(1−t)ε+tY`，保持 OCT 条件不变，直接比较预测速度与 `Y−ε`。验证患者、每块先平均3种噪声后的相对 RMSE：

| 模型 | t=0.1 | t=0.5 | t=0.9 |
|---|---:|---:|---:|
{chr(10).join(noise_rows)}

**这些权重从未按高斯噪声起点训练，因此这是分布外压力测试，不能把其误差当作原训练的实现错误。** 若要比较噪声条件 FM 的可学习性，需要单独重训，不能直接更换现有模型的推理起点。

## 实验 3：训练与推理公式一致性

直接导入本次从服务器冻结的两份生产代码，共 **{len(exp3['checks'])} 项检查全部通过**，而不是只对新写的公式做自证：

- 用记录器检查生产 `flow_loss` 在 t=0/0.1/0.5/0.9/1 的输入状态、固定条件与速度目标；提供精确速度后损失为零。
- 用解析已知的有正有负速度测试生产 `generate_volume`。覆盖 1/8/32 步、多个重叠窗、奇数尺寸、Haar padding、逆变换、裁回原尺寸与训练模式恢复；终点最大误差小于1e-5。
- 检查采样时间网格为 `0,1/N,...,(N−1)/N`，起点为 OCT，按正向步长积分；反转速度的负对照确实失败。
- Haar 正逆变换与体素/系数域 MSE 的数值一致性通过；图中噪声路径本身的终点代数恒等式也通过。

另外核对了两份生产训练循环：实际调用均以 `sample['oct']` 为 source、`sample['octa']` 为 target；小波循环将配置中的表示方式传给 `flow_loss`。未发现调用处把源和目标交换或重复变换。

因此目前没有证据支持“训练走 OCT→OCTA，推理却反向走”“漏了步长”“小波子带接口错位”等解释。这个结论覆盖上述路径和测试范围，不等于证明所有数据、所有边界或学习到的速度场都正确。

## 下一步

优先在同一固定块上做独立干预：保持主干和数据，加入明确的纯 OCT 起点监督，记录 t=0 速度误差以及真正从 OCT 起步的自由生成投影曲线。只有自由生成也随之恢复结构，再扩展多病例、投影损失与完整训练。噪声起点应作为需要重训的独立分支；本次不据其分布外表现否定或支持这条分支。

## 复核与存储

服务器完整目录：`/docsystem/niuke/lvyuehang/three_diagnostics_20261021`。包括冻结权重、生产源码、输入体块、监督模型检查点、日志和完整指标。原生产目录没有被本次实验修改。

本地保留协议、哈希、指标、投影数组与图像；大体积输入和模型保留服务器，见 `SERVER_RUN.md`。主要数据：`protocol.json`、`input_audit.json`、`summary.json`、`velocity_per_patch.csv`、各实验的 `result.json` 和 `final_audit.json`。基线仍在另一路续训，之后的 best 可能正常更新；本报告始终指冻结的第 {protocol['checkpoints']['voxel']['epoch']} 轮。
'''
    (ROOT/'诊断报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(supervised_passed=exp1['memorization_threshold_passed'], matched_supervised_passed=matched['memorization_threshold_passed'], train_final=final,
                         groups=summary['groups']), ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
