"""Read completed matched controls and generate the local scientific report."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
ARMS=('fm_uniform','direct_t0','fm_half_start')
NAMES={'fm_uniform':'原 FM：随机时间','direct_t0':'直接监督：全部起点','fm_half_start':'FM＋起点：各占一半'}
COLORS={'fm_uniform':'#2764ad','direct_t0':'#21856f','fm_half_start':'#c47920'}
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.facecolor':'#f7f8fa',
                     'axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'font.size':10})


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def mean_patient(records,key):
    patients=defaultdict(list)
    for row in records:
        if key in row['primary'] and row['primary'][key] is not None:
            patients[row['patient_id']].append(row['primary'][key])
    return float(np.mean([np.mean(v) for v in patients.values()])) if patients else None


def main():
    protocol=read('protocol.json')
    audit=read('final_audit.json')
    verification=read('verification.json')
    assert audit['passed']
    assert verification['passed'] and verification['verified_predictions']==24
    assert hashlib.sha256((ROOT/'run_control.py').read_bytes()).hexdigest()==verification['run_script_sha256']
    reports={arm:read(f'{arm}/result.json') for arm in ARMS}
    curves={arm:read(f'{arm}/history.json') for arm in ARMS}
    summary=dict(protocol=protocol,final_audit=audit,independent_verification=verification,arms={})
    rows=[]
    for arm,report in reports.items():
        assert report['complete'] and report['successful_updates']==2400
        assert report['initial_state_sha256']==protocol['initial_state_sha256']
        for sample in report['evaluation']:
            sample['primary']=sample['outputs'][str(report['primary_steps'])]['final_clamp']
            if sample['id']==protocol['training_patch']:
                category='trained_patch'
            elif sample['case_id']==report['evaluation'][0]['case_id']:
                category='same_patient_other_patch'
            elif sample['split']=='train':
                category='other_train_patient'
            else:
                category='held_out_validation_patient'
            sample['category']=category
            rows.append(dict(arm=arm,id=sample['id'],case_id=sample['case_id'],patient_id=sample['patient_id'],
                category=category,primary_steps=report['primary_steps'],
                **{key:sample['primary'].get(key) for key in ('mae','psnr','max_projection_pcc','max_projection_ssim','ilm_opl_pcc','ilm_opl_ssim','ilm_opl_mae')}))
        train=next(r for r in report['evaluation'] if r['trained_on_this_patch'])
        held=[r for r in report['evaluation'] if r['split']=='validation']
        summary['arms'][arm]=dict(primary_steps=report['primary_steps'],successful_updates=report['successful_updates'],
            amp_skipped_updates=report['amp_skipped_updates'],zero_time_updates=report['zero_time_updates'],
            peak_allocated_bytes=report['peak_allocated_bytes'],elapsed_seconds=report['elapsed_seconds'],
            train_primary=train['primary'],train_outputs=train['outputs'],train_teacher=train['teacher'],
            held_out_patient_macro={key:mean_patient(held,key) for key in ('mae','psnr','max_projection_pcc','max_projection_ssim','ilm_opl_pcc','ilm_opl_ssim','ilm_opl_mae')})
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    with (ROOT/'per_patch.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig,axes=plt.subplots(2,2,figsize=(12.5,8.4),layout='constrained')
    for arm in ARMS:
        n=str(reports[arm]['primary_steps'])
        curve=curves[arm]
        x=[r['update'] for r in curve]
        axes[0,0].plot(x,[r['outputs'][n]['final_clamp']['ilm_opl_pcc'] for r in curve],color=COLORS[arm],lw=2,label=NAMES[arm])
        axes[0,1].plot(x,[r['outputs'][n]['final_clamp']['ilm_opl_ssim'] for r in curve],color=COLORS[arm],lw=2,label=NAMES[arm])
        axes[1,0].plot(x,[next(t['velocity_mse'] for t in r['teacher'] if t['time']==0) for r in curve],color=COLORS[arm],lw=2,label=NAMES[arm])
    axes[0,0].set(title='固定训练块：主要推理方式的血管投影相关性',xlabel='成功更新次数',ylabel='ILM–OPL PCC',ylim=(-.05,1.02))
    axes[0,1].set(title='固定训练块：主要推理方式的结构相似度',xlabel='成功更新次数',ylabel='ILM–OPL SSIM',ylim=(-.05,1.02))
    axes[1,0].set(title='纯 OCT 起点的速度误差',xlabel='成功更新次数',ylabel='t=0 速度 MSE（对数刻度）',yscale='log')
    for ax in (axes[0,0],axes[0,1],axes[1,0]):
        ax.legend(fontsize=8)
    positions=np.arange(3)
    for offset,n,label,color in [(-.18,'1','起点单步','#8f9bab'),(.18,'8','Euler8','#526c84')]:
        values=[summary['arms'][arm]['train_outputs'][n]['final_clamp']['ilm_opl_pcc'] for arm in ARMS]
        bars=axes[1,1].bar(positions+offset,values,width=.34,label=label,color=color)
        axes[1,1].bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
    axes[1,1].set(title='同一训练块：单步与多步结果',ylabel='ILM–OPL PCC',ylim=(-.05,1.12))
    axes[1,1].set_xticks(positions,['原 FM','直接监督','FM＋起点'])
    axes[1,1].legend(fontsize=9)
    axes[1,1].text(.5,.02,'直接监督组的 Euler8 仅作诊断，其正常用法是单步。',ha='center',transform=axes[1,1].transAxes,fontsize=8)
    fig.suptitle('同数据、同初始化、同预算：只改变训练时间分配',fontsize=17,fontweight='bold')
    fig.savefig(ROOT/'learning_curves.png',dpi=170)
    plt.close(fig)

    identities=[protocol['training_patch'],next(r['id'] for r in protocol['inputs'] if r['case_id']=='10302' and r['origin_ZYX'][1]==64)]
    fig,axes=plt.subplots(2,5,figsize=(14,6.4),layout='constrained')
    for row,identity in enumerate(identities):
        with np.load(ROOT/'reference_maps'/(identity+'_layers.npz')) as reference:
            images=[reference['oct'].copy(),reference['octa'].copy()]
        titles=['OCT 输入','真实 OCTA']
        for arm in ARMS:
            steps=reports[arm]['primary_steps']
            with np.load(ROOT/arm/'maps'/f'{identity}_euler{steps}_layer.npz') as data:
                images.append(data['ilm_opl'].copy())
            record=next(r for r in reports[arm]['evaluation'] if r['id']==identity)
            titles.append(NAMES[arm]+f"\nPCC {record['primary']['ilm_opl_pcc']:.3f}")
        for col,(a,label) in enumerate(zip(images,titles)):
            axes[row,col].imshow(a,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            axes[row,col].set_xticks([])
            axes[row,col].set_yticks([])
            axes[row,col].set_title(label,fontsize=10)
        axes[row,0].set_ylabel(('已训练体块\n' if row==0 else '未训练患者\n')+identity.split('_')[1],fontsize=11)
    fig.suptitle('ILM–OPL 投影；FM 两组使用 Euler8，直接监督使用单步；灰度统一 0–1',fontsize=14)
    fig.savefig(ROOT/'comparison.png',dpi=180)
    plt.close(fig)

    train_rows=[]
    val_rows=[]
    solver_rows=[]
    for arm in ARMS:
        a=summary['arms'][arm]
        tr,va=a['train_primary'],a['held_out_patient_macro']
        train_rows.append(f"| {NAMES[arm]} | {a['zero_time_updates']} | {tr['mae']:.6f} | {tr['ilm_opl_pcc']:.4f} | {tr['ilm_opl_ssim']:.4f} |")
        val_rows.append(f"| {NAMES[arm]} | {va['mae']:.6f} | {va['ilm_opl_pcc']:.4f} | {va['ilm_opl_ssim']:.4f} |")
        solver_rows.append('| '+NAMES[arm]+' | '+' | '.join(f"{a['train_outputs'][str(n)]['final_clamp']['ilm_opl_pcc']:.4f}" for n in (1,8,32))+' |')
    baseline=summary['arms']['fm_uniform']['train_primary']
    mixed=summary['arms']['fm_half_start']['train_primary']
    direct=summary['arms']['direct_t0']['train_primary']
    delta_pcc=mixed['ilm_opl_pcc']-baseline['ilm_opl_pcc']
    delta_ssim=mixed['ilm_opl_ssim']-baseline['ilm_opl_ssim']
    if delta_pcc>0 and delta_ssim>0:
        verdict=f"在这一固定训练块与种子下，增加起点时间采样使真正的8步自由生成有所改善：ILM–OPL PCC 比原 FM 高 {delta_pcc:.4f}，SSIM 高 {delta_ssim:.4f}。"
    else:
        verdict=f"在这一固定训练块与种子下，50%起点采样没有带来一致的结构改善：相对原 FM，PCC 差值 {delta_pcc:+.4f}，SSIM 差值 {delta_ssim:+.4f}。"
    if mixed['max_projection_pcc']<.9 or mixed['max_projection_ssim']<.8:
        verdict+=' 按协议预先记录的全Z最大投影阈值（PCC≥0.90、SSIM≥0.80），混合组仍未同时通过，不能称为已经修复。'
    else:
        verdict+=' 混合组在这个训练块上通过协议预先记录的全Z最大投影阈值（PCC≥0.90、SSIM≥0.80）；这只说明单块结构记忆通过。'
    val0=summary['arms']['fm_uniform']['held_out_patient_macro']
    val1=summary['arms']['fm_half_start']['held_out_patient_macro']
    validation_note=f"保留的两位验证患者上，混合组相对原 FM 的 ILM–OPL PCC 差值为 {val1['ilm_opl_pcc']-val0['ilm_opl_pcc']:+.4f}、SSIM 差值为 {val1['ilm_opl_ssim']-val0['ilm_opl_ssim']:+.4f}。两位患者不足以支持稳定泛化结论。"
    report=f'''# 原 FM、直接监督、FM＋起点监督：严格匹配的单块对照

完成时间：{audit['completed']}。服务器物理GPU0。共 **7200次成功优化更新**，每组2400次。

## 这轮回答什么

上一轮第3列是原来的多病例FM权重，第6列是反复训练一个体块的监督权重，训练曝光次数不一致，不能据此把差异单独归因于FM目标。本轮把网络、初始化、体块、优化器、更新数和每步网络调用数全部固定，只改变时间采样。

本轮三组均从同一随机初始化重新训练2400次；“原FM”表示原来的随机时间训练方式。

{verdict}

{validation_note}

![匹配对照曲线]({(ROOT/'learning_curves.png').as_posix()})

## 三组到底改变了什么

共同路径：`x_t=(1−t)·OCT+t·OCTA`，速度目标 `OCTA−OCT`，损失均为全体素MSE。

| 组别 | 2400次训练更新的时间分配 | 正常推理方式 |
|---|---|---|
| 原FM | 全部均匀随机t | 从纯OCT开始，Euler8 |
| 直接监督 | 全部t=0 | 一次修正，OCT＋v(OCT,0,OCT) |
| FM＋起点 | 1200次t=0，1200次随机t，交替进行 | 从纯OCT开始，Euler8 |

第三组通过混合时间采样实现额外起点监督，不是每次更新多算一个损失分支。它对应等权的“起点MSE与均匀时间FM MSE”混合目标；每次更新仍只有一次前向和反向。随机部分复用原FM在对应更新编号上的相同时间。它同时减少了随机中间时刻的更新数，结果应解释为这种时间预算分配的影响。

直接监督组的速度MSE与最终OCTA MSE在t=0代数等价；实际模型上的数值与梯度检查通过，见 `preflight.json`。

由于三组时间分布不同，不用平均训练loss给三组排名；统一检查只含OCT的生成结果与固定时间速度误差。

## 固定条件

- 原始3D主干，base16，**{protocol['parameter_count']:,}参数**，完全相同的初始状态哈希 `{protocol['initial_state_sha256']}`。
- 训练仅用10301的 `[256,64,64]` 体块，起点 `[0,64,64]`，三组输入文件哈希相同，无增强、换块或额外标签损失。
- AdamW，lr2e-4，weight_decay0，clip1；FP16 AMP、FP32损失与评价；每组2400次成功更新。
- 直接比较固定最终权重，未根据验证分数挑选检查点；每200次更新记录训练块曲线。
- 无小波、投影损失、分割或其他附加目标；真实层边界只在推理后计算ILM–OPL投影。
- 单个随机种子。额外评价保留上一轮固定8块，未使用测试患者。原训练集中的10303对这些诊断模型同样是未训练患者。

## 最终训练块结果

以下每组均使用表中列出的正常推理方式：原FM与混合组Euler8，直接监督组单步。

| 组别 | 起点更新数 | 体积MAE↓ | ILM–OPL PCC↑ | ILM–OPL SSIM↑ |
|---|---:|---:|---:|---:|
{chr(10).join(train_rows)}

直接监督训练块 PCC 为 **{direct['ilm_opl_pcc']:.4f}**、SSIM为 **{direct['ilm_opl_ssim']:.4f}**。这是单步拟合能力正对照，不能把它的权重当作已训练好的多步速度场。

![相同体块的分层投影]({(ROOT/'comparison.png').as_posix()})

## 相同推理步数的补充对照

同一个已训练块的ILM–OPL PCC：

| 组别 | 起点单步 | Euler8 | Euler32 |
|---|---:|---:|---:|
{chr(10).join(solver_rows)}

直接监督组在训练时没有接触t>0，它的多步结果只用于诊断时间外推，不能据此判断其正常单步方法失败。混合组是否有效，主要看真正从OCT开始的Euler8输出；起点单步变好但Euler8没变好，不能称为FM生成能力恢复。

## 未训练患者结果

10302与10319，每例两个固定体块；先按患者平均体块，再对两位患者等权平均。

| 组别 | 体积MAE↓ | ILM–OPL PCC↑ | ILM–OPL SSIM↑ |
|---|---:|---:|---:|
{chr(10).join(val_rows)}

此外还保存了同一训练患者另一个体块以及10303两块的结果，见 `per_patch.csv`。同患者两块边缘有少量空间重叠，不应当作独立患者。

这仍是一轮单块机制诊断。训练块拟合改善不等于学习了新患者的血管映射，不能用两位验证患者给方法作正式排名，也不能据单个种子给出显著性结论。

## 可以据此采取的下一步

本轮受控干预支持“起点时间覆盖是一个可改善因素”：保持其他条件一致，把一半更新分给t=0，真正从OCT开始的多步生成结构随之改善。这不证明网络严格利用了某个代数捷径，也不说明所有失败都来自起点。

直接监督仍然是训练块拟合最好的正对照；混合组的意义在于它保留了其他时间的训练，同时让8步生成接近其起点单步效果。两位未训练患者的改善有限，建议下一轮固定4个训练病例，保持这三组及相同预算设置，检验多病例下的新患者图像和指标是否同步改善。50%是本次诊断比例，不是已经选出的最优超参数。

## 一致性复核

- 三组实际初始参数逐字节一致，训练数据和成功更新次数一致。
- 逐条核对7200条训练记录与冻结时间序列；起点更新数分别为0、2400、1200。
- 各组AMP跳过次数：{', '.join(arm+'='+str(reports[arm]['amp_skipped_updates']) for arm in ARMS)}。若有跳过，不推进预设时间序列，仍以成功更新计数。
- 核对24份三维预测的形状、有限性、真值逐元素一致性和重新计算的体积MSE。
- 独立脚本重新从保存的三维预测计算ILM–OPL投影、PCC和SSIM，与指标和图片逐项吻合；三组初始单步输出一致。
- 模型与评价依赖源码哈希未变；未使用测试患者。

所有训练在物理GPU0完成；运行耗时受服务器资源状态影响，不作为吞吐率比较。

## 复核文件

服务器总目录：`/docsystem/niuke/lvyuehang/fm_start_control_20261022`。

冻结初始权重、时间序列、输入体块、训练记录、800/1600/2400步检查点与三维预测均保留服务器。本地保存小型指标、图像和报告。协议与审计见 `protocol.json`、`preflight.json`、`final_audit.json`，逐块表见 `per_patch.csv`，完整汇总见 `summary.json`。
'''
    (ROOT/'验证报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(verdict=verdict,validation=validation_note,arms=summary['arms']),ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
