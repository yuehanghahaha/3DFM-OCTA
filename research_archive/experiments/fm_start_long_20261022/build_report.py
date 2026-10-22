"""Report equal-budget continuations and preserve the original2400 comparison."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'fm_start_control_20261022'
ARMS=('fm_uniform','direct_t0','fm_half_start')
NAMES={'fm_uniform':'原 FM：随机时间','direct_t0':'直接监督：全部起点','fm_half_start':'FM＋起点：各占一半'}
COLORS={'fm_uniform':'#2764ad','direct_t0':'#21856f','fm_half_start':'#c47920'}
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.facecolor':'#f7f8fa',
    'axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'font.size':10})


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def macro(records,key,steps):
    grouped=defaultdict(list)
    for r in records:
        value=r['outputs'][str(steps)]['final_clamp'].get(key)
        if value is not None:
            grouped[r['patient_id']].append(value)
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def folder(budget,arm):
    return BASE/arm if budget==2400 else ROOT/f'stage_{budget}'/arm


def main():
    protocol=read(ROOT/'protocol.json')
    audit=read(ROOT/'final_audit.json')
    verification=read(ROOT/'verification.json')
    assert audit['passed'] and verification['passed']
    results={}
    rows=[]
    for budget in (2400,4800,7200):
        results[str(budget)]={}
        for arm in ARMS:
            report=read(folder(budget,arm)/'result.json')
            assert report['complete'] and report['successful_updates']==budget
            steps=report['primary_steps']
            train=next(r for r in report['evaluation'] if r['trained_on_this_patch'])
            held=[r for r in report['evaluation'] if r['split']=='validation']
            item=dict(train=train['outputs'][str(steps)]['final_clamp'],primary_steps=steps,
                train_teacher=train['teacher'],
                held_out_patient_macro={key:macro(held,key,steps) for key in ('mae','ilm_opl_pcc','ilm_opl_ssim','ilm_opl_mae')},
                amp_skipped_updates=report['amp_skipped_updates'],
                resumed_metric_error=report.get('max_resumed_metric_error'))
            results[str(budget)][arm]=item
            rows.append(dict(updates=budget,arm=arm,primary_steps=steps,
                **{'train_'+key:item['train'][key] for key in ('mae','ilm_opl_pcc','ilm_opl_ssim','max_projection_pcc','max_projection_ssim')},
                **{'held_out_'+key:value for key,value in item['held_out_patient_macro'].items()}))
    summary=dict(protocol=protocol,audit=audit,verification=verification,budgets=results)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    with (ROOT/'budget_comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curves={}
    for arm in ARMS:
        by_update={}
        for budget in (2400,4800,7200):
            for record in read(folder(budget,arm)/'history.json'):
                by_update.setdefault(record['update'],record)
        curves[arm]=[by_update[n] for n in sorted(by_update)]
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.5),layout='constrained')
    for arm in ARMS:
        steps=str(results['7200'][arm]['primary_steps'])
        records=curves[arm]
        x=[r['update'] for r in records]
        for ax,key in [(axes[0,0],'ilm_opl_pcc'),(axes[0,1],'ilm_opl_ssim')]:
            ax.plot(x,[r['outputs'][steps]['final_clamp'][key] for r in records],color=COLORS[arm],lw=2,label=NAMES[arm])
        axes[1,0].plot(x,[next(v['velocity_mse'] for v in r['teacher'] if v['time']==0) for r in records],color=COLORS[arm],lw=2,label=NAMES[arm])
        axes[1,1].plot([2400,4800,7200],[results[str(n)][arm]['held_out_patient_macro']['ilm_opl_pcc'] for n in (2400,4800,7200)],color=COLORS[arm],lw=2,marker='o',label=NAMES[arm])
    axes[0,0].set(title='固定训练块：血管结构拟合',xlabel='累计成功更新次数',ylabel='ILM–OPL PCC',ylim=(-.05,1.02))
    axes[0,1].set(title='固定训练块：结构相似度',xlabel='累计成功更新次数',ylabel='ILM–OPL SSIM',ylim=(-.05,1.02))
    axes[1,0].set(title='纯 OCT 起点的速度误差',xlabel='累计成功更新次数',ylabel='t=0 速度 MSE',yscale='log')
    axes[1,1].set(title='两位未训练患者：宏平均相关性',xlabel='累计成功更新次数',ylabel='ILM–OPL PCC',ylim=(-.05,1.02))
    axes[1,1].set_xticks([2400,4800,7200])
    for ax in axes.flat:
        ax.legend(fontsize=8)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):
        ax.axvline(2400,color='#8c9299',ls=':',lw=1)
        ax.axvline(4800,color='#b0b4b9',ls=':',lw=1)
    fig.suptitle('2400 次是否够用？三组继续到相同 7200 次',fontsize=17,fontweight='bold')
    fig.savefig(ROOT/'long_learning_curves.png',dpi=170)
    plt.close(fig)

    trained=protocol['training_patch']
    fig,axes=plt.subplots(3,5,figsize=(14,9.3),layout='constrained')
    with np.load(BASE/'reference_maps'/(trained+'_layers.npz')) as data:
        oct_image,truth=data['oct'].copy(),data['octa'].copy()
    for row,budget in enumerate((2400,4800,7200)):
        images=[oct_image,truth]
        labels=['OCT 输入','真实 OCTA']
        for arm in ARMS:
            steps=results[str(budget)][arm]['primary_steps']
            with np.load(folder(budget,arm)/'maps'/f'{trained}_euler{steps}_layer.npz') as data:
                images.append(data['ilm_opl'].copy())
            labels.append(NAMES[arm]+f"\nPCC {results[str(budget)][arm]['train']['ilm_opl_pcc']:.3f}")
        for col,(a,label) in enumerate(zip(images,labels)):
            axes[row,col].imshow(a,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            axes[row,col].set_xticks([])
            axes[row,col].set_yticks([])
            axes[row,col].set_title(label,fontsize=10)
        axes[row,0].set_ylabel(f'{budget} 次更新',fontsize=11)
    fig.suptitle('同一训练块随预算变化；FM 两组 Euler8，直接监督单步；灰度统一 0–1',fontsize=14)
    fig.savefig(ROOT/'training_progression.png',dpi=180)
    plt.close(fig)
    table=[]
    for r in rows:
        table.append(f"| {r['updates']} | {NAMES[r['arm']]} | {r['train_ilm_opl_pcc']:.4f} | {r['train_ilm_opl_ssim']:.4f} | {r['held_out_ilm_opl_pcc']:.4f} | {r['held_out_ilm_opl_ssim']:.4f} |")
    velocity_rows=[]
    for budget in (2400,4800,7200):
        for arm in ('fm_uniform','fm_half_start'):
            teacher=results[str(budget)][arm]['train_teacher']
            e0=next(r['velocity_mse'] for r in teacher if r['time']==0)
            e5=next(r['velocity_mse'] for r in teacher if r['time']==.5)
            velocity_rows.append(f"| {budget} | {NAMES[arm]} | {e0:.8f} | {e5:.8f} | {e0/e5:.2f} |")
    old=results['2400']; new=results['7200']
    original_gain=new['fm_uniform']['train']['ilm_opl_pcc']-old['fm_uniform']['train']['ilm_opl_pcc']
    gap0=old['fm_half_start']['train']['ilm_opl_pcc']-old['fm_uniform']['train']['ilm_opl_pcc']
    gap1=new['fm_half_start']['train']['ilm_opl_pcc']-new['fm_uniform']['train']['ilm_opl_pcc']
    verdict=f"原FM的训练块PCC从 {old['fm_uniform']['train']['ilm_opl_pcc']:.4f} 变为 {new['fm_uniform']['train']['ilm_opl_pcc']:.4f}（{original_gain:+.4f}）；混合组相对原FM的PCC差值从 {gap0:.4f} 变为 {gap1:.4f}。"
    end_summary='；'.join(NAMES[arm]+f"从 {old[arm]['held_out_patient_macro']['ilm_opl_pcc']:.4f} 到 {new[arm]['held_out_patient_macro']['ilm_opl_pcc']:.4f}" for arm in ARMS)
    temporal0={v['time']:v['velocity_mse'] for v in old['fm_uniform']['train_teacher']}
    temporal1={v['time']:v['velocity_mse'] for v in new['fm_uniform']['train_teacher']}
    thresholds=[]
    for arm in ARMS:
        passed=[str(n) for n in (2400,4800,7200) if results[str(n)][arm]['train']['max_projection_pcc']>=.9 and results[str(n)][arm]['train']['max_projection_ssim']>=.8]
        thresholds.append(NAMES[arm]+'：'+('在'+ '、'.join(passed)+'次节点通过' if passed else '三个节点均未通过'))
    report=f'''# 从2400继续到7200次：训练预算验证

全部训练完成：{audit['completed']}。三组均从各自2400次断点恢复，最终均达到7200次成功更新。新增14400次，含原实验累计21600次。

## 怎样回答“2400次够不够”

2400次只能回答那个预算下的比较，不能证明原FM在更长训练后也学不好。本轮把三组都延长到相同的4800与7200次，保留完整曲线和固定预算结果，检查慢收敛、持续改善与训练/新患者差距。

{verdict}

本轮的直接回答是：**2400次对混合组尚不足以充分记住训练块，延长确实有收益；但在当前随机时间与固定学习率设置下，原FM增加到7200次仍没有持续恢复血管结构，因此不能仅用“训练次数少”解释它的失败。** 原FM后段有明显波动，不应把某个末轮分数单独当作训练更久必然更差的证据。

新患者PCC的2400→7200变化为：{end_summary}。这两位患者上，更多更新没有带来泛化改善。训练块记忆更充分和新患者有效是不同问题。

沿用上一轮预先写下的**全Z最大投影PCC≥0.90且SSIM≥0.80**这一单块记忆参考标准，{'；'.join(thresholds)}。该阈值用于记忆诊断，不代表新患者泛化。

这些是固定体块、单种子的观察，不是收敛极限或全数据性能排名。7200次同样是有限预算，不能据此断言再训练永远不会改善。

![延长训练曲线]({(ROOT/'long_learning_curves.png').as_posix()})

## 固定预算的结果

全部为ILM–OPL最大投影指标；原FM与混合组使用纯OCT起步Euler8，直接监督组使用正常的单步预测。两位未训练患者（10302、10319），各两块，先平均患者内体块再等权平均患者。

| 累计更新数 | 训练方式 | 训练PCC↑ | 训练SSIM↑ | 未训练患者PCC↑ | 未训练患者SSIM↑ |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

![训练块随预算变化]({(ROOT/'training_progression.png').as_posix()})

## 原FM在学什么：起点与中间状态的速度误差

同一个训练块，直接比较速度MSE，不经过ODE：

| 累计更新数 | 训练方式 | t=0速度MSE | t=0.5速度MSE | 起点/中间误差比 |
|---|---|---:|---:|---:|
{chr(10).join(velocity_rows)}

t=0.5输入含有真实OCTA，不能当作有效生成输入；其误差下降与纯OCT起步的生成改善需要分开判断。直接监督组未训练t>0，因此不以它的中间状态误差判断正常单步方法。

原FM从2400到7200次，t=0.5速度MSE下降约 {(1-temporal1[.5]/temporal0[.5])*100:.1f}%，而t=0误差变化为 {(temporal1[0]/temporal0[0]-1)*100:+.1f}%。这支持“中间路径拟合改善没有转化为起点能力”的解释，但没有证明网络内部精确使用了哪一种捷径。

## 控制条件与恢复检查

- 共同起始模型和同一训练体块沿用原实验；原2400次目录未改写。
- 从2400次模型、AdamW动量/二阶矩、AMP scaler继续训练；核对所有参数的优化器计数为2400，而不是重建优化器重新开始。
- 学习率始终2e-4、weight_decay0、clip1；原FM随机t、直接监督全t=0、混合组恰好50%t=0。
- 重新生成7200项时间序列，并验证前2400项与原序列逐元素一致；后续序列在训练前冻结。
- 每组先到4800，再全部到7200；采用固定最终节点，不根据验证集挑选检查点或调整学习率。
- 每段恢复后，先比较原断点的生成指标，要求误差小于1e-6；全部通过。每400次记录训练块曲线。
- 无小波、新损失或额外标签监督。真实层边界只用于推理后的投影评价。

审计逐条核对新增14400次成功更新、时间序列和优化器计数，并检查48份新增三维预测。独立复核见 `verification.json`。

## 解释边界

训练块变好说明记忆/拟合能力改善。新患者结果仍然需要单独看；单块反复训练不能检验覆盖多种病变的泛化能力。训练高分不能替代新患者评价。

下一轮更有信息量的方向是扩到固定4个训练病例，继续保留原FM、直接监督、混合组的同预算对照；检验起点监督能否帮助多个输入学习共同映射。50%起点比例是本次诊断设置，不是已选出的最佳超参数。

“原FM”指原来的随机时间训练方式，它在本次对照中与另两组使用相同样本和预算；不是原多病例训练的200轮权重。

## 文件与存储

服务器：`/docsystem/niuke/lvyuehang/fm_start_long_20261022`。原2400次结果：`/docsystem/niuke/lvyuehang/fm_start_control_20261022`。

新结果分别在 `stage_4800/`、`stage_7200/` 的三组子目录中。模型、优化器、三维预测保留服务器。本地保存小型指标、图像和报告；原始表格见 `budget_comparison.csv`，完整数值见 `summary.json`，恢复与冻结协议见 `protocol.json`。
'''
    (ROOT/'延长训练报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(verdict=verdict,budgets=results),ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
