"""Make equal-budget result tables and the same fixed-window views for all arms."""
import csv
import html
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))

import numpy as np
from PIL import Image,ImageDraw
from engine import ARMS,font,layer_projection,write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def overview(stage,selection,completed):
    destination=ROOT/'reports'
    for phase in ('train','validation'):
        candidates=[s for s in selection['patches'] if s['split']==phase and s['patch_id']=='y064_x064']
        # All selected patients are shown; no best/worst example selection.
        for sample in candidates:
            name=sample['case_id']+'_'+sample['patch_id']
            files={arm:ROOT/arm/stage/'predictions'/(name+'.npz') for arm in completed}
            files={arm:p for arm,p in files.items() if p.exists()}
            if not files:
                continue
            with np.load(ROOT/sample['file'],allow_pickle=False) as f:
                target,source,layers=f['octa'],f['oct'],f['layers']
            ground_truth,_=layer_projection(target,layers)
            columns=[('OCT Z mean',source.mean(0)),('Real ILM-OPL',ground_truth)]
            for arm,path in files.items():
                with np.load(path,allow_pickle=False) as f:
                    projection,_=layer_projection(f['octa'],layers)
                columns.append((arm.split('_')[0]+' generated',projection))
            canvas=Image.new('RGB',(232*len(columns),300),'white')
            draw=ImageDraw.Draw(canvas)
            draw.text((8,8),f'{stage} | {sample["case_id"]} {phase} | all columns fixed [0,1]',fill='black',font=font(16))
            for index,(title,values) in enumerate(columns):
                maximum=1.
                tile=Image.fromarray(np.rint(np.clip(values/maximum,0,1)*255).astype(np.uint8))
                canvas.paste(tile.resize((220,220),Image.Resampling.NEAREST),(8+index*232,60))
                draw.text((8+index*232,34),title,fill='black',font=font(16))
            canvas.save(destination/f'{stage}_{phase}_{sample["case_id"]}.png')


def main():
    destination=ROOT/'reports'
    destination.mkdir(exist_ok=True)
    config=read(ROOT/'suite.json')
    selection=read(ROOT/'data/selection.json')
    rows=[]
    lines=['# 五组 3D FM 小样本验证结果','','本报告比较相同成功更新次数的 `latest.pt`，不按最好看的图片或测试集分数挑选结果。',
           '这是固定体块的记忆能力与验证集诊断，不能当成整例或独立测试集的泛化结论。',
           '每个阶段均从头初始化；单病例结果不会作为四病例阶段的初始权重。',
           '第五组使用图像 L1 重建约束，按用户确认不使用 InfoNCE。','']
    panels=[]
    for stage in config['stages']:
        name=stage['name']
        lines += [f'## {name}: 每组 {stage["updates"]} 次成功更新','',
                  '| 组别 | 训练 ILM–OPL PCC | 训练 SSIM | 验证 PCC | 验证 SSIM | 验证血管区 MAE | 验证 3D MAE |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        completed=[]
        for arm in ARMS:
            path=ROOT/arm/name/'final_metrics.json'
            if not path.exists():
                lines.append(f'| {arm} | 等待完成 | | | | | |')
                continue
            result=read(path)
            train=result['summary']['train']['metrics']
            validation=result['summary']['validation']['metrics']
            row=dict(stage=name,arm=arm,updates=result['completed_updates'])
            for phase in ('train','validation'):
                row.update({phase+'_'+k:v for k,v in result['summary'][phase]['metrics'].items()})
            rows.append(row)
            completed.append(arm)
            def fmt(value):
                return 'null' if value is None else f'{value:.4f}'
            values=[train['ilm_opl_pcc'],train['ilm_opl_ssim'],validation['ilm_opl_pcc'],validation['ilm_opl_ssim'],
                    validation['ilm_opl_vessel_mae'],validation['volume_mae']]
            lines.append('| '+arm+' | '+' | '.join(map(fmt,values))+' |')
        overview(name,selection,completed)
        lines += ['','已完成 '+str(len(completed))+'/5 组。']
        if len(completed)==5:
            current=[r for r in rows if r['stage']==name]
            best=max(current,key=lambda r:r['validation_ilm_opl_ssim'])
            lines += [f'本阶段相同预算下验证 ILM–OPL SSIM 最高为 `{best["arm"]}`；这只描述本轮小样本结果，不代表方法已获充分验证。']
        lines += ['']
    lines += ['## 判读规则','',
        '- 训练 PCC/SSIM 都低：当前目标、容量或预算仍不足，不能只根据整体 MAE 认为血管恢复成功。',
        '- 训练明显改善、验证未改善：首先说明能拟合固定训练体块；需要独立验证泛化。',
        '- 03 对 01、04 对 02：投影约束的效果；02 对 01、04 对 03：小波表示的效果；05 对 04：自由生成图像重建监督的效果。',
        '- 体素与小波模型输入/输出通道不同，参数量略有差异，网络处理的空间网格也不同；这是表示实验的一部分。',
        '- ILM–OPL 层标签和血管标注只用于事后评价。阈值 Dice/clDice 是固定规则的二维代理指标，不能表述为可靠三维血管分割或拓扑恢复。',
        '- 4 位验证患者、固定两处体块和单个随机种子只支持可行性检查。正式结论需要多种子、更充分训练、全视野验证和锁定后的测试。',
        '- `best_validation.pt` 是开发用验证 SSIM 最优权重；主表按等更新预算的 `latest.pt` 比较。',
        '- 全部图片固定显示窗；不逐图拉伸。空投影相关性记 null，患者内先平均体块、再按患者等权汇总。','']
    (destination/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    if rows:
        with (destination/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(destination/'summary.json',dict(scope='small fixed-patch train/validation ablation',rows=rows))
    for path in sorted(destination.glob('*.png'),key=lambda p:(not p.stem.startswith('learning_curves'),p.name)):
        panels.append(f'<figure><figcaption>{html.escape(path.stem)}</figcaption><a href="{path.name}"><img src="{path.name}" loading="lazy" alt="{html.escape(path.stem)}"></a></figure>')
    labels=dict(zip(ARMS,['01 · 3D FM','02 · +三维小波','03 · +Z 投影',
                         '04 · +三维小波+Z 投影','05 · +三维小波+Z 投影+图像重建']))
    tables=[]
    for stage in config['stages']:
        title='单病例记忆实验' if stage['name']=='one_case' else '四病例实验'
        cells=[]
        for arm in ARMS:
            found=[r for r in rows if r['arm']==arm and r['stage']==stage['name']]
            if not found:
                cells.append('<tr><th>'+labels[arm]+'</th><td colspan="6">等待完成</td></tr>')
                continue
            row=found[0]
            keys=['train_ilm_opl_pcc','train_ilm_opl_ssim','validation_ilm_opl_pcc',
                  'validation_ilm_opl_ssim','validation_ilm_opl_vessel_mae','validation_volume_mae']
            cells.append('<tr><th>'+labels[arm]+'</th>'+''.join('<td>'+('null' if row[k] is None else f'{row[k]:.4f}')+'</td>' for k in keys)+'</tr>')
        tables.append(f'<h2>{title} · 每组 {stage["updates"]} 次更新</h2><div class="scroll"><table><thead><tr><th>实验组</th><th>训练 PCC ↑</th><th>训练 SSIM ↑</th><th>验证 PCC ↑</th><th>验证 SSIM ↑</th><th>验证血管区 MAE ↓</th><th>验证 3D MAE ↓</th></tr></thead><tbody>'+''.join(cells)+'</tbody></table></div>')
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D FM 五组验证</title><style>
body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#233144;background:#fafbfd;max-width:1640px;margin:32px auto;padding:0 20px}
h1{font-size:30px;margin-bottom:6px}h2{font-size:22px;margin-top:32px}a{color:#245c8f}nav{display:flex;flex-wrap:wrap;gap:10px 22px;margin:20px 0}
.scroll{overflow-x:auto;background:white;border:1px solid #dbe2ea;border-radius:8px}table{border-collapse:collapse;width:100%;min-width:880px;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:13px 15px;border-bottom:1px solid #e3e8ef;text-align:right}th:first-child{text-align:left}thead{background:#edf2f7}tbody th{font-weight:500}tbody tr:last-child>*{border-bottom:0}
figure{margin:24px 0;background:white;border:1px solid #dbe2ea;padding:12px;border-radius:8px}figcaption{font-weight:600;margin-bottom:8px;overflow-wrap:anywhere}img{max-width:100%;height:auto;display:block}.note{color:#53657b;max-width:1000px}
</style><h1>3D 流匹配 · 五组实验验证</h1>'''
    body+=f'<p class="note">已完成 {len(rows)}/{5*len(config["stages"])} 个运行。各阶段从头初始化；主表比较相同更新预算的最终权重。PCC/SSIM 来自 ILM–OPL 投影，验证指标按 4 位患者等权汇总。</p>'
    body+='<nav><a href="CONCLUSIONS.md">先看本轮结论</a><a href="RESULTS.md">完整指标与判读</a><a href="summary.csv">下载指标 CSV</a><a href="COMPUTE.md">资源实测</a><a href="../README.md">实验设计与运行方法</a><a href="../SERVER_RUN.md">权重保存位置</a></nav>'
    body+=''.join(tables)+'<p class="note">固定体块、单随机种子的小样本诊断，不能视作完整训练后的方法排名。单病例与四病例阶段预算不同，不能直接归因于病例数变化。图片固定显示窗，点击可查看原图。</p><h2>学习曲线与固定窗对照图</h2>'+''.join(panels)+'</html>'
    (destination/'index.html').write_text(body,encoding='utf-8')
    print('Updated reports/RESULTS.md, summary.csv, index.html',flush=True)


if __name__=='__main__':
    main()
