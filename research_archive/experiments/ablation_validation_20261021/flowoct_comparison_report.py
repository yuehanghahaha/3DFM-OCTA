"""Compare original five runs with the same-tensor FlowOCT and causal controls."""
import csv,html,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import numpy as np
from PIL import Image,ImageDraw
from engine import ARMS,font,layer_projection,write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def picture(samples,columns,destination,title):
    with np.load(ROOT/samples['file'],allow_pickle=False) as f:
        target,layers=f['octa'],f['layers']
    truth,_=layer_projection(target,layers)
    maps=[('Real OCTA',truth)]
    for name,path in columns:
        with np.load(path,allow_pickle=False) as f:
            projection,_=layer_projection(f['octa'],layers)
        maps.append((name,projection))
    canvas=Image.new('RGB',(232*len(maps),300),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((8,8),title+' | same ILM-OPL / fixed [0,1]',font=font(16),fill='black')
    for index,(name,a) in enumerate(maps):
        tile=Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8))
        draw.text((8+index*232,34),name,font=font(15),fill='black')
        canvas.paste(tile.resize((220,220),Image.Resampling.NEAREST),(8+index*232,60))
    canvas.save(destination)


def main():
    output=ROOT/'reports'
    selection=read(ROOT/'data/selection.json')
    suite=read(ROOT/'suite.json')
    rows=[]
    text=['# FlowOCT 同数据对照与故障排查','','同一病例、冻结体块、强度和配对真值；没有重新归一化或挑选图像。',
          '前五组仍使用原始保存结果。本次 FlowOCT 保留公开代码的二维主干和四项损失。','']
    for stage in suite['stages']:
        name=stage['name']
        text += [f'## {name}：每组 {stage["updates"]} 次更新','',
            '| 模型/推理 | 训练 PCC | 训练 SSIM | 验证 PCC | 验证 SSIM | 验证血管区 MAE |',
            '|---|---:|---:|---:|---:|---:|']
        entries=[(arm,'Euler8',ROOT/arm/name/'final_metrics.json') for arm in ARMS]
        entries += [('06_flowoct',mode,ROOT/'06_flowoct'/name/('final_'+mode+'.json')) for mode in ('native3','shared8')]
        for arm,variant,path in entries:
            report=read(path)
            row=dict(stage=name,arm=arm,solver=variant,updates=report['completed_updates'])
            for phase in ('train','validation'):
                row.update({phase+'_'+k:v for k,v in report['summary'][phase]['metrics'].items() if k in
                    ('ilm_opl_pcc','ilm_opl_ssim','ilm_opl_vessel_mae','volume_mae','volume_psnr')})
            rows.append(row)
            values=[row[k] for k in ('train_ilm_opl_pcc','train_ilm_opl_ssim','validation_ilm_opl_pcc',
                    'validation_ilm_opl_ssim','validation_ilm_opl_vessel_mae')]
            text.append('| '+arm+' / '+variant+' | '+' | '.join('null' if v is None else f'{v:.4f}' for v in values)+' |')
        text += ['']
        for sample in selection['patches']:
            if sample['patch_id']!='y064_x064': continue
            if name=='one_case' and sample['split']=='train' and sample['case_id']!='10301': continue
            sample_name=sample['case_id']+'_'+sample['patch_id']+'.npz'
            columns=[(arm[:2],ROOT/arm/name/'predictions'/sample_name) for arm in ARMS]
            columns += [('FlowOCT 3',ROOT/'06_flowoct'/name/'predictions_native3'/sample_name),
                        ('FlowOCT 8',ROOT/'06_flowoct'/name/'predictions_shared8'/sample_name)]
            if all(p.exists() for _,p in columns):
                picture(sample,columns,output/f'flowoct_{name}_{sample["split"]}_{sample["case_id"]}.png',
                        f'{name} | {sample["case_id"]} {sample["split"]}')
    text += ['## 单块机制对照','',
        '| 对照 | 更新次数 | 训练 PCC | 训练 SSIM | 验证 PCC | 验证 SSIM |',
        '|---|---:|---:|---:|---:|---:|']
    for kind in ('time_scale_1','start_only_mse'):
        checkpoints=[400] if kind=='time_scale_1' else [400,800,1600]
        for updates in checkpoints:
            report=read(ROOT/'diagnostics'/kind/f'metrics_{updates:05d}.json')
            values=[report['summary'][phase]['metrics'][key] for phase in ('train','validation') for key in ('ilm_opl_pcc','ilm_opl_ssim')]
            text.append('| '+kind+f' | {updates} | '+' | '.join('null' if v is None else f'{v:.4f}' for v in values)+' |')
    text += ['','### 同为 1600 次更新的长预算对照','',
        '| 单病例模型 | 训练 PCC | 训练 SSIM | 验证 PCC | 验证 SSIM |',
        '|---|---:|---:|---:|---:|']
    for kind in ('uniform_fm_1600','flowoct_1600','start_only_mse'):
        report=read(ROOT/'diagnostics'/kind/'final_metrics.json')
        values=[report['summary'][phase]['metrics'][key] for phase in ('train','validation') for key in ('ilm_opl_pcc','ilm_opl_ssim')]
        text.append('| '+kind+' | '+' | '.join('null' if v is None else f'{v:.4f}' for v in values)+' |')
    text += ['','长预算 FM 与 FlowOCT 从各自原始 400 次检查点继续训练，完整恢复优化器/AMP 状态，不覆盖原检查点。',
             '纯起点 MSE 对照与原 3D FM 共用初始参数；仍须区分其直接输出与自由积分输出，成功记忆不代表泛化成功。']
    text += ['','`time_scale_1` 与原 01 单病例实验共用初始化参数、400 次更新、相同时间序列及优化器，仅将时间编码尺度从 1000 改为 1。',
        '`start_only_mse` 是纯 OCT 起点的直接拟合能力检查：训练全部 t=0，评价 OCT+v(OCT,0,OCT)，不把它冒充训练好的多步 FM。',
        '后者训练 1600 次，另保留 400/800 次指标；不能把其更长训练和直接预测分数与 400 次自由积分混为同预算方法排名。','',
        '## 真实插值状态与纯 OCT 生成','',
        '下表仅取预先固定的训练块 10301/y064_x064。t>0 的网络状态含真实 OCTA，终点高分不能当成可部署的生成结果。',
        '同时比较速度 MSE，避免只看已经混入目标的终点相关性。','',
        '| 模型 | 阶段 | t=0 速度 MSE | t=.05 速度 MSE | t=0 终点 PCC | t=.05 终点 PCC | 纯 OCT 8步 PCC |',
        '|---|---|---:|---:|---:|---:|---:|']
    probes=read(ROOT/'diagnostics/teacher_probe.json')
    for kind in ('uniform_fm_1600','flowoct_1600'):
        probes['records'].append(read(ROOT/'diagnostics'/kind/'teacher_probe.json'))
    for record in probes['records']:
        if record['split']!='train': continue
        zero=next(p for p in record['probes'] if p['mode']=='teacher_state' and p['time']==0)
        early=next(p for p in record['probes'] if p['mode']=='teacher_state' and p['time']==.05)
        free=next(p for p in record['probes'] if p['mode']=='OCT_only_shared8' or (p['mode']=='OCT_only_rollout' and p['steps']==8))
        values=[zero['velocity_mse'],early['velocity_mse'],zero['metrics']['ilm_opl_pcc'],early['metrics']['ilm_opl_pcc'],free['metrics']['ilm_opl_pcc']]
        text.append('| '+record['model']+' | '+record['stage']+' | '+' | '.join('null' if v is None else f'{v:.6f}' for v in values)+' |')
    sample=next(s for s in selection['patches'] if s['case_id']=='10301' and s['patch_id']=='y064_x064')
    file='10301_y064_x064.npz'
    columns=[('01 FM 400',ROOT/'01_fm3d/one_case/predictions'/file),
             ('time=1 / 400',ROOT/'diagnostics/time_scale_1/predictions'/file),
             ('t0 direct / 1600',ROOT/'diagnostics/start_only_mse/predictions'/file),
             ('FlowOCT8 / 400',ROOT/'06_flowoct/one_case/predictions_shared8'/file)]
    if all(p.exists() for _,p in columns):
        picture(sample,columns,output/'flowoct_diagnostic_controls.png','10301 train | diagnostic controls; budgets/outputs differ')
    columns=[('3DFM8 /1600',ROOT/'diagnostics/uniform_fm_1600/predictions'/file),
             ('FlowOCT8 /1600',ROOT/'diagnostics/flowoct_1600/predictions'/file),
             ('t0 direct /1600',ROOT/'diagnostics/start_only_mse/predictions'/file)]
    if all(p.exists() for _,p in columns):
        picture(sample,columns,output/'flowoct_matched1600.png','10301 train | all 1600 updates; direct t0 is capacity control')
    text += ['','## 比较口径','',
        '- 数据完全相同：直接读原冻结 [0,1] Z256 体块，不使用原 loader 的 resize/min–max/flip。FlowOCT 只无损转换 ZYX/YXZ 轴序。',
        '- 训练病例、体块顺序与更新次数一致；FlowOCT 原生抽取 Y 轴四分之一的 B-scan，每片单独采样 t。不能声称逐体素曝光、时间样本和计算量完全相同。',
        '- FlowOCT 保留 Adam/FP32/无梯度裁剪；3D 使用原 AdamW/AMP/裁剪。此对照比较的是实现整体，不是只改变二维/三维的单因素实验。',
        '- native3 是作者每步裁剪的三步 Euler；shared8 是同一权重、统一八步且仅最终裁剪。两行不是两次独立训练。',
        '- 作者投影损失是所抽 B-scan 的最大值，不能改称 Z 均值投影或 ILM–OPL 投影。',
        '- FlowOCT 网络没有原 3D 代码额外拼接的固定 OCT 条件。其重建来自含目标的插值终点估计，与第五组自由积分重建不是同一个损失。',
        '- 数据和目标相同不等于复现整篇论文：这轮仍是小体块短训练和 4 位验证患者，尚无多种子或全视野最终测试。',
        '- 既有投影索引/device 修复和全部源码哈希见 source/flowoct_snapshot.json；没有修改原五组训练源文件或权重。','']
    (output/'FLOWOCT_COMPARISON.md').write_text('\n'.join(text),encoding='utf-8')
    with (output/'flowoct_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output/'flowoct_comparison.json',dict(rows=rows,scope='same frozen tensors; method-level comparison'))
    figures=[]
    for path in sorted(output.glob('flowoct_*.png')):
        figures.append(f'<figure><figcaption>{html.escape(path.stem)}</figcaption><a href="{path.name}"><img src="{path.name}" loading="lazy"></a></figure>')
    tables=[]
    for stage in suite['stages']:
        body=''
        for row in rows:
            if row['stage']!=stage['name']: continue
            body+='<tr><th>'+row['arm']+' / '+row['solver']+'</th>'+''.join(f'<td>{row[k]:.4f}</td>' for k in
                ('train_ilm_opl_pcc','train_ilm_opl_ssim','validation_ilm_opl_pcc','validation_ilm_opl_ssim','validation_ilm_opl_vessel_mae'))+'</tr>'
        tables.append(f'<h2>{stage["name"]} · {stage["updates"]} 次更新</h2><div class="scroll"><table><tr><th>模型 / 求解器</th><th>训练 PCC</th><th>训练 SSIM</th><th>验证 PCC</th><th>验证 SSIM</th><th>验证血管区 MAE</th></tr>'+body+'</table></div>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FlowOCT 同数据验证</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#233144;max-width:1800px;margin:32px auto;padding:0 20px;background:#fafbfd}a{color:#245c8f}table{border-collapse:collapse;min-width:900px;width:100%;font-size:14px;background:white}th,td{padding:11px;border-bottom:1px solid #dbe2ea;text-align:right}th:first-child{text-align:left}.scroll{overflow:auto}figure{margin:24px 0;border:1px solid #dbe2ea;background:white;padding:12px}img{max-width:100%;height:auto}nav a{margin-right:22px}</style><h1>FlowOCT 同数据验证与排查</h1><p>相同冻结体块、同一评价目标；原生二维方法与三维实现整体比较。FlowOCT 两种推理使用同一训练权重。</p><nav><a href="FLOWOCT_DIAGNOSIS.md">排查结论</a><a href="FLOWOCT_COMPARISON.md">完整表格与口径</a><a href="flowoct_comparison.csv">下载 CSV</a><a href="index.html">原五组报告</a></nav>'''
    page+=''.join(tables)+'<h2>固定窗对照图</h2>'+''.join(figures)+'</html>'
    long_rows=''
    for kind in ('uniform_fm_1600','flowoct_1600','start_only_mse'):
        report=read(ROOT/'diagnostics'/kind/'final_metrics.json')
        values=[report['summary'][phase]['metrics'][key] for phase in ('train','validation') for key in ('ilm_opl_pcc','ilm_opl_ssim')]
        long_rows+='<tr><th>'+kind+'</th>'+''.join('<td>'+('null' if v is None else f'{v:.4f}')+'</td>' for v in values)+'</tr>'
    long_table='<h2>同为1600次 · 单病例对照</h2><p>前两行采用8步自由生成；最后一行是纯起点直接拟合能力检查。高训练分数不能代替验证泛化。</p><div class="scroll"><table><tr><th>对照</th><th>训练 PCC</th><th>训练 SSIM</th><th>验证 PCC</th><th>验证 SSIM</th></tr>'+long_rows+'</table></div>'
    page=page.replace('<h2>固定窗对照图</h2>',long_table+'<h2>固定窗对照图</h2>')
    (output/'flowoct_comparison.html').write_text(page,encoding='utf-8')
    print('Wrote FlowOCT comparison tables and fixed-window images',flush=True)


if __name__=='__main__': main()
