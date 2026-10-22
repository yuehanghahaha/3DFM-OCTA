"""Native-pixel maps and matched physical ROI comparisons for the 256 pilot."""
import csv,html,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import numpy as np
from PIL import Image,ImageDraw
from engine import ARMS,font,layer_projection,load_samples,sample_metrics,write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def strip(maps,path,title,size=256,maximum=1.):
    canvas=Image.new('RGB',(len(maps)*(size+16)+8,size+86),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((8,8),title,fill='black',font=font(14))
    for i,(label,values) in enumerate(maps):
        assert values.shape==(size,size)
        x=8+i*(size+16)
        draw.text((x,38),label,fill='black',font=font(14))
        tile=Image.fromarray(np.rint(np.clip(values/maximum,0,1)*255).astype(np.uint8))
        canvas.paste(tile,(x,70))
    canvas.save(path)


def main():
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    samples=load_samples(ROOT,read(ROOT/'data/selection.json'))
    suite=read(ROOT/'suite.json')
    labels=['01 3D FM','02 Haar3D','03 +Z','04 Haar3D+Z','05 Haar3D+Z+L1']
    methods=[(name,label,'final_metrics.json','predictions') for name,label in zip(ARMS,labels)]
    methods += [('06_flowoct','FlowOCT3','final_native3.json','predictions_native3'),
                ('06_flowoct','FlowOCT8','final_metrics.json','predictions')]
    for arm,label in [('07_flowoct_fm_only','FlowOCT FM only'),('08_flowoct_wavelet_only','FlowOCT Haar only')]:
        methods += [(arm,label+' 3','final_native3.json','predictions_native3'),
                    (arm,label+' 8','final_metrics.json','predictions')]
    rows=[]
    for arm,label,metric,predictions in methods:
        path=ROOT/arm/'one_case'/metric
        if not path.exists():continue
        result=read(path)
        row=dict(arm=arm,method=label,updates=result.get('completed_updates',suite['updates']))
        for phase in ('train','validation'):
            for key in ('ilm_opl_pcc','ilm_opl_ssim','ilm_opl_vessel_mae','volume_mae'):
                row[phase+'_'+key]=result['summary'][phase]['metrics'][key]
        rows.append(row)
    write_json(reports/'summary.json',rows)
    if rows:
        with (reports/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    figures=[];roi_rows=[]
    old=ROOT.parent/'fm3d_validation_20261021'
    for sample in samples:
        name=sample['case_id']+'_'+sample['patch_id']
        truth,_=layer_projection(sample['octa'],sample['layers'])
        maps=[('Real OCTA',truth)]
        Image.fromarray(np.rint(np.clip(truth,0,1)*255).astype(np.uint8)).save(reports/(sample['case_id']+'_real256.png'))
        for arm,label,metric,predictions in methods:
            path=ROOT/arm/'one_case'/predictions/(name+'.npz')
            if not path.exists():continue
            with np.load(path,allow_pickle=False) as f:
                prediction=f['octa']
                assert np.array_equal(f['octa_gt'],sample['octa'])
            projection,_=layer_projection(prediction,sample['layers'])
            maps.append((label,projection))
            Image.fromarray(np.rint(np.clip(projection,0,1)*255).astype(np.uint8)).save(reports/(sample['case_id']+'_'+label.replace(' / ','_').replace('/','_')+'_256.png'))
            # Compare exactly the same original physical [64:128,64:128] region.
            oy,ox=sample['origin_ZYX'][1:]
            sy,sx=64-oy,64-ox
            if sy<0 or sx<0 or max(sy,sx)+64>256:continue
            region=(slice(None),slice(sy,sy+64),slice(sx,sx+64))
            roi=dict(sample)
            roi.update(oct=sample['oct'][region],octa=sample['octa'][region],
                layers=sample['layers'][:,sy:sy+64,sx:sx+64],vessel_mask=sample['vessel_mask'][sy:sy+64,sx:sx+64])
            new_metrics,new_map,real_map=sample_metrics(prediction[region],roi)
            oldfolder=('predictions_native3' if label=='FlowOCT3' else 'predictions_shared8') if arm=='06_flowoct' else 'predictions'
            oldpath=old/arm/'one_case'/oldfolder/(sample['case_id']+'_y064_x064.npz')
            if oldpath.exists():
                with np.load(oldpath,allow_pickle=False) as f:
                    assert np.array_equal(f['octa_gt'],roi['octa'])
                    old_metrics,old_map,_=sample_metrics(f['octa'],roi)
                roi_rows.append(dict(case_id=sample['case_id'],split=sample['split'],method=label,
                    old64_pcc=old_metrics['ilm_opl_pcc'],new256_pcc=new_metrics['ilm_opl_pcc'],
                    old64_ssim=old_metrics['ilm_opl_ssim'],new256_ssim=new_metrics['ilm_opl_ssim']))
                if sample['split']=='train':
                    filename='matchedROI_'+label.replace(' / ','_')+'.png'
                    strip([('Real OCTA',real_map),('Old64',old_map),('New256',new_map)],reports/filename,
                          'ROI64 | '+label,size=64)
        filename=sample['case_id']+'_'+sample['split']+'_comparison256.png'
        strip(maps,reports/filename,sample['case_id']+' '+sample['split']+' | native256 pixels; fixed [0,1]; central crop; '+str(suite['updates'])+' updates')
        figures.append(filename)
        if sample['split']=='train':
            lowfile='10301_train_shared_lowwindow256.png'
            strip(maps,reports/lowfile,'10301 train | same fixed [0,0.25] window for ALL maps including truth | native256',maximum=.25)
            figures.append(lowfile)
            available=dict(maps)
            chosen=['Real OCTA','01 3D FM','FlowOCT3']
            if all(key in available for key in chosen):
                strip([(key,available[key]) for key in chosen],reports/'preview_train256.png',
                      '10301 train | 400 updates | native256 pixels | fixed [0,1]')
    write_json(reports/'matched_roi64.json',roi_rows)
    matched_train=[r for r in roi_rows if r['split']=='train']
    roi_table='| 方法 | 旧64 PCC | 新256 PCC | 旧64 SSIM | 新256 SSIM |\n|---|---:|---:|---:|---:|'
    roi_lines=['| '+r['method']+' | '+' | '.join(f'{r[k]:.4f}' for k in ('old64_pcc','new256_pcc','old64_ssim','new256_ssim'))+' |' for r in matched_train]
    header='| 方法 | 训练 PCC | 训练 SSIM | 验证 PCC | 验证 SSIM |\n|---|---:|---:|---:|---:|'
    lines=[]
    for row in rows:
        values=[row[phase+'_'+key] for phase in ('train','validation') for key in ('ilm_opl_pcc','ilm_opl_ssim')]
        lines.append('| '+row['method']+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    notes=['# 256³ 同病例试跑结果','',f"训练病例10301，四位验证患者；各方法从头训练 {suite['updates']} 次成功更新。",
        '中央256³直接裁剪，未做横向缩放；仍不等于完整304×304视野。所有结果从纯OCT生成。',
        '原五组保留模型和损失，使用块级及积分步重算；FlowOCT保留原生64张抽片，全局损失不拆，网络分8片重算。',
        '新增07/08分别为无小波/二维Haar，仅保留作者ROI速度MSE；去掉重建、对比、投影监督和对比头。07网络分2片重算，08分8片，每次更新仍监督64片。',
        '同更新数下，256³每次看到的体素与计算量比旧小块更多，不能把收益单独解释为上下文。',
        '主图统一[0,1]；shared_lowwindow图统一[0,0.25]，含真值在内所有方法用同一个显示窗，评分与预测数值不变。',
        '',header,*lines,'','不同视野的PCC/SSIM不宜直接比较；matched_roi64.json另列相同物理64×64区域的前后指标。',
        '单病例、单种子、短程诊断；不能代表完整训练后的泛化性能。','',
        '## 相同训练位置64×64区域对照','',
        '新旧均为400次更新，目标区域逐元素一致。旧模型输入64块，新模型输入包含该区域的256块；同更新数下体素曝光与计算量不同。','',
        roi_table,*roi_lines,'']
    notes += ['!['+f+']('+f+')' for f in figures]
    (reports/'RESULTS.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    tr=''.join('<tr><th>'+html.escape(r['method'])+'</th>'+''.join('<td>'+f"{r[p+'_'+k]:.4f}"+'</td>' for p in ('train','validation') for k in ('ilm_opl_pcc','ilm_opl_ssim'))+'</tr>' for r in rows)
    figures_html=''.join('<figure><figcaption>'+f+'</figcaption><a href="'+f+'"><img src="'+f+'" alt="'+f+'"></a></figure>' for f in figures)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>256³ 同病例试跑</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1900px;margin:30px auto;padding:0 20px;color:#233144;background:#fafbfd}a{color:#245c8f}table{border-collapse:collapse;background:white}th,td{padding:12px 20px;border-bottom:1px solid #dbe2ea;text-align:right}th:first-child{text-align:left}figure{background:white;padding:14px;border:1px solid #dbe2ea}img{max-width:100%;height:auto}figcaption{margin-bottom:10px}.note{max-width:1100px}</style><h1>256³ 同病例试跑</h1>'''
    if (reports/'FLOWOCT_ABLATIONS.md').exists():
        text+='<p class="note"><strong>新增两组观察：</strong>本轮400次更新后，无小波、仅速度匹配的FlowOCT投影PCC/SSIM高于仅小波与完整FlowOCT。只移除辅助监督的收益较小。<a href="FLOWOCT_ABLATIONS.md">查看完整模块对照</a>。这是单病例短程检查，未验证充分收敛后的排名。</p>'
    elif (reports/'CONCLUSIONS.md').exists():
        text+='<p class="note"><strong>原六组观察：</strong>256³可以训练，但400次更新仍未恢复细血管。FlowOCT相对较好；扩大视野没有带来同一训练区域的PCC提升。<a href="CONCLUSIONS.md">阅读原六组结论</a>。</p>'
    text+='<p class="note">中央256³直接裁剪；每格投影原生256×256像素。训练病例10301，验证4位患者；已完成'+str(len(rows))+'/11个方法/求解器结果行，每个完成结果均为'+str(suite['updates'])+'次成功更新。单病例拟合检查，不代表完整训练效果。</p>'
    if (reports/'FLOWOCT_ABLATIONS.md').exists():
        text+='<p class="note"><strong>新增FlowOCT模块对照：</strong>07仅流匹配、08仅二维小波，均关闭重建/对比/投影监督。<a href="FLOWOCT_ABLATIONS.md">实验定义、训练曲线与逐例对照</a> · <a href="flowoct_ablation_summary.csv">新增对照指标</a>。</p>'
        for suffix in ('native3','native3_lowwindow','shared8'):
            f='flowoct_ablations_10301_'+suffix+'.png'
            text+='<figure><figcaption>训练病例：真实 / 仅流匹配 / 仅小波 / 完整FlowOCT（'+suffix+'）</figcaption><a href="'+f+'"><img src="'+f+'" alt="FlowOCT ablations"></a></figure>'
    text+='<p><a href="RESULTS.md">完整口径</a> · <a href="summary.csv">指标CSV</a> · <a href="matched_roi64.json">相同物理64区域对照</a> · <a href="../README.md">设计与资源</a></p>'
    text+='<table><tr><th>方法</th><th>训练PCC</th><th>训练SSIM</th><th>验证PCC</th><th>验证SSIM</th></tr>'+tr+'</table>'
    text+='<p class="note">均为ILM–OPL投影，验证按患者等权平均。256视野与旧64视野的分数不能直接解释为尺寸收益；请结合相同物理ROI比较。</p>'+figures_html+'</html>'
    roi_html='<p class="note">主图使用统一[0,1]显示窗；shared_lowwindow图使用统一[0,0.25]窗，所有方法与真值一致，用于查看低幅度结构。评分不变。</p><h2>训练病例：相同物理64×64区域</h2><p class="note">目标区域逐元素一致，新旧均400次更新；输入视野、体素曝光和计算量不同。</p><table><tr><th>方法</th><th>旧64 PCC</th><th>新256 PCC</th><th>旧64 SSIM</th><th>新256 SSIM</th></tr>'
    roi_html+=''.join('<tr><th>'+html.escape(r['method'])+'</th>'+''.join('<td>'+f'{r[k]:.4f}'+'</td>' for k in ('old64_pcc','new256_pcc','old64_ssim','new256_ssim'))+'</tr>' for r in matched_train)+'</table>'
    text=text.replace('</html>',roi_html+'</html>')
    (reports/'index.html').write_text(text,encoding='utf-8')
    print('Report written:',len(rows),'method/solver rows;',len(figures),'native256 figures')


if __name__=='__main__':main()
