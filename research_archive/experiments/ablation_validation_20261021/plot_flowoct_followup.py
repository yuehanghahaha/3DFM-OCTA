"""Matched-budget learning curves; distinguish direct t0 output from FM sampling."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def history(folders,initial):
    records={0:read(initial)['summary']}
    for folder in folders:
        for line in (folder/'history.jsonl').read_text().splitlines():
            row=json.loads(line)
            records[row['update']]=row['summary']
    return sorted(records.items())


def main():
    experiments=[
        ('3D FM: Euler8','#53657b','-',
         [ROOT/'01_fm3d/one_case',ROOT/'diagnostics/uniform_fm_1600'],ROOT/'01_fm3d/one_case/initial_metrics.json'),
        ('FlowOCT: Euler8','#d35655','-',
         [ROOT/'06_flowoct/one_case',ROOT/'diagnostics/flowoct_1600'],ROOT/'06_flowoct/one_case/initial_shared8.json'),
        ('t0 direct: capacity control','#2e927d','--',
         [ROOT/'diagnostics/start_only_mse'],ROOT/'diagnostics/start_only_mse/initial_metrics.json'),
        ('3D FM: time scale1 (400)','#b08a28',':',
         [ROOT/'diagnostics/time_scale_1'],ROOT/'diagnostics/time_scale_1/initial_metrics.json')]
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.8),constrained_layout=True)
    specifications=[('train','ilm_opl_pcc','Training ILM-OPL PCC'),
                    ('validation','ilm_opl_pcc','Validation ILM-OPL PCC'),
                    ('train','ilm_opl_ssim','Training ILM-OPL SSIM'),
                    ('validation','ilm_opl_ssim','Validation ILM-OPL SSIM')]
    for label,color,style,folders,initial in experiments:
        records=history(folders,initial)
        for axis,(phase,metric,title) in zip(axes.flat,specifications):
            pairs=[(step,r[phase]['metrics'][metric]) for step,r in records]
            pairs=[p for p in pairs if p[1] is not None]
            x,y=zip(*pairs)
            axis.plot(x,y,color=color,linestyle=style,marker='o',markersize=3.5,linewidth=1.7,label=label)
    for axis,(_,_,title) in zip(axes.flat,specifications):
        axis.set_title(title,loc='left',fontweight='bold',fontsize=11)
        axis.set_xlim(0,1600); axis.set_ylim(-.1,1.02)
        axis.set_xlabel('Total successful optimizer updates')
        axis.spines[['top','right']].set_visible(False)
        axis.grid(alpha=.2)
    axes[0,0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Same training patch | direct t0 control is not an ODE model | validation: 4 patients',fontsize=12)
    path=ROOT/'reports/flowoct_learning_curves.png'
    fig.savefig(path,dpi=180,facecolor='white')
    plt.close(fig)
    print(path)


if __name__=='__main__': main()
