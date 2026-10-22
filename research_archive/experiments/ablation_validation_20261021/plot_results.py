"""Standalone scientific learning curves from saved real evaluation records."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
ARMS=['01_fm3d','02_fm3d_wavelet','03_fm3d_zproj','04_fm3d_wavelet_zproj',
      '05_fm3d_wavelet_zproj_reconstruction']
LABELS=['01 FM','02 FM + wavelet','03 FM + Z projection','04 FM + wavelet + Z',
        '05 FM + wavelet + Z + reconstruction']
COLORS=['#53657b','#2e927d','#d0942b','#7064ac','#d35655']


def main():
    suite=json.loads((ROOT/'suite.json').read_text(encoding='utf-8'))
    for stage in suite['stages']:
        if not any((ROOT/arm/stage['name']/'initial_metrics.json').exists() for arm in ARMS):
            continue
        fig,axes=plt.subplots(2,2,figsize=(11.5,7.6),constrained_layout=True)
        specifications=[('train','ilm_opl_pcc','Training ILM-OPL PCC'),
                        ('validation','ilm_opl_ssim','Validation ILM-OPL SSIM'),
                        ('train','volume_mae','Training volume MAE'),
                        ('validation','ilm_opl_vessel_mae','Validation vessel-region MAE')]
        for arm,label,color in zip(ARMS,LABELS,COLORS):
            folder=ROOT/arm/stage['name']
            records=[]
            if (folder/'initial_metrics.json').exists():
                records.append((0,json.loads((folder/'initial_metrics.json').read_text())['summary']))
            if (folder/'history.jsonl').exists():
                entries={record['update']:record for line in (folder/'history.jsonl').read_text().splitlines()
                         if (record:=json.loads(line))}
                records += [(update,record['summary']) for update,record in sorted(entries.items())]
            for axis,(phase,metric,title) in zip(axes.flat,specifications):
                values=[(update,summary[phase]['metrics'][metric]) for update,summary in records]
                values=[(x,y) for x,y in values if y is not None]
                if values:
                    x,y=zip(*values)
                    axis.plot(x,y,'o-',color=color,label=label,linewidth=1.65,markersize=4)
        for axis,(phase,metric,title) in zip(axes.flat,specifications):
            axis.set_title(title,loc='left',fontsize=11,fontweight='bold')
            axis.set_xlabel('Successful optimizer updates')
            axis.set_xlim(0,stage['updates'])
            axis.grid(alpha=.2)
            axis.spines[['top','right']].set_visible(False)
            if 'pcc' in metric or 'ssim' in metric:
                axis.set_ylim(-.1,1.02)
            else:
                axis.set_ylim(bottom=0)
        axes[0,0].legend(fontsize=8,loc='upper left')
        fig.suptitle(stage['name']+' | fixed training patches; 4 held-out validation patients',fontsize=13)
        output=ROOT/'reports'/('learning_curves_'+stage['name']+'.png')
        fig.savefig(output,dpi=180,facecolor='white')
        plt.close(fig)
        print(output)


if __name__=='__main__':
    main()
