"""Independently recompute volume/projection scores for both continuation budgets."""
import hashlib
import json
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

ROOT=Path(__file__).resolve().parent


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert json.loads((ROOT/'final_audit.json').read_text())['passed']
    assert hashlib.sha256((ROOT/'long_control.py').read_bytes()).hexdigest()==protocol['run_script_sha256']
    for arm,record in protocol['resume'].items():
        original=Path(protocol['original_study'])/arm/'update_2400.pt'
        assert hashlib.sha256(original.read_bytes()).hexdigest()==record['original_sha256']
        assert hashlib.sha256((ROOT/'seeds'/(arm+'.pt')).read_bytes()).hexdigest()==record['sha256']
    checks=[]
    for budget in (4800,7200):
        for arm in protocol['arms']:
            folder=ROOT/f'stage_{budget}'/arm
            report=json.loads((folder/'result.json').read_text())
            prior=Path(protocol['original_study'])/arm/'result.json' if budget==4800 else ROOT/'stage_4800'/arm/'result.json'
            previous=json.loads(prior.read_text())
            assert report['loaded_state_sha256']==previous['final_state_sha256']
            assert report['optimizer_restored_step']==budget-2400
            assert report['optimizer_final_step']==report['successful_updates']==budget
            for row in report['evaluation']:
                with np.load(folder/'predictions'/(row['id']+'.npz')) as saved,np.load(row['file']) as inputs:
                    a,b=saved['octa'].copy(),inputs['octa'][0,0].copy()
                    assert np.array_equal(saved['octa_gt'],b)
                    assert a.shape==b.shape==(256,64,64) and np.isfinite(a).all()
                    assert 0<=a.min()<=a.max()<=1
                metrics=row['outputs'][str(report['primary_steps'])]['final_clamp']
                delta=a.astype(np.float64)-b
                assert abs(np.mean(delta**2)-metrics['mse'])<1e-10
                assert abs(np.mean(np.abs(delta))-metrics['mae'])<1e-10
                if 'annotation' in row:
                    with np.load(row['annotation']) as annotation:
                        layers=annotation['layers'].copy()
                    z=((np.arange(256)+.5)*640/256-.5)[:,None,None]
                    mask=(z>=layers[0]-1)&(z<=layers[2]-1)
                    pa,pb=(np.where(mask,v,0).max(0) for v in (a,b))
                    corr=float(np.corrcoef(pa.ravel().astype(np.float64),pb.ravel().astype(np.float64))[0,1])
                    ssim=float(structural_similarity(pa,pb,data_range=1.,win_size=7))
                    assert abs(corr-metrics['ilm_opl_pcc'])<1e-8
                    assert abs(ssim-metrics['ilm_opl_ssim'])<1e-8
                    with np.load(folder/'maps'/f"{row['id']}_euler{report['primary_steps']}_layer.npz") as image:
                        assert np.array_equal(image['ilm_opl'],pa)
                checks.append(dict(budget=budget,arm=arm,id=row['id'],passed=True))
    assert len(checks)==48
    (ROOT/'verification.json').write_text(json.dumps(dict(passed=True,verified_predictions=48,
        independent_projection_recomputation=True,optimizer_resume_counters_verified=True,records=checks),indent=2),encoding='utf-8')
    print('Independent verification:48 predictions passed')


if __name__=='__main__':
    main()
