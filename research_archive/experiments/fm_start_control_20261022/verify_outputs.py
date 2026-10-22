"""Independently verify saved 3D results, projection scores and budget records."""
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

ROOT=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    audit=json.loads((ROOT/'final_audit.json').read_text())
    assert audit['passed']
    checked=[]
    reference_initial=None
    with np.load(ROOT/'schedule.npz') as schedules:
        for arm in protocol['arms']:
            result=json.loads((ROOT/arm/'result.json').read_text())
            history=json.loads((ROOT/arm/'history.json').read_text())
            initial=history[0]['outputs']['1']['raw']
            if reference_initial is None:
                reference_initial=initial
            else:
                for key in ('mae','mse','max_projection_pcc','max_projection_ssim'):
                    assert abs(initial[key]-reference_initial[key])<1e-8
            trace=[json.loads(line) for line in (ROOT/arm/'steps.jsonl').read_text().splitlines()]
            assert len(trace)==protocol['updates']==result['successful_updates']
            assert np.array_equal(np.array([r['time'] for r in trace],np.float32),schedules[arm])
            assert result['initial_state_sha256']==protocol['initial_state_sha256']
            assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) and r['gradient_norm']>0 for r in trace)
            for row in result['evaluation']:
                assert digest(row['file'])==row['sha256']
                with np.load(ROOT/arm/'predictions'/(row['id']+'.npz')) as saved, np.load(row['file']) as original:
                    prediction,truth=saved['octa'].copy(),original['octa'][0,0].copy()
                    assert np.array_equal(saved['octa_gt'],truth)
                    assert prediction.shape==truth.shape==(256,64,64)
                    assert np.isfinite(prediction).all() and 0<=prediction.min()<=prediction.max()<=1
                metric=row['outputs'][str(result['primary_steps'])]['final_clamp']
                error=prediction.astype(np.float64)-truth
                assert abs(np.mean(error**2)-metric['mse'])<1e-10
                assert abs(np.mean(np.abs(error))-metric['mae'])<1e-10
                if 'annotation' in row:
                    with np.load(row['annotation']) as annotation:
                        boundaries=annotation['layers'].copy()
                    z=((np.arange(256)+.5)*640/256-.5)[:,None,None]
                    mask=(z>=boundaries[0]-1)&(z<=boundaries[2]-1)
                    a=np.where(mask,prediction,0).max(axis=0)
                    b=np.where(mask,truth,0).max(axis=0)
                    correlation=float(np.corrcoef(a.ravel().astype(np.float64),b.ravel().astype(np.float64))[0,1])
                    ssim=float(structural_similarity(a,b,data_range=1.,win_size=7))
                    assert abs(correlation-metric['ilm_opl_pcc'])<1e-8
                    assert abs(ssim-metric['ilm_opl_ssim'])<1e-8
                    with np.load(ROOT/arm/'maps'/f"{row['id']}_euler{result['primary_steps']}_layer.npz") as image:
                        assert np.array_equal(image['ilm_opl'],a)
                checked.append(dict(arm=arm,id=row['id'],passed=True))
    assert len(checked)==24
    result=dict(passed=True,verified_predictions=24,projection_metric_and_saved_image_recomputation=True,
        schedule_and_nonzero_finite_gradients_verified=True,
        same_initial_source_only_predictions_verified=True,
        run_script_sha256=digest(ROOT/'run_control.py'),
        completed=datetime.now(timezone(timedelta(hours=8))).isoformat(),records=checked,
        gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True),
        jobs=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines())
    result['jobs']=[r for r in result['jobs'] if '/bin/python' in r and ('-m fm3d.train' in r or 'run256.py --arm' in r)]
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','jobs')},indent=2))


if __name__=='__main__':
    main()
