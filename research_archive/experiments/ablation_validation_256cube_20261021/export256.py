"""Audit final cubes and export a small local report mirror."""
from pathlib import Path
import hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import numpy as np
from engine import ARMS,load_samples,sha256,write_json


def main():
    suite=json.loads((ROOT/'suite.json').read_text())
    samples=load_samples(ROOT,json.loads((ROOT/'data/selection.json').read_text()))
    cases={s['case_id']:s for s in samples}
    checked=0;updates=0;skipped=0
    for arm in [*ARMS,'06_flowoct']:
        output=ROOT/arm/'one_case'
        status=json.loads((output/'status.json').read_text())
        assert status['status']=='complete' and status['updates']==suite['updates']
        updates+=status['updates'];skipped+=status['amp_skipped']
        modes=[('predictions','final_metrics.json')]
        if arm=='06_flowoct':modes.append(('predictions_native3','final_native3.json'))
        for folder,metric in modes:
            report=json.loads((output/metric).read_text())
            metric_rows={r['case_id']:r['metrics'] for r in report['cases']}
            files=list((output/folder).glob('*.npz'))
            assert len(files)==5
            for path in files:
                case=path.name.split('_')[0];sample=cases[case]
                with np.load(path,allow_pickle=False) as f:
                    prediction=f['octa'];truth=f['octa_gt']
                    assert prediction.shape==(256,256,256)
                    assert np.array_equal(truth,sample['octa'])
                    assert np.isfinite(prediction).all() and prediction.min()>=0 and prediction.max()<=1
                    mae=float(np.abs(prediction.astype(np.float64)-truth.astype(np.float64)).mean())
                    assert abs(mae-metric_rows[case]['volume_mae'])<1e-8
                checked+=1
    assert checked==35
    artifacts=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and (p.suffix=='.pt' or p.suffix=='.npz' and p.parent.name.startswith('predictions')):
            artifacts.append(dict(file=p.relative_to(ROOT).as_posix(),server_path=str(p),bytes=p.stat().st_size,sha256=sha256(p)))
    write_json(ROOT/'server_artifacts.json',artifacts)
    audit=dict(status='passed',completed_arms=6,successful_updates=updates,amp_skipped=skipped,
        checked_3d_predictions=checked,large_artifacts=len(artifacts),shape_ZYX=[256]*3,
        target_exactly_matches_frozen_data=True,volume_mae_independently_recomputed=True)
    write_json(ROOT/'verification/final_audit.json',audit)
    metadata=json.loads((ROOT/'server_run.json').read_text())
    metadata.update(state='complete',successful_updates=updates,checked_predictions=checked,
                    server_checkpoints=6,large_artifacts=len(artifacts),amp_skipped=skipped)
    write_json(ROOT/'server_run.json',metadata)
    (ROOT/'last_error.json').unlink(missing_ok=True)
    archive=ROOT/'results256_small.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pt','.zip'):continue
            if p.suffix=='.npz' and p.name!='schedule.npz':continue
            z.write(p,p.relative_to(ROOT).as_posix())
    print(json.dumps(dict(audit=audit,archive=str(archive),bytes=archive.stat().st_size,sha256=sha256(archive))),flush=True)


if __name__=='__main__':main()
