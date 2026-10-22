"""Audit actual experiment controls and every exported prediction after completion."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))

import numpy as np
from engine import ARMS,write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    suite,selection=read(ROOT/'suite.json'),read(ROOT/'data/selection.json')
    target_cache={}
    for sample in selection['patches']:
        with np.load(ROOT/sample['file'],allow_pickle=False) as f:
            target_cache[sample['case_id']+'_'+sample['patch_id']]=f['octa'].copy()
    checked_runs=[]
    checked_predictions=0
    for stage in suite['stages']:
        reference_schedule=None
        expected_records=None
        training=[s for s in selection['patches'] if s['split']=='train']
        if stage['name']=='one_case':
            training=[s for s in training if s['case_id']+'_'+s['patch_id']==selection['one_case_patch']]
        for arm in ARMS:
            folder=ROOT/arm/stage['name']
            config=read(ROOT/arm/'config.json')
            result=read(folder/'final_metrics.json')
            assert result['completed_updates']==stage['updates'],folder
            assert result['attempts']-result['amp_skipped']==stage['updates'],folder
            with np.load(folder/'schedule.npz',allow_pickle=False) as f:
                actual=(f['sample_indices'].copy(),f['times'].copy())
            if reference_schedule is None:
                reference_schedule=actual
            else:
                for a,b in zip(actual,reference_schedule):
                    np.testing.assert_array_equal(a,b)
            steps={r['update']:r for line in (folder/'steps.jsonl').read_text().splitlines() if (r:=json.loads(line))}
            assert set(steps)==set(range(1,stage['updates']+1)),folder
            for update,row in steps.items():
                expected=training[int(actual[0][update-1])]
                assert row['case_id']==expected['case_id'] and row['patch_id']==expected['patch_id']
                assert row['time']==float(actual[1][update-1])
                expected_loss=row['fm_loss']+config['lambda_projection']*row['z_projection_loss']+config['lambda_reconstruction']*row['reconstruction_loss']
                assert abs(row['loss']-expected_loss)<1e-6*max(1.,abs(expected_loss))
            for weight,key in (('lambda_projection','z_projection_loss'),('lambda_reconstruction','reconstruction_loss')):
                values=[r[key] for r in steps.values()]
                assert (max(values)>0) if config[weight] else (max(values)==0),folder
            records={(r['case_id'],r['patch_id'],r['split']) for r in result['cases']}
            if expected_records is None:
                expected_records=records
            else:
                assert records==expected_records,folder
            assert all(r['split'] in ('train','validation') for r in result['cases'])
            for row in result['cases']:
                name=row['case_id']+'_'+row['patch_id']
                with np.load(folder/'predictions'/(name+'.npz'),allow_pickle=False) as f:
                    prediction,target=f['octa'],f['octa_gt']
                    assert f['axis_order'].item()=='ZYX'
                    assert prediction.shape==target.shape==(256,64,64)
                    assert np.isfinite(prediction).all() and prediction.min()>=0 and prediction.max()<=1
                    np.testing.assert_array_equal(target,target_cache[name])
                    mae=float(np.abs(prediction.astype(np.float64)-target).mean())
                    assert abs(mae-row['metrics']['volume_mae'])<1e-9
                assert (folder/'predictions'/(name+'.png')).is_file()
                checked_predictions+=1
            assert (folder/'latest.pt').is_file() and (folder/'best_validation.pt').is_file()
            checked_runs.append(dict(arm=arm,stage=stage['name'],updates=result['completed_updates'],
                                     amp_skipped=result['amp_skipped'],predictions=len(result['cases'])))
    report=dict(status='passed',runs=checked_runs,total_successful_updates=sum(r['updates'] for r in checked_runs),
        checked_predictions=checked_predictions,
        controls='Identical per-stage cases, crop IDs, training schedule, times and update counts across five arms',
        outputs='All predictions finite in [0,1]; cached GT exactly equal; saved volume MAE independently recomputed',
        scope='Pipeline and control-variable verification; does not establish clinical efficacy or generalization')
    write_json(ROOT/'verification/final_audit.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
