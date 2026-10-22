"""Verify same-data follow-up, preserve original evidence, export small results."""
import hashlib,json,math,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def main():
    selection=read(ROOT/'data/selection.json')
    targets={}
    for sample in selection['patches']:
        path=ROOT/sample['file']
        assert sha(path)==sample['sha256']
        with np.load(path,allow_pickle=False) as f:
            targets[sample['case_id']+'_'+sample['patch_id']]=f['octa'].copy()
    original=read(ROOT/'01_fm3d/one_case/protocol.json')['source_sha256']
    for name,digest in original.items(): assert sha(ROOT/name)==digest,name
    inventory=read(ROOT/'server_artifacts.json')['artifacts']
    preserved=0
    for item in inventory:
        if item['path'].endswith('.pt'):
            assert sha(Path(item['path']))==item['sha256'],item['path']
            preserved+=1
    assert preserved==20
    count=0
    def check_predictions(report_path,folder):
        nonlocal count
        result=read(report_path)
        for row in result['cases']:
            assert row['split'] in ('train','validation')
            name=row['case_id']+'_'+row['patch_id']
            with np.load(folder/(name+'.npz'),allow_pickle=False) as f:
                a,b=f['octa'],f['octa_gt']
                assert a.shape==b.shape==(256,64,64)
                assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
                np.testing.assert_array_equal(b,targets[name])
                mae=float(np.abs(a.astype(np.float64)-b).mean())
                assert abs(mae-row['metrics']['volume_mae'])<1e-9
            assert (folder/(name+'.png')).is_file()
            count+=1
        return result
    for stage,updates in (('one_case',400),('few_case',800)):
        root=ROOT/'06_flowoct'/stage
        with np.load(ROOT/'01_fm3d'/stage/'schedule.npz',allow_pickle=False) as f: reference=f['sample_indices'].copy()
        with np.load(root/'schedule.npz',allow_pickle=False) as f: np.testing.assert_array_equal(f['sample_indices'],reference)
        training=[s for s in selection['patches'] if s['split']=='train']
        if stage=='one_case': training=[s for s in training if s['case_id']+'_'+s['patch_id']==selection['one_case_patch']]
        steps={r['update']:r for line in (root/'steps.jsonl').read_text().splitlines() if (r:=json.loads(line))}
        assert set(steps)==set(range(1,updates+1))
        for update,row in steps.items():
            expected=training[int(reference[update-1])]
            assert (row['case_id'],row['patch_id'])==(expected['case_id'],expected['patch_id'])
            assert abs(row['loss']-(10*row['vloss']+.5*row['L1loss']+.005*row['contrastloss']+.1*row['proj_loss']))<1e-6
        for mode in ('native3','shared8'):
            result=check_predictions(root/('final_'+mode+'.json'),root/('predictions_'+mode))
            assert result['completed_updates']==updates
            old=read(ROOT/'01_fm3d'/stage/'final_metrics.json')
            assert {(r['case_id'],r['patch_id'],r['split']) for r in result['cases']}=={(r['case_id'],r['patch_id'],r['split']) for r in old['cases']}
    for kind in ('time_scale_1','start_only_mse','uniform_fm_1600','flowoct_1600'):
        root=ROOT/'diagnostics'/kind
        result=check_predictions(root/'final_metrics.json',root/'predictions')
        if (root/'final_native3.json').exists(): check_predictions(root/'final_native3.json',root/'predictions_native3')
        assert result['amp_skipped']==0
        steps={r['update']:r for line in (root/'steps.jsonl').read_text().splitlines() if (r:=json.loads(line))}
        expected=set(range(401,1601)) if kind in ('uniform_fm_1600','flowoct_1600') else set(range(1,401 if kind=='time_scale_1' else 1601))
        assert set(steps)==expected
        if kind=='start_only_mse': assert all(r['time']==0 for r in steps.values())
        if kind=='time_scale_1':
            with np.load(ROOT/'01_fm3d/one_case/schedule.npz',allow_pickle=False) as f:
                assert all(r['time']==float(f['times'][u-1]) for u,r in steps.items())
        if kind in ('uniform_fm_1600','flowoct_1600'):
            cfg=read(root/'config.json')
            assert sha(Path(cfg['original_checkpoint']))==cfg['original_sha256']
            assert result['completed_updates']==1600 and result['added_updates']==1200
    one=read(ROOT/'diagnostics/time_scale_1/config.json')
    two=read(ROOT/'diagnostics/start_only_mse/config.json')
    assert one['initial_parameter_sha256']==two['initial_parameter_sha256']
    assert count==95,count
    audit=dict(status='passed',flowoct_added_updates=1200,short_diagnostic_updates=2000,
        long_added_updates=2400,total_added_updates=5600,new_3d_predictions_checked=count,
        original_checkpoints_hash_unchanged=preserved,original_training_sources_unchanged=True,
        controls='Same frozen data and case order. FlowOCT retains its native sampling/loss/optimizer; not a dimensionality-only ablation.',
        outputs='All new GT arrays exactly match the frozen cache; finite [0,1] predictions; independently recomputed saved volume MAE.')
    (ROOT/'verification/flowoct_followup_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    roots=[ROOT/'06_flowoct',ROOT/'diagnostics']
    artifacts=[]
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_file() and (path.suffix=='.pt' or (path.suffix=='.npz' and path.parent.name.startswith('predictions'))):
                artifacts.append(dict(path=str(path),relative_path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    (ROOT/'flowoct_followup_artifacts.json').write_text(json.dumps(dict(artifacts=artifacts),indent=2),encoding='utf-8')
    paths=[]
    for root in roots:
        paths+=list(root.rglob('*'))
    paths+=list((ROOT/'reports').glob('flowoct*'))+list((ROOT/'reports').glob('FLOWOCT*'))
    paths+=[ROOT/'flowoct_followup_artifacts.json',ROOT/'verification/flowoct_followup_audit.json',
            ROOT/'verification/server_flowoct_tests.txt']
    paths+=list((ROOT/'logs').glob('*flowoct*'))+list((ROOT/'logs').glob('long_*.log'))
    archive=ROOT/'flowoct_followup_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as f:
        for path in sorted(set(paths)):
            if not path.is_file() or '__pycache__' in path.parts: continue
            if path.suffix in ('.json','.jsonl','.log','.txt','.md','.png','.csv','.html') or path.name=='schedule.npz':
                f.write(path,path.relative_to(ROOT).as_posix())
    print(json.dumps(dict(audit=audit,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),large_artifacts=len(artifacts)),indent=2),flush=True)


if __name__=='__main__': main()
