"""Prove the larger cubes and previous patches share the same physical data."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import numpy as np
from engine import write_json
selection=json.loads((ROOT/'data/selection.json').read_text())
old=ROOT.parent/'fm3d_validation_20261021/data/patches'
checks=0
for record in selection['patches']:
    with np.load(ROOT/record['file'],allow_pickle=False) as current:
        assert current['oct'].shape==current['octa'].shape==(256,256,256)
        for y,x in ((64,64),(120,120)):
            sy,sx=y-record['origin_ZYX'][1],x-record['origin_ZYX'][2]
            with np.load(old/(record['case_id']+f'_y{y:03d}_x{x:03d}.npz'),allow_pickle=False) as previous:
                for key in ('oct','octa','layers','vessel_mask'):
                    a=current[key]
                    region=a[...,sy:sy+64,sx:sx+64]
                    assert np.array_equal(region,previous[key]),(record['case_id'],key,y,x)
                    checks+=1
result=dict(status='passed',patients=len(selection['patches']),previous_patches_checked=10,
    exact_array_comparisons=checks,fields=['oct','octa','layers','vessel_mask'],
    description='Both old64 regions are exact subsets of new central256 cubes; no lateral resampling.')
write_json(ROOT/'verification/data_alignment.json',result)
print(json.dumps(result))
