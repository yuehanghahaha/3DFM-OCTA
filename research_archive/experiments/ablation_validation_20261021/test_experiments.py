"""Check actual generation gradients, wavelet correctness, isolation and run artifacts."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))

import numpy as np
import torch
from torch import nn
from engine import (ARMS, aggregate, encode, objectives, rollout, schedule, sha256, write_json)


class RecordingVelocity(nn.Module):
    def __init__(self, value=.1):
        super().__init__()
        self.values = nn.Parameter(torch.full((8,),value))
        self.inputs=[]

    def forward(self,state,time,source):
        self.inputs.append((state.detach().clone(),time.detach().clone(),source.detach().clone()))
        return torch.ones_like(state)*self.values[(time*8).long().clamp_max(7)][:,None,None,None,None]


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rollout_reaches_every_step_and_never_requires_a_target(self):
        source=torch.full((1,1,16,16,16),.2)
        for representation in ('voxel','haar3d'):
            model=RecordingVelocity()
            generated=rollout(model,source,representation,8)
            self.assertEqual(generated.shape,source.shape)
            generated.square().mean().backward()
            self.assertTrue(torch.all(model.values.grad.abs()>0))
            self.assertEqual(len(model.inputs),8)
            condition=encode(source,representation)
            torch.testing.assert_close(model.inputs[0][0],condition)
            for _,time,fixed_condition in model.inputs:
                torch.testing.assert_close(fixed_condition,condition,rtol=0,atol=0)
            torch.testing.assert_close(torch.cat([r[1] for r in model.inputs]),torch.arange(8)/8)

    def test_wavelet_free_path_and_loss_are_in_voxel_units(self):
        source=torch.rand(1,1,16,16,16)
        target=source*.6
        class Oracle(nn.Module):
            def __init__(self,representation):
                super().__init__()
                self.delta=nn.Parameter(encode(target-source,representation))
            def forward(self,state,time,condition):
                return self.delta
        for representation in ('voxel','haar3d'):
            arm=dict(representation=representation,lambda_projection=.1,lambda_reconstruction=1.)
            model=Oracle(representation)
            total,parts=objectives(model,source,target,torch.tensor([.37]),arm,8)
            self.assertLess(total.item(),1e-6)
            torch.testing.assert_close(rollout(model,source,representation,8),target,atol=1e-6,rtol=1e-6)

    def test_projection_axis_and_unclamped_reconstruction_gradients(self):
        source=torch.full((1,1,16,16,16),.1)
        target=torch.full_like(source,.2)
        for representation in ('voxel','haar3d'):
            for value in (-2.,2.):
                arm=dict(representation=representation,lambda_projection=.1,lambda_reconstruction=1.)
                model=RecordingVelocity(value)
                loss,parts=objectives(model,source,target,torch.tensor([.4]),arm,8,True)
                loss.backward()
                self.assertGreater(parts['reconstruction_loss'].item(),.2)
                self.assertTrue(torch.isfinite(model.values.grad).all())
                self.assertTrue(torch.all(model.values.grad.abs()>0))
                self.assertAlmostEqual(loss.item(),parts['fm_loss'].item()+.1*parts['z_projection_loss'].item()+parts['reconstruction_loss'].item(),places=5)

    def test_five_arms_are_only_the_requested_factor_changes(self):
        self.assertEqual(len(ARMS),5)
        self.assertEqual(ARMS['01_fm3d'],('voxel',0.,0.))
        self.assertEqual(ARMS['02_fm3d_wavelet'],('haar3d',0.,0.))
        self.assertEqual(ARMS['03_fm3d_zproj'],('voxel',.1,0.))
        self.assertEqual(ARMS['04_fm3d_wavelet_zproj'],('haar3d',.1,0.))
        self.assertEqual(ARMS['05_fm3d_wavelet_zproj_reconstruction'],('haar3d',.1,1.))
        a,t=schedule(25,101,8)
        np.random.seed(999)
        b,u=schedule(25,101,8)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(t,u)
        for start in range(0,96,8):
            self.assertEqual(set(a[start:start+8]),set(range(8)))

    def test_aggregation_counts_patients_and_nulls_correctly(self):
        records=[dict(split='train',patient_id='A',metrics=dict(pcc=None,mae=1.)),
                 dict(split='train',patient_id='A',metrics=dict(pcc=.4,mae=3.)),
                 dict(split='train',patient_id='B',metrics=dict(pcc=.8,mae=8.))]
        report=aggregate(records)['train']
        self.assertEqual(report['patients'],2)
        self.assertAlmostEqual(report['metrics']['mae'],5.)
        self.assertAlmostEqual(report['metrics']['pcc'],.6)

    def test_real_cli_checkpoint_predictions_and_resume_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('engine.py','run.py','prepare.py'):
                shutil.copy2(ROOT/name,root/name)
            shutil.copytree(ROOT/'source',root/'source')
            config=dict(seed=41,base_channels=4,condition=True,lr=2e-4,weight_decay=0.,grad_clip=1.,
                amp=False,ode_steps=2,checkpoint_rollout=False,eval_every=1,
                stages=[dict(name='one_case',updates=2,seed=42)])
            write_json(root/'suite.json',config)
            selection=dict(one_case_patch='1_y064_x064',patches=[])
            rng=np.random.default_rng(42)
            for number,phase in ((1,'train'),(2,'validation')):
                path=root/f'data/{number}.npz'
                path.parent.mkdir(exist_ok=True)
                source=rng.random((16,16,16),dtype=np.float32)
                layers=np.broadcast_to(np.array([160,200,320,400,480,560],dtype=np.float32)[:,None,None],(6,16,16))
                np.savez(path,oct=source,octa=source*.6,layers=layers,vessel_mask=np.ones((16,16),dtype=np.uint8))
                selection['patches'].append(dict(file=f'data/{number}.npz',sha256=sha256(path),
                    case_id=str(number),patient_id=str(number),split=phase,disease='NORMAL',patch_id='y064_x064'))
            write_json(root/'data/selection.json',selection)
            name=list(ARMS)[-1]
            representation,lp,lr=ARMS[name]
            write_json(root/name/'config.json',dict(name=name,representation=representation,lambda_projection=lp,lambda_reconstruction=lr))
            command=[sys.executable,str(root/'run.py'),'--arm',name,'--stage','one_case','--device','cpu']
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            def run(extra=(),expected=0):
                result=subprocess.run(command+list(extra),capture_output=True,text=True,env=env,timeout=120)
                self.assertEqual(result.returncode,expected,result.stdout+result.stderr)
                return result.stdout+result.stderr
            run()
            checkpoint=root/name/'one_case/latest.pt'
            original=sha256(checkpoint)
            report=json.loads((root/name/'one_case/final_metrics.json').read_text())
            self.assertEqual(report['completed_updates'],2)
            self.assertEqual(len(list((root/name/'one_case/predictions').glob('*.npz'))),2)
            run(['--resume'])
            self.assertEqual(sha256(checkpoint),original)
            self.assertIn('Use --resume',run(expected=1))
            config['lr']=1e-3
            write_json(root/'suite.json',config)
            self.assertIn('Code, data or protocol changed',run(['--resume'],expected=1))


if __name__=='__main__':
    unittest.main(verbosity=2)
