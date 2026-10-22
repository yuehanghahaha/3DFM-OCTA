"""Continue each original 400-step model to 1600, preserving the old files."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import numpy as np
import torch
from engine import (load_samples,make_model,objectives,rollout,save_checkpoint,seed_all,sha256,write_json)
from run import stage_samples
from flowoct_control import (evaluate_flow,evaluate_predictor,make_flowoct,read,to_flowoct)
from diagnose_comparison import predict_control,probe_3d,probe_flow,tensors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['uniform3d','flowoct'],required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    device=torch.device('cuda')
    suite,selection=read(ROOT/'suite.json'),read(ROOT/'data/selection.json')
    samples=load_samples(ROOT,selection)
    evaluation,training=stage_samples(samples,'one_case',selection)
    sample=training[0]
    source,target=tensors(sample,device)
    folder='uniform_fm_1600' if a.kind=='uniform3d' else 'flowoct_1600'
    output=ROOT/'diagnostics'/folder
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True)
    original=ROOT/('01_fm3d' if a.kind=='uniform3d' else '06_flowoct')/'one_case/latest.pt'
    saved=torch.load(original,map_location='cpu',weights_only=False)
    assert saved['completed_updates']==400
    seed_all(suite['seed'])
    if a.kind=='uniform3d':
        model=make_model(16,True,'voxel').to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=suite['lr'],weight_decay=0)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        with np.load(ROOT/'01_fm3d/one_case/schedule.npz',allow_pickle=False) as f:
            old_times=f['times'].copy()
        rng=np.random.default_rng(2026102241)
        times=np.concatenate([old_times,rng.random(1200).astype(np.float32)])
        np.savez(output/'schedule.npz',times=times)
        def evaluate(destination=None):
            previous=model.training
            model.eval()
            try: return evaluate_predictor(lambda x:predict_control(model,x,device,'uniform3d',True),evaluation,destination)
            finally: model.train(previous)
    else:
        model=make_flowoct(read(ROOT/'06_flowoct/config.json'),device)
        model.load_state_dict(saved['model'])
        optimizer=model.optimizer_G
        optimizer.load_state_dict(saved['optimizer'])
        scaler=None
        model.set_input(dict(A=to_flowoct(source),B=to_flowoct(target)))
        def evaluate(destination=None): return evaluate_flow(model,evaluation,'shared8',device,destination)
    del saved
    config=dict(kind=a.kind,original_checkpoint=str(original),original_sha256=sha256(original),
        start_updates=400,target_updates=1600,new_updates=1200,case_id=sample['case_id'],patch_id=sample['patch_id'],
        selection_sha256=sha256(ROOT/'data/selection.json'),source_sha256=sha256(Path(__file__)),
        policy='Restore original model/optimizer/scaler at400. Same fixed training block; no changes to original checkpoint or loss.')
    write_json(output/'config.json',config)
    write_json(output/'metrics_00400.json',evaluate())
    model.train()
    completed,skipped=400,0
    tick=time.perf_counter()
    while completed<1600:
        if a.kind=='uniform3d':
            optimizer.zero_grad(set_to_none=True)
            t=torch.tensor([float(times[completed])],device=device)
            with torch.autocast(device_type='cuda',enabled=True):
                loss,_=objectives(model,source,target,t,dict(representation='voxel',lambda_projection=0.,lambda_reconstruction=0.),8)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            old_scale=scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale()<old_scale:
                skipped+=1
                if skipped>100: raise FloatingPointError('Repeated overflow')
                continue
            row=dict(loss=float(loss.detach()),time=float(times[completed]))
        else:
            update_seed=20261022*10000+completed+1
            seed_all(update_seed)
            model.forward()
            row=dict(model.get_current_loss())
            row.update(loss=float(model.loss.detach()),seed=update_seed)
        completed+=1
        row['update']=completed
        with (output/'steps.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        if completed%100==0:
            write_json(output/'status.json',dict(status='running',updates=completed,target=1600,seconds=time.perf_counter()-tick))
            print(folder,completed,1600,row['loss'],flush=True)
        if completed%400==0:
            report=evaluate()
            write_json(output/f'metrics_{completed:05d}.json',report)
            with (output/'history.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(update=completed,summary=report['summary'],seconds=time.perf_counter()-tick))+'\n')
            save_checkpoint(output/'latest.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict() if scaler else None,completed_updates=completed,config=config))
            print('Long control evaluation',folder,completed,report['summary'],flush=True)
    final=evaluate(output/'predictions')
    final.update(kind=a.kind,completed_updates=1600,added_updates=1200,amp_skipped=skipped,config=config)
    write_json(output/'final_metrics.json',final)
    model.eval()
    probes=(probe_3d(model,source,target,sample,'voxel',True) if a.kind=='uniform3d'
            else probe_flow(model,source,target,sample))
    write_json(output/'teacher_probe.json',dict(model=folder,stage='one_case_1600',case_id=sample['case_id'],split='train',probes=probes))
    if a.kind=='flowoct':
        native=evaluate_flow(model,evaluation,'native3',device,output/'predictions_native3')
        native.update(kind='flowoct_native3',completed_updates=1600,added_updates=1200)
        write_json(output/'final_native3.json',native)
    write_json(output/'status.json',dict(status='complete',updates=1600,new_updates=1200,amp_skipped=skipped))


if __name__=='__main__': main()
