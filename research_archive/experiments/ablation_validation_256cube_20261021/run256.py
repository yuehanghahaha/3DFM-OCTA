"""A separate 256-cube same-patient pilot; preserve all earlier experiments."""
import argparse, csv, gc, hashlib, json, os, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from engine import (ARMS, evaluate, load_samples, objectives, save_checkpoint, schedule,
                    seed_all, sha256, write_json)
from fm3d.data import load_pair
from memory_model import MemoryUNet, verify_equivalence
from flowoct_control import make_flowoct, to_flowoct, evaluate_flow


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare(data_root):
    if (ROOT/'data/selection.json').exists():
        return
    old = read(ROOT/'data/reference_selection.json')
    with (data_root/'split_3M_z256.csv').open(encoding='utf-8-sig') as f:
        rows = {r['case_id']:r for r in csv.DictReader(f)}
    # Single NORMAL training case and the same four held-out validation patients.
    selected = [p for p in old['patients'] if p['case_id']=='10301' or p['split']=='validation']
    selection = dict(schema='central_256cube_single_patient_v1',patch_ZYX=[256]*3,
        policy='Central crop in original XY; no lateral resize, augmentation or minmax; full preprocessed Z.',
        one_case_patch=None,patients=selected,patches=[])
    (ROOT/'data/patches').mkdir(parents=True,exist_ok=True)
    for patient in selected:
        row = rows[patient['case_id']]
        path = Path(row['path'])
        path = path if path.is_absolute() else data_root/path
        assert sha256(path)==patient['source_npz_sha256']
        a,b = load_pair(path,'ZYX',255,256)
        a,b = a[0].numpy(),b[0].numpy()
        assert a.shape[0]==256 and min(a.shape[1:])>=256
        y,x = [(n-256)//2 for n in a.shape[1:]]
        labels = ROOT/'data/labels_native'
        layer_path=labels/'GT_Layers'/(row['case_id']+'.mat')
        assert sha256(layer_path)==patient['layers_sha256']
        for kind in ('GT_LargeVessel','GT_Capillary'):
            assert sha256(labels/kind/(row['case_id']+'.bmp'))==patient['masks_sha256'][kind]
        layers = np.asarray(loadmat(layer_path)['Layer'],dtype=np.float32)
        masks = [np.asarray(Image.open(labels/kind/(row['case_id']+'.bmp')))>0
                 for kind in ('GT_LargeVessel','GT_Capillary')]
        mask = (masks[0]|masks[1]).T
        assert layers.shape==(6,*a.shape[1:]) and mask.shape==a.shape[1:]
        patch_id=f'y{y:03d}_x{x:03d}_256'
        output=ROOT/'data/patches'/(row['case_id']+'_'+patch_id+'.npz')
        np.savez_compressed(output,oct=a[:,y:y+256,x:x+256],octa=b[:,y:y+256,x:x+256],
            layers=layers[:,y:y+256,x:x+256],vessel_mask=mask[y:y+256,x:x+256].astype(np.uint8))
        record={k:patient[k] for k in ('case_id','patient_id','split','disease')}
        record.update(patch_id=patch_id,origin_ZYX=[0,y,x],source_shape_ZYX=list(a.shape),
            source_npz_sha256=sha256(path),file=output.relative_to(ROOT).as_posix(),sha256=sha256(output))
        selection['patches'].append(record)
        if row['case_id']=='10301': selection['one_case_patch']=row['case_id']+'_'+patch_id
        print('Prepared',row['case_id'],a.shape,'origin',y,x,flush=True)
    write_json(ROOT/'data/selection.json',selection)


def make(name, device, config):
    seed_all(config['seed'])
    if name=='06_flowoct':
        return make_flowoct(config['flowoct'],device)
    representation,_,_=ARMS[name]
    return MemoryUNet(16,True,8 if representation=='haar3d' else 1).to(device)


def update(model,name,source,target,t,optimizer,scaler):
    if name=='06_flowoct':
        model.set_input(dict(A=to_flowoct(source),B=to_flowoct(target)))
        model.forward()
        return {'loss':float(model.loss.detach()),**model.get_current_loss()},True
    rep,lp,lr=ARMS[name]
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',enabled=True):
        loss,parts=objectives(model,source,target,t,
            dict(representation=rep,lambda_projection=lp,lambda_reconstruction=lr),8,True)
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    previous=scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return {k:float(v.detach()) for k,v in parts.items()},scaler.get_scale()>=previous


def evaluate_model(model,name,samples,destination=None,mode='shared8'):
    if name=='06_flowoct':
        return evaluate_flow(model,samples,mode,torch.device('cuda'),destination)
    rep,lp,lr=ARMS[name]
    return evaluate(model,samples,dict(representation=rep,lambda_projection=lp,lambda_reconstruction=lr),
                    dict(amp=True,ode_steps=8),destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',type=Path)
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--arm',choices=[*ARMS,'06_flowoct'])
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.prepare:
        prepare(a.prepare);return
    config=read(ROOT/'suite.json')
    device=torch.device('cuda')
    selection=read(ROOT/'data/selection.json')
    samples=load_samples(ROOT,selection)
    sample=next(s for s in samples if s['split']=='train')
    source,target=[torch.from_numpy(sample[k])[None,None].to(device) for k in ('oct','octa')]
    model=make(a.arm,device,config)
    optimizer=model.optimizer_G if a.arm=='06_flowoct' else torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.)
    scaler=None if a.arm=='06_flowoct' else torch.amp.GradScaler('cuda',init_scale=1024)
    if a.benchmark:
        torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        records=[]
        try:
            for _ in range(2):
                row,success=update(model,a.arm,source,target,torch.tensor([.37],device=device),optimizer,scaler)
                records.append(dict(losses=row,applied=success))
            torch.cuda.synchronize()
            result=dict(status='passed',seconds_per_attempt=(time.perf_counter()-tick)/2,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        updates=records)
        except torch.cuda.OutOfMemoryError as error:
            result=dict(status='out_of_memory',error=str(error),peak_allocated_bytes=torch.cuda.max_memory_allocated())
        result.update(arm=a.arm,shape_ZYX=list(source.shape[2:]),flowoct=config['flowoct'] if a.arm=='06_flowoct' else None,
                      block_checkpointing=a.arm!='06_flowoct',rollout_checkpointing=a.arm!='06_flowoct')
        write_json(ROOT/'verification'/('benchmark_'+a.arm+'.json'),result)
        print(json.dumps(result),flush=True)
        return
    output=ROOT/a.arm/'one_case'
    output.mkdir(parents=True,exist_ok=True)
    signature=dict(config=config,arm=a.arm,selection_sha256=sha256(ROOT/'data/selection.json'),
        sources={str(x.relative_to(ROOT)):sha256(x) for x in [ROOT/'run256.py',ROOT/'memory_model.py',ROOT/'engine.py']+
                 sorted((ROOT/'source/fm3d').glob('*.py'))+sorted((ROOT/'source/flowoct').glob('*.py'))})
    signature_hash=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    completed,attempts,skipped=0,0,0
    latest=output/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        assert saved['signature_sha256']==signature_hash,'Protocol/source/data changed'
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        if scaler:scaler.load_state_dict(saved['scaler'])
        completed,attempts,skipped=[saved[k] for k in ('completed_updates','attempts','skipped_updates')]
        del saved
        if completed==config['updates'] and (output/'final_metrics.json').exists():return
    write_json(output/'protocol.json',signature)
    _,times=schedule(20261022,config['updates'],1)
    np.savez(output/'schedule.npz',times=times)
    tick=time.perf_counter()
    if not (output/'initial_metrics.json').exists():
        initial=evaluate_model(model,a.arm,samples)
        write_json(output/'initial_metrics.json',initial)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    while completed<config['updates']:
        if a.arm=='06_flowoct':seed_all(20261022*10000+completed+1)
        row,success=update(model,a.arm,source,target,torch.tensor([float(times[completed])],device=device),optimizer,scaler)
        attempts+=1
        if not success:
            skipped+=1
            if skipped>100:raise FloatingPointError('Repeated overflow')
            continue
        completed+=1
        row.update(update=completed,attempt=attempts,time=float(times[completed-1]))
        with (output/'steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if completed%10==0:
            status=dict(status='running',arm=a.arm,updates=completed,target=config['updates'],attempts=attempts,
                        amp_skipped=skipped,seconds=time.perf_counter()-tick,last_loss=row['loss'])
            write_json(output/'status.json',status);print(json.dumps(status),flush=True)
        if completed%config['eval_every']==0 or completed==config['updates']:
            report=evaluate_model(model,a.arm,samples)
            write_json(output/f'metrics_{completed:05d}.json',report)
            save_checkpoint(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict() if scaler else None,
                completed_updates=completed,attempts=attempts,skipped_updates=skipped,signature_sha256=signature_hash,protocol=signature))
            with (output/'history.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(update=completed,summary=report['summary'],seconds=time.perf_counter()-tick))+'\n')
            print('Evaluated',a.arm,completed,report['summary'],flush=True)
    final=evaluate_model(model,a.arm,samples,output/'predictions')
    final.update(arm=a.arm,completed_updates=completed,amp_skipped=skipped,scope='Central256 cube; 1 training patient, 4 validation patients; not whole304 FOV')
    write_json(output/'final_metrics.json',final)
    if a.arm=='06_flowoct':
        write_json(output/'final_native3.json',evaluate_model(model,a.arm,samples,output/'predictions_native3','native3'))
    write_json(output/'status.json',dict(status='complete',updates=completed,attempts=attempts,amp_skipped=skipped,
        seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    try: main()
    except Exception as error:
        write_json(ROOT/'last_error.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise
