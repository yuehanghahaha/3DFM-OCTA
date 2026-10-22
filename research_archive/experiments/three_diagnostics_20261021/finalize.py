"""Verify the completed diagnostics and preserve the final server state."""
import json
import subprocess
from pathlib import Path

from run_diagnostics import ROOT, sha256, stamp, write_json


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    previous=json.loads((ROOT/'final_audit.json').read_text())
    sources=protocol['source_hashes']
    production={}
    for name,digest in sources.items():
        parts=Path(name).parts
        project=Path(protocol['checkpoints'][parts[1]]['original']).parents[2]
        original=project/Path(*parts[2:])
        production[str(original)]=sha256(original)==digest
    status={}
    for directory in ('experiment1_supervised','experiment1_same_backbone','experiment2_velocity','experiment3_formulas','posthoc_layers'):
        status[directory]=json.loads((ROOT/directory/'result.json').read_text())['complete']
    previous.update(snapshot_hashes_unchanged=all(sha256(ROOT/'snapshots'/(arm+'.pt'))==r['sha256'] for arm,r in protocol['checkpoints'].items()),
        copied_production_source_hashes_unchanged=all(sha256(ROOT/name)==digest for name,digest in sources.items()),
        original_production_python_sources_unchanged=all(production.values()),
        production_source_checks=production,stages_complete=status,completed=stamp(),
        gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True),
        original_processes=subprocess.check_output(['ps','-p','272550,279725','-o','pid,etime,args'],text=True))
    if not all(status.values()) or not all(production.values()) or not previous['snapshot_hashes_unchanged']:
        raise AssertionError('Final audit failed')
    write_json(ROOT/'final_audit.json',previous)
    write_json(ROOT/'status.json',dict(state='complete',completed=stamp(),diagnostics=3,
        supervised_successful_updates=4800,additional_capacity_control=True,posthoc_layer_views=True))
    print(json.dumps({key:value for key,value in previous.items() if key not in ('production_source_checks','original_processes')},indent=2))


if __name__=='__main__':
    main()
