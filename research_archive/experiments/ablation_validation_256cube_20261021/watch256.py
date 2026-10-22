"""Read-only compact progress stream, exits after the worker finishes."""
from pathlib import Path
import json,subprocess,time
r=Path(__file__).resolve().parent
while True:
    rows=[]
    for p in sorted(r.glob('*/one_case/status.json')):
        row=json.loads(p.read_text())
        steps=p.parent/'steps.jsonl'
        if steps.exists():
            lines=steps.read_text().splitlines()
            if lines:
                try:row['updates']=json.loads(lines[-1])['update']
                except json.JSONDecodeError:pass
        rows.append(dict(arm=p.parent.parent.name,status=row['status'],updates=row['updates'],seconds=row.get('seconds')))
    print(json.dumps(dict(progress=rows)),flush=True)
    if subprocess.run(['tmux','has-session','-t','lyh_256cube_train_20261021'],capture_output=True).returncode:
        print('WORKER_EXITED',flush=True)
        p=r/'logs/export.log'
        if p.exists():print(p.read_text()[-2500:],flush=True)
        p=r/'last_error.json'
        if p.exists():print(p.read_text(),flush=True)
        break
    time.sleep(40)
