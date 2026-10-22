"""Read-only structured status for the one-time local result synchronizer."""
from pathlib import Path
import json,subprocess
r=Path(__file__).resolve().parent
done=[];active=[]
for p in sorted(r.glob('*/one_case/status.json')):
    x=json.loads(p.read_text());arm=p.parent.parent.name
    if x['status']=='complete':done.append(arm)
    else:
        lines=(p.parent/'steps.jsonl').read_text().splitlines()
        if lines:x['updates']=json.loads(lines[-1])['update']
        active.append(dict(arm=arm,updates=x['updates'],target=400))
audit=r/'verification/final_audit.json'
error=r/'last_error.json'
print(json.dumps(dict(completed=done,active=active,
    worker_alive=subprocess.run(['tmux','has-session','-t','lyh_256cube_train_20261021'],capture_output=True).returncode==0,
    audit=json.loads(audit.read_text()) if audit.exists() else None,
    error=json.loads(error.read_text()) if error.exists() else None)))
