"""Record the running process and frozen experiment settings, without reading credentials."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


root = Path('/docsystem/niuke/lvyuehang/fm3d_zproj_96_200')
run = root / 'runs/fm3d_zproj_3M_96_200'
config = json.loads((run / 'config.json').read_text())
baseline = json.loads(Path('/docsystem/niuke/lvyuehang/fm3d_ubuntu/runs/fm3d_3M_run01/config.json').read_text())
assert config['manifest_sha256'] == baseline['manifest_sha256']
assert config['epochs'] == 200 and config['patch'] == [256, 96, 96]
assert config['lambda_projection'] == 0.1 and not config['resume']
processes = []
for folder in Path('/proc').iterdir():
    if not folder.name.isdigit():
        continue
    try:
        command = (folder / 'cmdline').read_bytes().split(b'\0')
        if b'fm3d.train_projection' not in command or str(run).encode() not in command:
            continue
        # Keep only the GPU visibility setting; never serialize the environment.
        visibility = [entry.split(b'=', 1)[1].decode() for entry in (folder / 'environ').read_bytes().split(b'\0')
                      if entry.startswith(b'CUDA_VISIBLE_DEVICES=')]
        processes.append(dict(pid=int(folder.name), physical_gpu=visibility))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
assert len(processes) == 1 and processes[0]['physical_gpu'] == ['1'], processes
log = (root / 'logs/train_3M_200.log').read_text()
assert 'epoch=1 step=' in log and 'Traceback' not in log and 'TRAIN_EXIT_CODE=' not in log
now = datetime.now(timezone.utc)
report = dict(status='running', checked_at_utc=now.isoformat(),
              checked_at_shanghai=now.astimezone(timezone(timedelta(hours=8))).isoformat(),
              host='eias-member.eitech.edu.cn', session='lyh_fm3d_zproj96_200', processes=processes,
              run_directory=str(run), log=str(root / 'logs/train_3M_200.log'), config=config,
              data_manifest_identical_to_baseline=True, initial_log_tail=log.splitlines()[-8:])
(root / 'verification/launch_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
