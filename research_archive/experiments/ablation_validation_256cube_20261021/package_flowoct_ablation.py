"""Package small reports and reproducibility records, keeping large volumes on server."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
archive = ROOT / 'results_flowoct_ablation_small.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix not in (
                '.py', '.json', '.jsonl', '.md', '.sh', '.csv', '.png', '.log', '.txt', '.html', '.npz'):
            continue
        if p.suffix == '.npz' and p.name != 'schedule.npz':
            continue
        zipped.write(p, p.relative_to(ROOT).as_posix())
metadata = {'archive': str(archive), 'bytes': archive.stat().st_size,
            'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
(ROOT / 'verification/flowoct_ablation_package.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata), flush=True)
