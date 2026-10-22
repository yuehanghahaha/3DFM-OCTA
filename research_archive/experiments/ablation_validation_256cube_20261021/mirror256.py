"""Snapshot small outputs for local viewing while large files remain on server."""
from pathlib import Path
import zipfile,json,hashlib
root=Path(__file__).resolve().parent
archive=root/'mirror256.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in root.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix not in ('.py','.json','.sh','.md','.log','.csv','.png','.jsonl','.mat','.bmp','.npz'):continue
        if p.suffix=='.npz' and p.name!='schedule.npz':continue
        z.write(p,p.relative_to(root).as_posix())
print(json.dumps({'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
