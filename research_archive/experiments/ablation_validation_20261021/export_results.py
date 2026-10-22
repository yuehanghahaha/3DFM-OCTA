"""Export reviewable small results; leave full 3D predictions/checkpoints on server."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partial',action='store_true')
    args=p.parse_args()
    artifacts=[]
    roots=[path for path in ROOT.iterdir() if path.is_dir() and path.name[:2].isdigit()]
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.suffix=='.pt' or (path.suffix=='.npz' and path.parent.name=='predictions'):
                artifacts.append(dict(path=str(path),relative_path=path.relative_to(ROOT).as_posix(),
                    bytes=path.stat().st_size,sha256=None if args.partial else sha(path)))
    (ROOT/'server_artifacts.json').write_text(json.dumps(dict(partial=args.partial,
        scope='Full checkpoints and 3D predictions remain at these server paths',artifacts=artifacts),indent=2),encoding='utf-8')
    archive=ROOT/'results_export.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as f:
        for path in sorted(ROOT.rglob('*')):
            relative=path.relative_to(ROOT)
            if not path.is_file() or '__pycache__' in relative.parts or relative.parts[0] in ('data','source'):
                continue
            if path.suffix in ('.json','.jsonl','.csv','.txt','.log','.png','.html','.md') or path.name=='schedule.npz':
                f.write(path,relative.as_posix())
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
                         remote_large_artifacts=len(artifacts)),indent=2))


if __name__=='__main__':
    main()
