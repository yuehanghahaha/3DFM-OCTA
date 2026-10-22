"""One-time result synchronizer for the two new arms; exits after final verification."""
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import hashlib
import json
import os
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
HOST = 'eias-member.eitech.edu.cn'
REMOTE = '/docsystem/niuke/lvyuehang/fm3d_validation_256cube_20261021'
PYTHON = '/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python'
ARCHIVE = ROOT.parent.parent / 'artifacts/validation_flowoct_ablation_results.zip'
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
PROBE = '''from pathlib import Path
import json,subprocess
r=Path(REMOTE_PATH)
rows=[]
for name in ('07_flowoct_fm_only','08_flowoct_wavelet_only'):
    p=r/name/'one_case'
    row={'arm':name,'status':'queued','updates':0,'target':400}
    if (p/'status.json').exists():row.update(json.loads((p/'status.json').read_text()))
    if (p/'steps.jsonl').exists():
        lines=(p/'steps.jsonl').read_text().splitlines()
        if lines:
            try:
                last=json.loads(lines[-1]);row.update(updates=last['update'],last_loss=last['loss'])
            except json.JSONDecodeError:pass
    rows.append(row)
metadata=r/'verification/flowoct_ablation_package.json'
audit=r/'verification/flowoct_ablation_audit.json'
worker=subprocess.run(['tmux','has-session','-t','lyh_flowoct_ablation256_20261022'],capture_output=True).returncode==0
print(json.dumps({'arms':rows,'worker_alive':worker,
 'package':json.loads(metadata.read_text()) if metadata.exists() else None,
 'audit':json.loads(audit.read_text()) if audit.exists() else None}))
'''.replace('REMOTE_PATH', repr(REMOTE))


def call(args, source=None, timeout=35):
    result = subprocess.run(args, input=source, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=timeout, creationflags=FLAGS)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-2000:])
    return result.stdout.strip()


def save_status(record):
    record['utc'] = datetime.now(timezone.utc).isoformat()
    destination = ROOT / 'flowoct_ablation_sync_status.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(destination)
    print(json.dumps(record, ensure_ascii=False), flush=True)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.missing, self.count = [], 0

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key not in ('src', 'href') or not value or value.startswith('#'):
                continue
            parsed = urlsplit(value)
            if parsed.scheme or parsed.netloc:
                continue
            self.count += 1
            if not (ROOT / 'reports' / unquote(parsed.path)).exists():
                self.missing.append(value)


def main():
    failures = 0
    while True:
        try:
            remote = json.loads(call(['ssh', '-4', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                      HOST, PYTHON, '-'], PROBE))
            failures = 0
            if remote['audit'] and remote['package']:
                assert remote['audit']['status'] == 'passed'
                assert remote['audit']['new_predictions_checked'] == 20
                metadata = remote['package']
                ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
                call(['scp', '-4', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                      HOST + ':' + metadata['archive'], str(ARCHIVE)], timeout=180)
                assert ARCHIVE.stat().st_size == metadata['bytes']
                assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == metadata['sha256']
                with zipfile.ZipFile(ARCHIVE) as zipped:
                    assert all((ROOT / name).resolve().is_relative_to(ROOT.resolve()) for name in zipped.namelist())
                    for item in zipped.infolist():
                        if item.filename not in ('sync_flowoct_ablation.py', 'flowoct_ablation_sync_status.json'):
                            zipped.extract(item, ROOT)
                (ROOT / 'verification/flowoct_ablation_package.json').write_text(
                    json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
                parser = Links()
                parser.feed((ROOT / 'reports/index.html').read_text(encoding='utf-8'))
                assert not parser.missing, parser.missing
                from PIL import Image
                figures = list((ROOT / 'reports').glob('flowoct_ablations_*.png'))
                assert len(figures) == 12
                for path in figures:
                    assert Image.open(path).size == (1096, 342), path.name
                result = {'status': 'passed', 'archive_sha256': metadata['sha256'],
                          'html_links_checked': parser.count, 'missing_links': [],
                          'focused_native256_figures': len(figures), 'server_audit': remote['audit']}
                (ROOT / 'verification/local_flowoct_ablation_audit.json').write_text(
                    json.dumps(result, indent=2) + '\n', encoding='utf-8')
                save_status({'state': 'complete', 'remote': remote, 'local_audit': result})
                return
            if not remote['worker_alive']:
                save_status({'state': 'worker_exited_before_final_package', 'remote': remote})
                return
            save_status({'state': 'running', 'remote': remote})
        except Exception as error:
            failures += 1
            save_status({'state': 'sync_retry', 'consecutive_failures': failures, 'error': str(error)})
        time.sleep(min(60, 40 + 10 * failures))


if __name__ == '__main__':
    main()
