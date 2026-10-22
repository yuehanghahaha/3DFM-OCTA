"""Remove only the explicitly superseded baseline weights and duplicate run files."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/docsystem/niuke/lvyuehang/fm3d_ubuntu')
OBSOLETE = ROOT / 'runs/fm3d_3M_run01_superseded_val5_20261021'
AUDIT = ROOT / 'verification/cleanup_superseded_20261021'
WEIGHTS = ['latest.pt', 'best.pt', 'best_window_0101_0110.pt',
           'best_window_0111_0120.pt', 'best_window_0121_0130.pt']
OLD_LOG = 'fm3d_3M_run01_resume_101_200_20261021.log'
PRESERVE = [ROOT / 'verification/pre_resume_20261021/latest.pt',
            ROOT / 'verification/pre_resume_20261021/best.pt']
ACTIVE_RUNS = [ROOT / 'runs/fm3d_3M_run01',
               Path('/docsystem/niuke/lvyuehang/fm3d_zproj_96_200/runs/fm3d_zproj_3M_96_200')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_snapshot():
    records = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            command = (folder / 'cmdline').read_bytes().split(b'\0')
            if not any(module in command for module in (b'fm3d.train', b'fm3d.train_projection')):
                continue
            assert not any(str(OBSOLETE).encode() in argument for argument in command), 'Obsolete run is active'
            records.append({'pid': int(folder.name), 'command': [part.decode() for part in command if part]})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    assert all(any(str(run) in item['command'] for item in records) for run in ACTIVE_RUNS), 'Expected training process missing'
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    assert ROOT.resolve() == ROOT and OBSOLETE.resolve() == OBSOLETE
    assert OBSOLETE.parent == ROOT / 'runs' and OBSOLETE.is_dir() and not OBSOLETE.is_symlink()
    reason = json.loads((OBSOLETE / 'superseded_reason.json').read_text())
    assert reason['last_completed_epoch'] == 133
    assert reason['replacement_run'] == str(ACTIVE_RUNS[0])
    before = process_snapshot()
    preserved = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in PRESERVE]
    for run in ACTIVE_RUNS:
        assert (run / 'latest.pt').is_file() and (run / 'config.json').is_file()
    assert sha(OBSOLETE / OLD_LOG) == sha(ROOT / 'logs' / OLD_LOG), 'Old log is not a duplicate'
    files = [OBSOLETE / name for name in WEIGHTS + ['resume_val5.sh', OLD_LOG]]
    assert {p.name for p in OBSOLETE.glob('*.pt')} == set(WEIGHTS), 'Unexpected obsolete weight inventory'
    deletion = []
    for path in files:
        assert path.parent == OBSOLETE and path.resolve().parent == OBSOLETE
        assert path.is_file() and not path.is_symlink()
        stat = path.stat()
        deletion.append(dict(path=str(path), bytes=stat.st_size, sha256=sha(path), hard_links=stat.st_nlink))
    report = dict(status='planned', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  obsolete_run=str(OBSOLETE), deleted_files=deletion,
                  logical_bytes=sum(item['bytes'] for item in deletion),
                  preserved_checkpoints=preserved, training_processes_before=before,
                  retained_duplicate_log=str(ROOT / 'logs' / OLD_LOG))
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / 'plan.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.apply:
        # Every absolute file target was resolved and checked against the single
        # explicitly superseded directory before any unlink occurs.
        for item in deletion:
            path = Path(item['path'])
            assert path.resolve().parent == OBSOLETE and sha(path) == item['sha256']
            path.unlink()
        for item in preserved:
            assert sha(Path(item['path'])) == item['sha256']
        assert not list(OBSOLETE.glob('*.pt'))
        report.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      training_processes_after=process_snapshot(),
                      retained_files=sorted(path.name for path in OBSOLETE.iterdir() if path.is_file()))
        (AUDIT / 'cleanup.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (OBSOLETE / 'CLEANED.md').write_text(
            '# Superseded run cleaned on 2026-10-21\n\n'
            'The five obsolete checkpoints, old launcher, and duplicate log were deleted at the user request.\n'
            'Configuration, history, and validation metrics remain for provenance only.\n'
            'Any checkpoint filenames in these historical JSON records refer to deleted files.\n'
            'The original epoch-100 recovery point and epoch-90 best are retained under verification/pre_resume_20261021/.\n'
            f'Deletion inventory: {AUDIT / "cleanup.json"}\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'files': len(deletion),
                      'weights': len(WEIGHTS), 'bytes': report['logical_bytes'],
                      'MiB': report['logical_bytes'] / 1024 ** 2,
                      'training_pids': [item['pid'] for item in before],
                      'audit_directory': str(AUDIT)}, indent=2))


if __name__ == '__main__':
    main()
