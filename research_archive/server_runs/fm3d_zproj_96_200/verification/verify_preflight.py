"""Check deployed source and all preflight evidence before allowing the long run."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


root = Path('/docsystem/niuke/lvyuehang/fm3d_zproj_96_200')
manifest = json.loads((root / 'source_manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
assert hashlib.sha256((root / 'fm3d_zproj96_source.zip').read_bytes()).hexdigest() == manifest['archive_sha256']
tests = (root / 'logs/server_unit_tests.log').read_text()
repair = (root / 'logs/server_repaired_test.log').read_text()
# The first deployment omitted the shared-data helper used by an existing core
# test. Preserve that failed log; check the specific failure and its rerun.
assert 'Ran 22 tests' in tests
assert 'FAILED (errors=1, skipped=1)' in tests
assert "ModuleNotFoundError: No module named 'tools'" in tests
assert 'test_duplicate_group_repair_preserves_counts' in repair and repair.rstrip().endswith('OK')
bench = json.loads((root / 'verification/server_3090_benchmark.json').read_text())
smoke = json.loads((root / 'verification/real_data_smoke.json').read_text())
assert bench['patch_ZYX'] == [256, 96, 96] and bench['lambda_projection'] == 0.1
assert bench['optimizer_updates_including_warmup'] == 4 and bench['amp_skipped_updates'] == 0
assert smoke['patch_ZYX'] == [256, 96, 96] and smoke['split'] == 'train'
assert len(smoke['steps']) == 3 and all(not step['amp_skipped_update'] for step in smoke['steps'])
report = dict(passed=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
              source_files_verified=len(manifest['files']), archive_sha256=manifest['archive_sha256'],
              server_tests='21 passed, 1 optional PyWavelets check skipped; shared-data test passed after helper was included',
              deployment_repair='Added unchanged tools/prepare_common_octa500.py, required by existing duplicate-group test',
              synthetic_optimizer_updates=4, real_training_updates=3, physical_gpu=1,
              benchmark_peak_allocated_bytes=bench['peak_allocated_bytes'],
              real_peak_allocated_bytes=smoke['peak_allocated_bytes'])
(root / 'verification/preflight_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
(root / 'verification/preflight_passed.txt').write_text(report['checked_at_utc'] + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
