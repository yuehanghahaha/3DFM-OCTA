"""A separate one-variable diagnostic: public-code FM weight 10 vs adapted weight 1."""
from pathlib import Path
import copy
import shutil
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
from flowoct3d.utils import read_json, sha256, write_json

destination = ROOT/'diagnostics/fm_weight10'
if destination.exists():
    raise FileExistsError(destination)
destination.mkdir(parents=True)
for folder in ('source', 'data'):
    shutil.copytree(ROOT/folder, destination/folder, ignore=shutil.ignore_patterns('__pycache__'))
for name in ('engine.py', 'run.py', 'verify.py'):
    shutil.copy2(ROOT/name, destination/name)
config = copy.deepcopy(read_json(ROOT/'protocol.json'))
arm = copy.deepcopy(config['arms']['01_full3d'])
arm['losses']['fm_weight'] = 10.
arm['label'] = 'Full 3D, FM weight 10'
config['arms'] = {'08_full3d_fm_weight10': arm}
config['comparison'] = 'Single-variable weight diagnostic: fm_weight=10 instead of 1; same model/data/schedule/other losses/optimizer/budget as main 01_full3d'
config['control_parent_protocol_sha256'] = sha256(ROOT/'protocol.json')
write_json(destination/'protocol.json', config)
print(destination)

