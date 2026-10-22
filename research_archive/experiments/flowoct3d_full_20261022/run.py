"""Run the frozen experiment implementation, independent of later workspace edits."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from flowoct3d.study import main
if __name__ == '__main__':
    if '--root' not in sys.argv:
        sys.argv.extend(['--root', str(ROOT)])
    main()
