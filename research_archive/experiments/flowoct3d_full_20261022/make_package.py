"""Package runnable source and frozen small-study inputs; exclude large run outputs."""
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE))
from flowoct3d.utils import sha256, write_json

archive = WORKSPACE/'FlowOCT3D_full_20261022.zip'
paths = [(p, 'flowoct3d/'+p.relative_to(WORKSPACE/'flowoct3d').as_posix())
         for p in (WORKSPACE/'flowoct3d').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
paths += [(WORKSPACE/'fm3d'/name, 'fm3d/'+name)
          for name in ('__init__.py', 'model.py', 'wavelets.py', 'inference.py', 'data.py')]
for folder in ('source', 'data'):
    paths += [(p, 'small_study/'+p.relative_to(ROOT).as_posix()) for p in (ROOT/folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts]
paths += [(ROOT/name, 'small_study/'+name) for name in ('run.py', 'suite.json')]
paths += [(p, 'reference_results/'+p.relative_to(ROOT).as_posix()) for p in (ROOT/'reports').glob('*') if p.is_file()]
paths += [(p, 'reference_results/verification/'+p.name) for p in (ROOT/'verification').glob('*')
          if p.is_file() and p.name != 'package.json']
start = '''# FlowOCT3D 完整实验包\n\n从本目录运行。详细公式、差异与完整数据训练见 flowoct3d/README.md。\n\n```bash\npython -m unittest flowoct3d.test_core flowoct3d.test_cli -v\npython small_study/run.py --benchmark\npython small_study/run.py\n# 需要恢复时：python small_study/run.py --resume\npython -m flowoct3d.verify --root small_study\npython -m flowoct3d.report --root small_study\n```\n\nsmall_study 包含原 16 个固定体块和实际训练源码快照，可以从头复跑 12 个运行。\nreference_results 是已完成的参考报告，不会被当作新训练输出。\n包内不含全量 OCTA-500、不含已有模型权重、不含完整三维预测。\n完整数据训练请指向自己现有的 CSV、NPZ 与 GT_Layers。\n需要继续原本机实验时，在原实验目录使用 run.py --resume，不要在新包内寻找原权重。\n'''
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    bundle.writestr('START_HERE.md', start)
    manifest = {}
    for path, name in sorted(paths, key=lambda r: r[1]):
        bundle.write(path, name)
        manifest[name] = dict(sha256=sha256(path), bytes=path.stat().st_size)
    bundle.writestr('package_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
digest = sha256(archive)
archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n', encoding='ascii')
write_json(ROOT/'verification/package.json', dict(path=str(archive), sha256=digest,
           bytes=archive.stat().st_size, files=len(paths)+2, zip_crc_passed=True))
print(archive, archive.stat().st_size, digest)

