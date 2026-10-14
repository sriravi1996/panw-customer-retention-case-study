"""Generate both preregistered independent populations without refitting the model."""
from pathlib import Path
import os, shutil, subprocess, sys, json, hashlib
import yaml
root=Path(__file__).resolve().parents[2]
base=root/'output/model_demo_revision/replications'
for seed in (991,2027):
    dst=base/f'seed_{seed}'
    dst.mkdir(parents=True,exist_ok=True)
    for name in ('src','config'):
        shutil.copytree(root/name,dst/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(root/'part2/src',dst/'part2/src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    cfg=yaml.safe_load((root/'config/simulation.yaml').read_text());cfg['seed']=seed
    (dst/'config/simulation.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (dst/'pipeline.log').open('w') as log:
        for script in ('src/generate_data.py','src/ingest_clean.py','src/integrate.py','src/eda.py','src/validate.py','part2/src/build_features.py','part2/src/make_splits.py'):
            print(f'Seed {seed}: {script}',flush=True)
            log.write('\nSTAGE '+script+'\n');log.flush()
            subprocess.run([sys.executable,script],cwd=dst,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Seed {seed} complete. No model fitted in replication.',flush=True)
