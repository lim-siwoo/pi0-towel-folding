"""Run mixed 10k, then warm-start native-only 10k, preserving all four checkpoints."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare import PROJECT, WORK


def status(**fields):
    fields['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    tmp=WORK/'status.tmp'
    tmp.write_text(json.dumps(fields,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(WORK/'status.json')
    print(json.dumps(fields,ensure_ascii=False),flush=True)


def main():
    with (WORK/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (WORK/'pipeline.pid').write_text(str(os.getpid())+'\n')
        env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
        try:
            if not (WORK/'manifest.json').exists():
                status(phase='preparing')
                subprocess.run([sys.executable,str(WORK/'prepare.py')],env=env,check=True)
            for stage in ['mixed','native']:
                run=PROJECT/f'finetune/lerobot_runs/pi05_new31_{stage}_lora'
                final=run/'checkpoints/010000'
                if (final/'pretrained_model/adapter_model.safetensors').exists() and (final/'training_state/training_step.json').exists():
                    status(phase='stage_already_complete',stage=stage)
                    continue
                command=[sys.executable,str(WORK/'train_stage.py'),'--stage',stage]
                if (run/'checkpoints/last/pretrained_model/train_config.json').exists():
                    command += ['--resume']
                with (WORK/f'{stage}.log').open('a') as log:
                    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=PROJECT)
                    status(phase='training',stage=stage,training_pid=child.pid,output_dir=str(run),target_steps=10000,save_every=5000)
                    code=child.wait()
                if code:
                    raise RuntimeError(f'{stage} exited {code}; see {stage}.log')
                for step in ['005000','010000']:
                    path=run/'checkpoints'/step/'pretrained_model'
                    assert (path/'adapter_model.safetensors').is_file() and not (path/'model.safetensors').exists()
                status(phase='stage_complete',stage=stage)
            status(phase='complete',stages=['mixed','native'],checkpoints_each=[5000,10000],total_updates=20000)
        except BaseException as e:
            status(phase='failed',error=repr(e))
            raise


if __name__=='__main__':
    main()
