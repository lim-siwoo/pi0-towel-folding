"""Stage or register the tested native 10k LoRA in the control-center catalog."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import shutil
import time

from lerobot.policies.pi05.configuration_pi05 import PI05Config

PROJECT=Path('/home/dit/pi0_aloha_towel')
WORK=Path(__file__).resolve().parent
SOURCE=PROJECT/'finetune/lerobot_runs/pi05_new31_native_lora/checkpoints/010000'
STAGING=WORK/'center_export'
JOB_ID='20261107-013422-pi05-new31-native'
DEST=Path('/home/dit/control_center/data/jobs')/JOB_ID
MODEL=STAGING/'training/checkpoints/010000/pretrained_model'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--register',action='store_true')
    args=ap.parse_args()
    if not MODEL.exists():
        shutil.copytree(SOURCE,MODEL.parent)
    config=json.loads((SOURCE/'pretrained_model/config.json').read_text())
    supported={f.name for f in fields(PI05Config)}|{'type'}
    removed={k:v for k,v in config.items() if k not in supported}
    allowed={'use_visual_memory':False,'use_proprioceptive_memory':False,'memory_frames':6,
        'memory_stride':30,'memory_temporal_attention_every':4,'rtc_training_max_delay':10,
        'text_tokenizer_name':'google/paligemma-3b-pt-224','optimizer_fused':None}
    assert removed==allowed, 'Review new inference configuration differences before exporting'
    (MODEL/'config.json').write_text(json.dumps({k:v for k,v in config.items() if k in supported},indent=2)+'\n')
    processor=json.loads((SOURCE/'pretrained_model/policy_preprocessor.json').read_text())
    for step in processor['steps']:
        if step.get('registry_name')=='tokenizer_processor':
            step['config']['tokenizer_name']=str(DEST/'training/checkpoints/010000/pretrained_model/tokenizer')
            step.pop('artifacts',None)
    (MODEL/'policy_preprocessor.json').write_text(json.dumps(processor,indent=2)+'\n')
    retained={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file() and p.name not in {'config.json','policy_preprocessor.json'}}
    assert all(sha(STAGING/'training/checkpoints/010000'/name)==digest for name,digest in retained.items())
    manifest=json.loads((WORK/'manifest.json').read_text())
    store=json.loads(Path('/home/dit/control_center/data/center.json').read_text())
    task=next(t for t in store['tasks'] if t['id']=='task-e2b573c17dc9')
    assert task['instruction']=='fold towel'
    spec={'id':JOB_ID,'kind':'train','task':task,'created_at':'2026-11-07 01:34:22',
        'policy':'pi05','policy_name':'π0.5 새 수건 · 추가 10000 (누적 20000)',
        'steps':10000,'cumulative_steps':20000,'batch_size':8,'save_freq':5000,
        'episodes':list(range(31)), 'pretrained':'lerobot/pi05_base','test_small':False,
        'imported':True,'imported_at':time.strftime('%Y-%m-%d %H:%M:%S'),
        'import_source':str(SOURCE),'training_mode':'LoRA only',
        'training_split':{'train_episodes':manifest['train'],'eval_episodes':manifest['val'],'correction_episodes':[]}}
    proof={'source':str(SOURCE),'model_name':spec['policy_name'],'removed_inference_inactive_fields':removed,
           'preserved_files_sha256':retained,'config_source_sha256':sha(SOURCE/'pretrained_model/config.json'),
           'export_config_sha256':sha(MODEL/'config.json'), 'hardware_connected':False}
    proof['tokenizer_path_compatibility']='Resolved checkpoint-relative tokenizer artifact to the registered local checkpoint directory for the older control-center loader.'
    (STAGING/'spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
    (STAGING/'import_manifest.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    if args.register:
        probe=json.loads((STAGING/'offline_probe.json').read_text())
        assert probe['all_adapter_tensors_loaded'] and probe['all_finite'] and not probe['hardware_connected']
        assert 'All keys loaded successfully!' in (WORK/'probe_center.log').read_text(), 'Base-weight loading was not confirmed'
        assert probe['config_sha256']==sha(MODEL/'config.json')
        assert probe['preprocessor_sha256']==sha(MODEL/'policy_preprocessor.json')
        cp=f'jobs/{JOB_ID}/training/checkpoints/010000/pretrained_model'
        status={'phase':'completed','message':'외부 학습 π0.5 LoRA 가져오기 완료 · 비구동 추론 확인',
                'step':10000,'steps':10000,'progress':1,'checkpoint':cp,
                'training_split':spec['training_split'],'imported':True}
        (STAGING/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
        if DEST.exists():
            assert json.loads((DEST/'import_manifest.json').read_text())['preserved_files_sha256']==retained
            print('ALREADY_REGISTERED',DEST)
        else:
            pending=DEST.parent.parent/'import_staging'/JOB_ID
            pending.parent.mkdir(parents=True,exist_ok=True)
            if pending.exists():
                raise RuntimeError('Pending import already exists; inspect it before retrying')
            shutil.copytree(STAGING,pending)
            pending.rename(DEST)
            print('REGISTERED',DEST)
    else:
        print('STAGED',MODEL)


if __name__=='__main__':
    main()
