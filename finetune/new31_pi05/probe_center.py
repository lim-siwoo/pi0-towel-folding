"""Test the control-center's exact native PI05 loader and inference engine, without hardware."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from peft import PeftConfig,PeftModel,get_peft_model_state_dict
from safetensors.torch import load_file
from lerobot.policies import get_policy_class,make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.rollout.inference.sync import SyncInferenceEngine

WORK=Path(__file__).resolve().parent
ROOT=WORK/'center_export'
CK=ROOT/'training/checkpoints/010000/pretrained_model'


def main():
    torch.set_num_threads(8)
    config=PreTrainedConfig.from_pretrained(CK)
    config.device='cuda'
    pre,post=make_pre_post_processors(policy_cfg=config,pretrained_path=CK,
        preprocessor_overrides={'device_processor':{'device':'cuda'},
                               'tokenizer_processor':{'tokenizer_name':str(CK/'tokenizer')}})
    peft=PeftConfig.from_pretrained(CK)
    base=get_policy_class(config.type).from_pretrained(peft.base_model_name_or_path,config=config)
    policy=PeftModel.from_pretrained(base,CK).to('cuda').eval()
    expected=load_file(str(CK/'adapter_model.safetensors'))
    actual=get_peft_model_state_dict(policy)
    assert set(actual)==set(expected), (set(expected)-set(actual),set(actual)-set(expected))
    assert all(torch.equal(v.cpu().float(),expected[k].float()) for k,v in actual.items()), 'Adapter mismatch'
    source=Path('/home/dit/control_center/data/datasets/task-e2b573c17dc9')
    info=json.loads((source/'meta/info.json').read_text())
    ft=info['features']
    names=ft['action']['names']
    assert config.action_feature_names==names and config.output_features['action'].shape==(14,)
    engine=SyncInferenceEngine(policy,pre,post,ft,names,'fold towel','cuda',info['robot_type'])
    with np.load(WORK/'center_probe_observation.npz') as d:
        obs={k:d[k].copy() for k in d.files}
    torch.manual_seed(12345)
    engine.reset()
    actions=[];timing=[]
    for _ in range(3):
        started=time.monotonic()
        action=engine.get_action(obs)
        assert action.shape==(14,) and torch.isfinite(action).all()
        actions.append(action.numpy().tolist());timing.append((time.monotonic()-started)*1000)
    result={'policy':'pi05','task':'fold towel','all_adapter_tensors_loaded':True,
        'adapter_tensors':len(actual),'all_finite':True,'actions':actions,'timing_ms':timing,
        'action_shape':[14],'hardware_connected':False,'actions_sent_to_hardware':0,
        'config_sha256':hashlib.sha256((CK/'config.json').read_bytes()).hexdigest(),
        'preprocessor_sha256':hashlib.sha256((CK/'policy_preprocessor.json').read_bytes()).hexdigest()}
    (ROOT/'offline_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
