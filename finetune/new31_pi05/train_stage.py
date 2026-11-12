"""Native + public LoRA experiment on top of the installed LeRobot trainer."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import traceback

from data import Mixture, manifest
from prepare import PROJECT, WORK

RUNS = PROJECT/'finetune/lerobot_runs'
TARGETS = r'(.*\.(language_model|gemma_expert)\..*\.self_attn\.(q|v)_proj|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['mixed','native'],required=True)
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args = ap.parse_args()
    m = manifest()
    if args.check:
        import torch
        ds = Mixture(args.stage)
        indices = [0,len(ds.native)-1]
        if args.stage=='mixed':
            indices.extend([len(ds.native),len(ds)-1])
        frames = [ds[i] for i in indices]
        for frame in frames:
            assert frame['action'].shape==(50,14) and frame['observation.state'].shape==(14,)
            assert frame['action_is_pad'].shape==(50,)
            assert torch.isfinite(frame['action']).all()
            assert all(frame['observation.images.'+c].shape==(3,224,224) for c in ['cam_high','cam_low','cam_left_wrist','cam_right_wrist'])
        batch = torch.utils.data.default_collate(frames)
        assert batch['action'].shape==(len(frames),50,14)
        from lerobot.datasets.sampler import EpisodeAwareSampler
        sampler = EpisodeAwareSampler(ds.meta.episodes['dataset_from_index'],ds.meta.episodes['dataset_to_index'],episode_indices_to_use=ds.episodes)
        assert len(sampler)==len(ds)
        print(json.dumps({'stage':args.stage,'samples':len(ds),'episodes_with_public_repeats':ds.num_episodes,
                          'task_samples':[f['task'] for f in frames], 'train':m['train'],'val':m['val'],
                          'batch_shapes':{k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')}},indent=2),flush=True)
        return
    from lerobot.scripts import lerobot_train as train
    from validation import evaluate
    out = RUNS/f'pi05_new31_{args.stage}_lora'
    base = 'lerobot/pi05_base' if args.stage=='mixed' else str(RUNS/'pi05_new31_mixed_lora/checkpoints/010000/pretrained_model')
    if args.stage=='native' and not Path(base,'adapter_model.safetensors').is_file():
        raise RuntimeError('Stage 1 final LoRA checkpoint is required before stage 2')
    train.make_train_eval_datasets = lambda cfg:(Mixture(args.stage),None)
    optimizer_factory = train.make_optimizer_and_scheduler

    def guarded_optimizer(cfg, policy):
        names = [n for n,p in policy.named_parameters() if p.requires_grad]
        assert names and all('lora_' in n for n in names), 'Refusing non-LoRA training'
        assert any('language_model' in n for n in names) and any('gemma_expert' in n for n in names)
        learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total = sum(p.numel() for p in policy.parameters())
        assert learnable/total < 0.02
        report = {'stage':args.stage,'base_or_adapter':base,'trainable_parameters':learnable,
                  'total_parameters':total,'all_trainable_are_lora':True,'trainable_names':names}
        (WORK/f'{args.stage}_lora_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print('LORA_ONLY_AUDIT',json.dumps({k:v for k,v in report.items() if k!='trainable_names'}),flush=True)
        return optimizer_factory(cfg,policy)

    train.make_optimizer_and_scheduler = guarded_optimizer
    save = train.save_checkpoint

    def save_and_validate(**kwargs):
        save(**kwargs)
        checkpoint = Path(kwargs['checkpoint_dir'])
        adapter = checkpoint/'pretrained_model/adapter_model.safetensors'
        assert adapter.is_file() and not (checkpoint/'pretrained_model/model.safetensors').exists()
        try:
            policy = kwargs['accelerator'].unwrap_model(kwargs['policy'])
            evaluate(policy,kwargs['preprocessor'],kwargs['postprocessor'],args.stage,kwargs['step'],checkpoint)
        except Exception:
            error = traceback.format_exc()
            (WORK/f'{args.stage}_{kwargs["step"]}_validation_error.txt').write_text(error)
            logging.exception('Held-out validation failed; checkpoint retained and training continues')

    train.save_checkpoint = save_and_validate
    if args.resume:
        config_path = out/'checkpoints/last/pretrained_model/train_config.json'
        sys.argv = [sys.argv[0],f'--config_path={config_path}','--resume=true']
    else:
        lr = 2.5e-4 if args.stage=='mixed' else 1e-4
        sys.argv = [sys.argv[0],f'--dataset.repo_id={m["native_repo"]}',f'--dataset.root={m["native_root"]}',
            '--policy.type=pi05',f'--policy.pretrained_path={base}', '--peft.method_type=LORA',
            f'--peft.target_modules={TARGETS}','--peft.full_training_modules=[]','--peft.r=16','--peft.lora_alpha=16',
            f'--policy.optimizer_lr={lr}',f'--policy.scheduler_decay_lr={lr/10}',
            '--policy.scheduler_warmup_steps=500','--policy.scheduler_decay_steps=10000',
            '--policy.rtc_training_max_delay=10','--policy.gradient_checkpointing=true',
            '--policy.dtype=bfloat16','--policy.device=cuda','--policy.push_to_hub=false',
            f'--output_dir={out}',f'--job_name=pi05_new31_{args.stage}_lora','--batch_size=8','--num_workers=8',
            '--steps=10000','--save_freq=5000','--log_freq=100','--seed=1000']
        if args.stage=='native':
            sys.argv += ['--policy.use_peft=true']
    print('TRAIN_COMMAND',json.dumps(sys.argv),flush=True)
    train.main()


if __name__=='__main__':
    main()
