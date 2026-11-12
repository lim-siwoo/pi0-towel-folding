"""Offline held-out scoring of control-center pi0.5 checkpoints on the towel dataset.

Usage: eval_cc_job.py NAME CHECKPOINT --episodes 0,1,... [--episodes-train 8,9,...]
Same metric as new31_pi05/validation.py (50-step chunk vs demo, every 90 frames, hold baseline).
Feeds full 480x640 camera frames, exactly what the control center gives the policy live.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/dit/control_center')
sys.path.insert(0, str(Path(__file__).resolve().parent / 'new31_pi05'))
from native_policy_runtime import load_policy  # noqa: E402
from validation import metrics  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402

ROOT = '/home/dit/control_center/data/datasets/task-e2b573c17dc9'
CAMS = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']

ap = argparse.ArgumentParser()
ap.add_argument('name'); ap.add_argument('checkpoint')
ap.add_argument('--episodes', required=True)
ap.add_argument('--episodes-train', default='')
args = ap.parse_args()
parse = lambda s: [int(x) for x in s.split(',') if x]

_, policy, pre, post = load_policy(args.checkpoint)
base = policy.get_base_model() if hasattr(policy, 'get_base_model') else policy
result = {'name': args.name, 'checkpoint': args.checkpoint, 'every_frames': 90, 'dataset': ROOT}
started = time.time()
with torch.no_grad():
    torch.manual_seed(12345)
    for split, eps in [('val', parse(args.episodes)), ('train', parse(args.episodes_train))]:
        if not eps:
            continue
        pred, gt, state, mask = [], [], [], []
        for ep in eps:
            ds = LeRobotDataset('local/towel_cc_new', root=ROOT, episodes=[ep],
                                delta_timestamps={'action': [i / 30 for i in range(50)]}, video_backend='pyav')
            for i in range(0, len(ds) - 49, 90):
                frame = ds[i]
                batch = {k: v.unsqueeze(0) for k, v in frame.items()
                         if k == 'observation.state' or k.startswith('observation.images.')}
                batch['task'] = ['fold towel']
                p = post(base.predict_action_chunk(pre(batch)))[0, :50, :14].float().cpu().numpy()
                assert np.isfinite(p).all()
                pred.append(p); gt.append(frame['action'].numpy())
                state.append(frame['observation.state'].numpy()); mask.append(~frame['action_is_pad'].numpy())
        result[split] = metrics(pred, gt, state, mask)
        result[split]['episodes'] = eps
v = result['val']
result['arm_improvement_vs_hold_percent'] = 100 * (1 - v['model']['arm_deg_mean'] / v['hold']['arm_deg_mean'])
result['seconds'] = time.time() - started
out = Path(__file__).resolve().parent / 'eval' / f'{args.name}.json'
out.write_text(json.dumps(result, indent=2) + '\n')
print('RESULT', out, json.dumps({k: result[k] for k in ('arm_improvement_vs_hold_percent', 'seconds')}))
