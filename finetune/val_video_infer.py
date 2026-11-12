"""검증 영상용 추론: 검증 에피소드 하나에서 15프레임(0.5초)마다 모델이 50스텝 행동을 예측한다.
(실제 실행처럼 예측의 앞 15스텝을 쓰고 다시 예측하는 방식. 다만 입력은 시연 당시 관측이다.)
출력: npz (demo 행동 전체, 상태, 예측 시작 프레임, 예측 묶음). 렌더링은 val_video_render.py (MuJoCo venv).
사용 (trossen_openpi 폴더): .venv/bin/python val_video_infer.py CKPT OUT.npz [--episode 2] [--replan 15] [--cpu]
"""
import argparse
import os
import pathlib
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--episode", type=int, default=2)
ap.add_argument("--replan", type=int, default=15)
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()
if args.cpu:
    os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.shared import normalize  # noqa: E402
from openpi.training import config as _config  # noqa: E402

ckpt = pathlib.Path(args.ckpt)
cfg = _config.get_config(os.environ.get("TRAIN_CONFIG", "pi0_trossen_towel_cc"))
norm = normalize.load(sorted((ckpt / "assets").glob("**/norm_stats.json"))[0].parent)
policy = policy_config.create_trained_policy(cfg, ckpt, repack_transforms=cfg.data.repack_transforms,
                                             default_prompt=os.environ.get("TRAIN_PROMPT") or None, norm_stats=norm)
ds = LeRobotDataset("local/trossen_towel_cc_val", delta_timestamps={"action": [t / 30 for t in range(50)]})
a, b = int(ds.episode_data_index["from"][args.episode]), int(ds.episode_data_index["to"][args.episode])
demo = np.stack(ds.hf_dataset.select(range(a, b))["action"]).astype(np.float32)
state = np.stack(ds.hf_dataset.select(range(a, b))["observation.state"]).astype(np.float32)
starts, preds = [], []
t0 = time.time()
for f in range(0, b - a, args.replan):
    s = ds[a + f]
    obs = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in s.items()}
    preds.append(np.asarray(policy.infer(obs)["actions"])[:50, :14]); starts.append(f)
np.savez(args.out, demo=demo, state=state, starts=np.array(starts), preds=np.stack(preds),
         episode=args.episode, replan=args.replan, ckpt=str(ckpt))
print(f"추론 {len(starts)}회, {time.time() - t0:.0f}s → {args.out}")
