"""LeRobot π0.5(+LoRA) 체크포인트 검증 — eval_val.py / val_video_infer.py 와 같은 결과 형식.

  1) 검증 지표 JSON: eval/<exp>_<step>.json (val 8개 + 비교용 학습 8개, 50스텝 예측 vs 시연, 기준선 '그대로 있기')
  2) 검증 영상용 예측: eval/<exp>_<step>_val.npz (검증 에피소드 2, 15프레임마다 예측) → val_video_render.py
데이터: local/trossen_towel_cc_v3 (원본 번호, finetune/lerobot_episodes.json 의 train/val)
사용 (lerobot_pi05 폴더): .venv/bin/python ~/pi0_aloha_towel/finetune/eval_lerobot.py CKPT_DIR/pretrained_model EXP STEP [--device cuda|cpu]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("exp"); ap.add_argument("step")
ap.add_argument("--device", default="cuda")
ap.add_argument("--every", type=int, default=90)
ap.add_argument("--replan", type=int, default=15)
ap.add_argument("--video-episode", type=int, default=2, help="검증 목록에서 몇 번째 (openpi 쪽과 같은 에피소드)")
args = ap.parse_args()

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402

HOME = pathlib.Path.home()
ROOT = HOME / "pi0_aloha_towel/datasets/local/trossen_towel_cc_v3"
EV = HOME / "pi0_aloha_towel/finetune/eval"
EPS = json.loads((HOME / "pi0_aloha_towel/finetune/lerobot_episodes.json").read_text())
H = 50
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP = [6, 13]
STEPS = [0, 9, 24, 49]

sys.path.insert(0, str(HOME / "pi0_aloha_towel/finetune"))

cfg = PreTrainedConfig.from_pretrained(args.ckpt)
cfg.pretrained_path = args.ckpt
cfg.device = args.device
ds_all = LeRobotDataset("local/trossen_towel_cc_v3", root=ROOT, episodes=EPS["val"])
policy = make_policy(cfg, ds_meta=ds_all.meta)
policy.eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=args.ckpt,
                                     preprocessor_overrides={"device_processor": {"device": args.device}})
base = policy.get_base_model() if hasattr(policy, "get_base_model") else policy


@torch.no_grad()
def predict(frame):
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v] if isinstance(v, str) else v) for k, v in frame.items()
             if k.startswith("observation.") or k == "task"}
    batch = pre(batch)
    chunk = base.predict_action_chunk(batch)  # (1, H, D) 정규화된 값
    a = post(chunk)  # 역정규화 (lerobot_eval 과 같은 호출, 마지막 축 기준)
    return a[0, :H, :14].float().cpu().numpy()


def run(episodes):
    ds = LeRobotDataset("local/trossen_towel_cc_v3", root=ROOT, episodes=episodes,
                        delta_timestamps={"action": [t / 30 for t in range(H)]})
    res = {"pred": [], "gt": [], "state": [], "mask": [], "where": []}
    starts, ends = ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"]
    t0 = time.time()
    for k, e in enumerate(episodes):
        a, b = int(starts[e]), int(ends[e])
        # 부분 집합 데이터셋에서는 전역 index 가 아니라 0 부터 이어진 위치를 쓴다
        rel0 = sum(int(ends[x]) - int(starts[x]) for x in episodes[:k])
        for i in range(0, b - a, args.every):
            s = ds[rel0 + i]
            res["pred"].append(predict(s)); res["gt"].append(s["action"].numpy())
            res["state"].append(s["observation.state"].numpy()); res["mask"].append(~s["action_is_pad"].numpy())
            res["where"].append((int(e), i))
    print(f"{len(res['pred'])} 샘플, {time.time() - t0:.0f}s", flush=True)
    return {k: np.asarray(v) for k, v in res.items()}


def metrics(r):
    gt, pred, st, m = r["gt"], r["pred"], r["state"], r["mask"]
    hold = np.repeat(st[:, None, :], H, 1)
    out = {}
    for name, p in (("model", pred), ("hold", hold)):
        err = np.abs(p - gt)
        arm = np.degrees(err[..., ARM]).mean(-1); grip = (err[..., GRIP] * 1000).mean(-1); mm = m.astype(float)
        out[name] = {"arm_deg_mean": float((arm * mm).sum() / mm.sum()), "grip_mm_mean": float((grip * mm).sum() / mm.sum()),
                     "arm_deg_at_step": {str(k + 1): float(arm[m[:, k], k].mean()) for k in STEPS},
                     "grip_mm_at_step": {str(k + 1): float(grip[m[:, k], k].mean()) for k in STEPS}}
    last = m.sum(1) - 1
    cos, big = [], []
    for side in ([0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]):
        d_gt = gt[np.arange(len(gt)), last][:, side] - st[:, side]
        d_pr = pred[np.arange(len(gt)), last][:, side] - st[:, side]
        ok = np.degrees(np.linalg.norm(d_gt, axis=1)) > 5
        c = (d_gt * d_pr).sum(1) / (np.linalg.norm(d_gt, axis=1) * np.linalg.norm(d_pr, axis=1) + 1e-9)
        cos.append(c[ok]); big.append(np.linalg.norm(d_pr[ok], axis=1) / np.linalg.norm(d_gt[ok], axis=1))
    cos = np.concatenate(cos); big = np.concatenate(big)
    out["arm_direction"] = {"moves": int(len(cos)), "cos_median": float(np.median(cos)) if len(cos) else None,
                            "same_direction_ratio": float((cos > 0.7).mean()) if len(cos) else None,
                            "move_size_ratio_median": float(np.median(big)) if len(big) else None}
    dg_gt = (gt[:, -1, GRIP] - st[:, GRIP]) * 1000; dg_pr = (pred[:, -1, GRIP] - st[:, GRIP]) * 1000
    sel = np.abs(dg_gt) > 3
    out["gripper_direction"] = {"events": int(sel.sum()), "same_direction": int((np.sign(dg_gt[sel]) == np.sign(dg_pr[sel])).sum())}
    out["samples"] = int(len(gt))
    return out


val = run(EPS["val"])
train_eps = [EPS["train"][i] for i in np.linspace(0, len(EPS["train"]) - 1, 8).round().astype(int)]
trn = run(train_eps)
result = {"checkpoint": args.ckpt, "every_frames": args.every, "val": metrics(val), "train": metrics(trn)}
EV.mkdir(exist_ok=True)
(EV / f"{args.exp}_{args.step}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps({k: result[k]["model"] for k in ("val", "train")}, indent=1))

# 검증 영상용 예측 (val_video_render.py 가 읽는 형식). 영상 파일은 v2.1 검증 데이터(같은 에피소드)를 쓴다.
ep = EPS["val"][args.video_episode]
ds = LeRobotDataset("local/trossen_towel_cc_v3", root=ROOT, episodes=[ep])
n = ds.num_frames
demo = np.stack([ds[i]["action"].numpy() for i in range(n)]); state = np.stack([ds[i]["observation.state"].numpy() for i in range(n)])
starts, preds = [], []
for f in range(0, n, args.replan):
    preds.append(predict(ds[f])); starts.append(f)
np.savez(EV / f"{args.exp}_{args.step}_val.npz", demo=demo, state=state, starts=np.array(starts), preds=np.stack(preds),
         episode=args.video_episode, replan=args.replan, ckpt=args.ckpt)
print("영상 예측 저장", n, "프레임")
