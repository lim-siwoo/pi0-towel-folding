"""체크포인트 검증: 학습에 쓰지 않은 검증 에피소드에서 모델이 예측한 50스텝 행동을 실제 시연과 비교한다.

비교 대상
  - 모델 예측 (policy.infer, 실제 서빙과 같은 전처리·정규화·후처리)
  - 기준선 "그대로 있기": 지금 관절값을 50스텝 내내 유지 (이것보다 확실히 나아야 학습된 것)
  - 같은 수의 학습 에피소드 샘플 (학습/검증 차이 = 외우기만 했는지)
지표 (에피소드 끝 뒤 패딩 제외)
  - 팔 관절 평균 절대 오차(도), 그리퍼 평균 절대 오차(mm) — 예측 1/10/25/50 스텝째와 50스텝 평균
  - 그리퍼 열림/닫힘 판단: 50스텝 안에서 시연이 닫히는지(열림→닫힘)·열리는지 vs 예측이 같은 방향인지
출력: finetune/eval/<체크포인트 이름>.json, 예측 대 시연 궤적 그림 .png
사용 (trossen_openpi 폴더):
  HF_LEROBOT_HOME=~/pi0_aloha_towel/datasets .venv/bin/python ~/pi0_aloha_towel/finetune/eval_val.py CKPT_DIR [--every 60] [--cpu]
"""
import argparse
import os
import json
import pathlib
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--every", type=int, default=60, help="몇 프레임마다 샘플 (30fps)")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--train-eps", type=int, default=8, help="비교용 학습 에피소드 수")
args = ap.parse_args()
if args.cpu:
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.shared import normalize  # noqa: E402
from openpi.training import config as _config  # noqa: E402

H = 50
OUT = pathlib.Path.home() / "pi0_aloha_towel/finetune/eval"
OUT.mkdir(exist_ok=True)
ckpt = pathlib.Path(args.ckpt)
cfg = _config.get_config(os.environ.get("TRAIN_CONFIG", "pi0_trossen_towel_cc"))
assets = sorted((ckpt / "assets").glob("**/norm_stats.json"))
norm = normalize.load(assets[0].parent)
policy = policy_config.create_trained_policy(cfg, ckpt, repack_transforms=cfg.data.repack_transforms,
                                             default_prompt=os.environ.get("TRAIN_PROMPT") or None, norm_stats=norm)
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP = [6, 13]
STEPS = [0, 9, 24, 49]


def run(repo, eps):
    ds = LeRobotDataset(repo, delta_timestamps={"action": [t / 30 for t in range(H)]})
    starts = ds.episode_data_index["from"].numpy(); ends = ds.episode_data_index["to"].numpy()
    res = {"model": [], "hold": [], "mask": [], "gt": [], "pred": [], "state": [], "where": []}
    t0 = time.time()
    for e in eps:
        for i in range(int(starts[e]), int(ends[e]), args.every):
            s = ds[i]
            obs = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in s.items()}
            pred = np.asarray(policy.infer(obs)["actions"])[:H, :14]
            gt = obs["action"]; state = obs["observation.state"]
            mask = ~obs["action_is_pad"]
            res["pred"].append(pred); res["gt"].append(gt); res["state"].append(state); res["mask"].append(mask)
            res["where"].append((int(e), int(i - starts[e])))
    print(f"{repo}: {len(res['pred'])} 샘플, {time.time() - t0:.0f}s", flush=True)
    return {k: np.asarray(v) for k, v in res.items()}


def metrics(r):
    gt, pred, st, m = r["gt"], r["pred"], r["state"], r["mask"]
    hold = np.repeat(st[:, None, :], H, 1)
    out = {}
    for name, p in (("model", pred), ("hold", hold)):
        err = np.abs(p - gt)
        arm = np.degrees(err[..., ARM]).mean(-1)          # (N, H) 도
        grip = (err[..., GRIP] * 1000).mean(-1)           # (N, H) mm
        mm = m.astype(float)
        out[name] = {
            "arm_deg_mean": float((arm * mm).sum() / mm.sum()),
            "grip_mm_mean": float((grip * mm).sum() / mm.sum()),
            "arm_deg_at_step": {str(k + 1): float(arm[m[:, k], k].mean()) for k in STEPS},
            "grip_mm_at_step": {str(k + 1): float(grip[m[:, k], k].mean()) for k in STEPS},
        }
    # 팔 이동 방향: 50스텝(패딩 전 마지막) 뒤 관절 변화 벡터(팔별 6관절)의 코사인 유사도, 시연이 5도 넘게 움직인 경우만
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
    # 그리퍼 방향: 50스텝 안 시연 변화가 3mm 넘는 샘플만
    dg_gt = (gt[:, -1, GRIP] - st[:, None, GRIP][:, 0]) * 1000
    dg_pr = (pred[:, -1, GRIP] - st[:, None, GRIP][:, 0]) * 1000
    sel = np.abs(dg_gt) > 3
    out["gripper_direction"] = {"events": int(sel.sum()),
                                "same_direction": int((np.sign(dg_gt[sel]) == np.sign(dg_pr[sel])).sum()),
                                "pred_move_ratio_median": float(np.median(dg_pr[sel] / dg_gt[sel])) if sel.any() else None}
    out["samples"] = int(len(gt))
    return out


val = run("local/trossen_towel_cc_val", range(8))
trn_ds_eps = np.linspace(0, 147, args.train_eps).round().astype(int)
trn = run("local/trossen_towel_cc_train", trn_ds_eps)
result = {"checkpoint": str(ckpt), "every_frames": args.every, "val": metrics(val), "train": metrics(trn)}
name = f"{ckpt.parent.name}_{ckpt.name}"
(OUT / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps(result, indent=2, ensure_ascii=False))

# 그림: 검증 샘플 4개, 왼팔 관절 0~5 + 그리퍼 (시연 실선, 예측 점선)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.linspace(0, len(val["gt"]) - 1, 4).round().astype(int)
    fig, axes = plt.subplots(4, 7, figsize=(21, 10), sharex=True)
    names = ["L j0", "L j1", "L j2", "L j3", "L j4", "L j5", "L grip(mm)"]
    for r, k in enumerate(idx):
        for c in range(7):
            sc = 1000 if c == 6 else 180 / np.pi
            ax = axes[r, c]
            ax.plot(val["gt"][k][:, c] * sc, "k-", label="demo")
            ax.plot(val["pred"][k][:, c] * sc, "r--", label="model")
            ax.axhline(val["state"][k][c] * sc, color="gray", lw=0.6)
            if r == 0:
                ax.set_title(names[c])
        axes[r, 0].set_ylabel(f"ep{val['where'][k][0]} f{val['where'][k][1]}")
    axes[0, 0].legend()
    fig.suptitle(f"{name}: val demo (black) vs model (red dashed), 50 steps")
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=70)
except Exception as ex:  # noqa: BLE001
    print("plot skipped:", ex, file=sys.stderr)
