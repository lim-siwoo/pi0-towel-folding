"""공개 ALOHA 수건 데이터를 이 로봇(WidowX AI) 관절·30fps 로 변환해 LeRobot v3 데이터로 쓴다.

데이터와 지시문
  - mo0821/aloha_fold_towel      → "fold towel"             (실제 접기, 30개)
  - zhifeishen/aloha_fold_towel  → "spread the towel flat"  (구겨진 수건 펴기, 100개)
관절: 표준 ALOHA(ViperX 300s) 관절 → 실물 실행과 같은 경로(env.apply_action)로 변환
      ALOHA FK → 손가락 끝 기준 + 실측 책상 높이(height_calib, fingertip) → 작업 영역 제한(workspace)
      → WidowX AI IK (직전 프레임에서 시작, 실패·해 전환 시 가까운 근사 해)
그리퍼: 데이터셋마다 닫힘/열림(1%/99% 분위)을 우리 녹화의 닫힘/열림(1%/99% 분위)으로 직선 대응
        (상태와 행동을 따로 — 두 데이터 모두 상태와 행동의 그리퍼 눈금이 다르다)
시간: 50fps → 30fps (관절 선형 보간, 영상은 가장 가까운 프레임)
품질: 에피소드마다 근사 해 비율·최대 관절 변화·변환 오차를 기록하고, 기준을 넘으면 뺀다.
출력: ~/pi0_aloha_towel/datasets/local/towel_public_wxai (v3), finetune/public_convert_report.json
사용: lerobot_pi05/.venv/bin/python convert_public.py [--limit N] [--workers 16]
"""
import argparse
import concurrent.futures as cf
import glob
import json
import logging
import pathlib
import shutil
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq

HOME = pathlib.Path.home()
sys.path.insert(0, str(HOME / "pi0_aloha_towel/trossen_aloha_real"))
logging.disable(logging.WARNING)
import retarget as R  # noqa: E402
from height_calib import HeightCalibration  # noqa: E402
from workspace import Workspace  # noqa: E402

RAW = HOME / "pi0_aloha_towel/datasets/public_raw"
OURS = HOME / "pi0_aloha_towel/datasets/local/trossen_towel_cc_v3"
OUT = HOME / "pi0_aloha_towel/datasets/local/towel_public_wxai"
SOURCES = [
    {"name": "mo0821__aloha_fold_towel", "task": "fold towel",
     "cams": {"cam_high": "observation.images.camera_high", "cam_low": "observation.images.camera_low",
              "cam_left_wrist": "observation.images.camera_wrist_left", "cam_right_wrist": "observation.images.camera_wrist_right"}},
    {"name": "zhifeishen__aloha_fold_towel", "task": "spread the towel flat",
     "cams": {c: f"observation.images.{c}" for c in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")}},
]
SRC_FPS, FPS = 50, 30
MAX_APPROX_RATIO = 0.05     # 근사 해 비율이 이보다 크면 에피소드 제외
MAX_FRAME_JUMP = 0.35       # 30fps 한 프레임 관절 변화(rad)가 이보다 크면 제외 (해 전환)
MAX_APPROX_ERR = 0.03       # 근사 해 위치 오차(m) 상한

CAL = HeightCalibration(mode="fingertip")
WS = Workspace(enabled=True, margin_m=0.02)


def approx_ik(pos, rot, seed, reg=0.02):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    lo, hi = R.WXAI_LOWER, R.WXAI_UPPER
    seed = np.clip(np.asarray(seed, float), lo + 1e-6, hi - 1e-6)

    def err(q):
        p, r = R.fk(q, native=True)
        return np.concatenate((p - pos, 0.15 * Rotation.from_matrix(rot.T @ r).as_rotvec(), reg * (q - seed)))

    res = least_squares(err, seed, bounds=(lo, hi), max_nfev=300)
    return res.x, float(np.linalg.norm(R.fk(res.x, native=True)[0] - pos))


def to_native(aloha14, seeds):
    """ALOHA 14 → WidowX AI 관절 12개 + (approx 사용 여부, 오차). 그리퍼는 따로."""
    out, approx, errs = np.zeros(14), 0, []
    for side, off in (("left", 0), ("right", 7)):
        seed = seeds[off:off + 6]
        p, rot = R.fk(aloha14[off:off + 6], native=False)
        p = WS.clamp(side, CAL.to_native(side, p, rot))
        q = None
        try:
            q = R.ik(p, rot, seed, native=True)
            if np.abs(q - seed).max() > 0.3:
                q = None
        except R.RetargetError:
            pass
        if q is None:
            q, e = approx_ik(p, rot, seed)
            approx += 1; errs.append(e)
        out[off:off + 6] = q
    return out, approx, errs


def resample(x, n_out):
    t_src = np.arange(len(x)) / SRC_FPS
    t = np.arange(n_out) / FPS
    return np.stack([np.interp(t, t_src, x[:, j]) for j in range(x.shape[1])], 1)


def grip_scale(src_lo, src_hi, dst_lo, dst_hi):
    return lambda g: np.clip(dst_lo + (g - src_lo) / (src_hi - src_lo) * (dst_hi - dst_lo), min(dst_lo, 0.0), 0.04)


def convert_episode(path, src, gmap):
    t = pq.read_table(path, columns=["observation.state", "action"])
    st = np.stack(t.column(0).to_numpy(zero_copy_only=False)).astype(np.float64)
    ac = np.stack(t.column(1).to_numpy(zero_copy_only=False)).astype(np.float64)
    n = int((len(st) - 1) * FPS / SRC_FPS) + 1
    st30, ac30 = resample(st, n), resample(ac, n)
    seeds = np.array(list(R.WXAI_STAGED) + [0] + list(R.WXAI_STAGED) + [0], float)
    ns, na = np.zeros((n, 14)), np.zeros((n, 14))
    approx, errs = 0, []
    for k in range(n):
        q, a1, e1 = to_native(st30[k], seeds)
        seeds = q
        qa, a2, e2 = to_native(ac30[k], q)
        ns[k], na[k] = q, qa
        approx += a1 + a2; errs += e1 + e2
    for off in (6, 13):
        ns[:, off] = gmap["state"](st30[:, off]); na[:, off] = gmap["action"](ac30[:, off])
    arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    jump = float(np.abs(np.diff(ns[:, arm], axis=0)).max()) if n > 1 else 0.0
    q = {"file": pathlib.Path(path).name, "frames": n, "approx_ratio": approx / (4 * n),
         "approx_err_max_mm": round(max(errs) * 1000, 1) if errs else 0.0, "max_frame_jump_rad": round(jump, 3)}
    q["ok"] = q["approx_ratio"] <= MAX_APPROX_RATIO and jump <= MAX_FRAME_JUMP and (not errs or max(errs) <= MAX_APPROX_ERR)
    return ns.astype(np.float32), na.astype(np.float32), q


def frames(path, src, n):
    """30fps 프레임 인덱스에 맞는 카메라 영상 (RGB 480x640)."""
    cols = list(src["cams"].values())
    t = pq.read_table(path, columns=cols)
    idx = np.minimum(np.round(np.arange(n) * SRC_FPS / FPS).astype(int), t.num_rows - 1)
    for k in range(n):
        out = {}
        for ours, col in src["cams"].items():
            cell = t.column(col)[int(idx[k])].as_py(); b = cell["bytes"] if isinstance(cell, dict) else cell
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if img.shape[:2] != (480, 640):
                img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
            out[f"observation.images.{ours}"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        yield out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    from lerobot.datasets import LeRobotDataset

    ours_stats = json.loads((OURS / "meta/stats.json").read_text())
    dst = {k: (ours_stats[k]["q01"][6], ours_stats[k]["q99"][6]) for k in ("observation.state", "action")}
    info = json.loads((OURS / "meta/info.json").read_text())
    feats = {k: v for k, v in info["features"].items() if k not in ("timestamp", "frame_index", "episode_index", "index", "task_index")}
    if OUT.exists():
        shutil.rmtree(OUT)
    ds = LeRobotDataset.create("local/towel_public_wxai", FPS, feats, root=OUT, robot_type=info["robot_type"],
                               use_videos=True, image_writer_threads=8)
    report = {}
    for src in SOURCES:
        files = sorted(glob.glob(str(RAW / src["name"] / "data/chunk-*/*.parquet")))
        if args.limit:
            files = files[:args.limit]
        # 데이터셋 전체 그리퍼 분위수 (상태, 행동 따로)
        g = {"state": [], "action": []}
        for f in files:
            t = pq.read_table(f, columns=["observation.state", "action"])
            for key, col in (("state", 0), ("action", 1)):
                x = np.stack(t.column(col).to_numpy(zero_copy_only=False))
                g[key].append(x[:, [6, 13]].ravel())
        gmap, gq = {}, {}
        for key, ours in (("state", "observation.state"), ("action", "action")):
            lo, hi = np.percentile(np.concatenate(g[key]), [1, 99])
            gmap[key] = grip_scale(lo, hi, *dst[ours]); gq[key] = [float(lo), float(hi)]
        with cf.ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(convert_episode, files, [src] * len(files), [None] * len(files))) if False else None
        # 그리퍼 함수는 피클이 안 되므로 관절만 병렬로 변환하고 그리퍼는 여기서 적용
        with cf.ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(_convert_joints, files))
        kept = 0
        for f, (ns, na, st30g, ac30g, q) in zip(files, res):
            for i, off in enumerate((6, 13)):
                ns[:, off] = gmap["state"](st30g[:, i]); na[:, off] = gmap["action"](ac30g[:, i])
            report[f"{src['name']}/{q['file']}"] = q
            if not q["ok"]:
                continue
            for k, imgs in enumerate(frames(f, src, len(ns))):
                ds.add_frame({"observation.state": ns[k], "action": na[k], **imgs, "task": src["task"]})
            ds.save_episode()
            kept += 1
            print(f"{src['name']} {q['file']} 저장 ({kept})", flush=True)
        report[src["name"]] = {"files": len(files), "kept": kept, "gripper_src_q01_q99": gq,
                               "gripper_dst_q01_q99": {k: list(v) for k, v in dst.items()}}
        print(src["name"], "kept", kept, "/", len(files), flush=True)
    ds.finalize()
    (HOME / "pi0_aloha_towel/finetune/public_convert_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("완료", OUT)


def _convert_joints(path):
    """관절 변환만 (병렬 작업용). 그리퍼는 원래 값(30fps)을 함께 돌려준다."""
    ns, na, q = convert_episode(path, None, {"state": lambda g: g, "action": lambda g: g})
    t = pq.read_table(path, columns=["observation.state", "action"])
    st = np.stack(t.column(0).to_numpy(zero_copy_only=False)).astype(np.float64)
    ac = np.stack(t.column(1).to_numpy(zero_copy_only=False)).astype(np.float64)
    n = len(ns)
    return ns, na, resample(st[:, [6, 13]], n), resample(ac[:, [6, 13]], n), q


if __name__ == "__main__":
    main()
