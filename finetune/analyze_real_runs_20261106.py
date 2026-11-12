"""Offline audit of five saved runs. Never imports robot clients or drivers."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trossen_aloha_real"))
import retarget as R
from height_calib import HeightCalibration
from workspace import Workspace

RUNS = ("131745", "132009", "132331", "132556", "132847")
ARM = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
OUT = ROOT / "finetune/eval/real_runs_20261106"


def intervals(mask, times):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [[round(float(times[a]), 3), round(float(times[b - 1]), 3)]
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def audit(run, pi05):
    path = ROOT / "trossen_aloha_real/runs" / f"20261106-{run}"
    with np.load(path / "log.npz", allow_pickle=False) as data:
        t, raw, cmd = (data[k].astype(float) for k in ("t", "aloha_action", "native_cmd"))
        obs_step = data["obs_step"]
        assert raw.shape == cmd.shape == (len(t), 14)
        assert np.isfinite(raw).all() and np.isfinite(cmd).all() and np.all(np.diff(t) > 0)
        assert len(obs_step) == len(data["obs_native"]) and np.all(obs_step < len(t))
    cal = HeightCalibration(floor_margin_m=0.040)
    ws = Workspace(margin_m=0.060, z_max_m=0.600)
    changed = np.max(np.abs(raw[:, ARM] - cmd[:, ARM]), axis=1) > 1e-5
    result = {
        "run": path.name,
        "model_inferred_from_handoff": "pi05_lora_30000" if pi05 else "pi0_lora_5000",
        "commands": len(t), "observations": len(obs_step),
        "first_command_s": float(t[0]), "last_command_s": float(t[-1]),
        "active_command_span_s": float(t[-1] - t[0]),
        "rate_hz": float((len(t) - 1) / (t[-1] - t[0])),
        "step_interval_p99_ms": float(np.percentile(np.diff(t), 99) * 1000),
        "step_interval_max_ms": float(np.diff(t).max() * 1000),
        "corrected_arm_commands": int(changed.sum()),
        "corrected_arm_percent": float(changed.mean() * 100),
        "max_arm_correction_deg": float(np.rad2deg(np.abs(raw[:, ARM] - cmd[:, ARM]).max())),
        "correction_intervals_s": intervals(changed, t),
        "sides": {},
    }
    for side, off in (("left", 0), ("right", 7)):
        clipped = np.clip(raw[:, off:off + 6], R.WXAI_LOWER, R.WXAI_UPPER)
        points = np.array([R.fk(q, native=True)[0] for q in clipped])
        floors = np.array([cal._table(side, p) - cal.floor_margin for p in points])
        floor_hits = points[:, 2] < floors
        lo, hi, zmax = ws.boxes[side]
        ws_hits = ((points[:, :2] < lo).any(axis=1)
                   | (points[:, :2] > hi).any(axis=1) | (points[:, 2] > zmax))
        g = cmd[:, off + 6] * 1000
        near_closed = g <= 5
        result["sides"][side] = {
            "raw_floor_hits_current_defaults": int(floor_hits.sum()),
            "raw_floor_max_below_limit_mm": float(np.maximum(floors - points[:, 2], 0).max() * 1000),
            "raw_workspace_hits_current_defaults": int(ws_hits.sum()),
            "gripper_min_mm": float(g.min()), "gripper_max_mm": float(g.max()),
            "gripper_final_mm": float(g[-1]),
            "gripper_near_closed_intervals_s": intervals(near_closed, t),
        }
    cap = cv2.VideoCapture(str(path / "cam_high.mp4"))
    assert cap.isOpened(), path
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frames - 1, 12).round().astype(int)
    sheet = Image.new("RGB", (1280, 3 * 270 + 40), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    draw.text((8, 8), f"{path.name} | {result['model_inferred_from_handoff']} (inferred)", fill="black", font=font)
    sampled = []
    for tile, frame_id in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        assert ok, (path, frame_id)
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame.thumbnail((320, 240))
        x, y = (tile % 4) * 320, 40 + (tile // 4) * 270
        sheet.paste(frame, (x, y))
        step = min(int(frame_id) * 5, len(t) - 1)
        draw.text((x + 5, y + 240), f"t={t[step]:.1f}s  step={step}", fill="black", font=font)
        sampled.append({"frame": int(frame_id), "step_assuming_frame_every_5": step, "log_time_s": float(t[step])})
    cap.release()
    sheet.save(OUT / f"{path.name}.jpg", quality=92)
    result["video_frames"] = frames
    result["video_frame_count_matches_recorder_cadence"] = frames == (len(t) + 4) // 5
    result["sampled_frames"] = sampled
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "limits": [
            "Model identities are inferred from HANDOFF.md, not stored in log.npz.",
            "Arm correction is raw action vs sent command; it does not identify collision interventions.",
            "Floor/workspace counts use current calibration and native defaults, not saved execution counters.",
            "Gripper positions do not prove grasp success; close threshold is an analysis convention (5 mm).",
            "Video/log alignment assumes recorder frame_every=5 and no dropped camera frames; see cadence check.",
            "Early termination cause, collision mode counts and task success are not saved in log.npz.",
        ],
        "user_feedback": {"pi0_lora_5000": "나쁘지 않았음", "pi05_lora_30000": "쓰기 힘들었음",
                          "per_run_success": "미확인"},
        "runs": [audit(run, i < 2) for i, run in enumerate(RUNS)],
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for r in report["runs"]:
        print(r["run"], "steps", r["commands"], "active_s", round(r["active_command_span_s"], 2),
              "hz", round(r["rate_hz"], 2), "corrected", r["corrected_arm_commands"],
              "max_deg", round(r["max_arm_correction_deg"], 2), "sides", r["sides"])


if __name__ == "__main__":
    main()
