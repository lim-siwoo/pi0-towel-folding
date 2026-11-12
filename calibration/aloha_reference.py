"""1단계: 실제 ALOHA 수건 시연(lerobot/aloha_static_towel)에서 ViperX 그리퍼 끝 위치 분포를 구한다.

각 팔 베이스 좌표계(retarget.py 의 ALOHA FK, VX300S URDF 기반) 기준:
  - 전체 프레임의 그리퍼 끝 높이(z) 분포 -> 하위 백분위 = 책상 근처 높이
  - 그리퍼가 닫혀 있는(수건을 쥔) 프레임의 높이
  - 작업 공간 (x, y) 범위
출력: calibration/aloha_reference.json
"""

import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trossen_aloha_real"))
import retarget as R  # noqa: E402

table = pq.read_table(HERE / "aloha_static_towel/data.parquet").to_pandas()
state = np.stack(table["observation.state"].to_numpy()).astype(np.float64)
action = np.stack(table["action"].to_numpy()).astype(np.float64)
episode = table["episode_index"].to_numpy()
print(f"frames {len(state)}, episodes {len(np.unique(episode))}")

out = {"source": "lerobot/aloha_static_towel (data/chunk-000/file-000.parquet)", "frames": int(len(state)),
       "frame": "each ViperX arm base frame, retarget.fk(native=False), metres", "arms": {}}
for name, off in (("left", 0), ("right", 7)):
    P = np.array([R.fk(q[off:off + 6], native=False)[0] for q in state])
    grip = state[:, off + 6]
    z = P[:, 2]
    # 그리퍼 '닫힘' 판정: 해당 팔 그리퍼 값의 하위 20% 구간 (단위는 데이터셋 정규화 그대로)
    g_thr = np.percentile(grip, 20)
    closed = grip <= g_thr
    # 에피소드별 최저 높이
    ep_min = np.array([z[episode == e].min() for e in np.unique(episode)])
    arm = {
        "gripper_state_range": [float(grip.min()), float(grip.max())],
        "gripper_closed_threshold": float(g_thr),
        "z_percentiles_mm": {p: round(float(np.percentile(z, p)) * 1000, 1) for p in (0.5, 1, 5, 25, 50, 75, 95)},
        "z_min_per_episode_mm": {"median": round(float(np.median(ep_min)) * 1000, 1),
                                 "p10": round(float(np.percentile(ep_min, 10)) * 1000, 1),
                                 "p90": round(float(np.percentile(ep_min, 90)) * 1000, 1)},
        "z_when_closed_mm": {p: round(float(np.percentile(z[closed], p)) * 1000, 1) for p in (5, 25, 50)},
        "x_range_mm": [round(float(np.percentile(P[:, 0], 1)) * 1000), round(float(np.percentile(P[:, 0], 99)) * 1000)],
        "y_range_mm": [round(float(np.percentile(P[:, 1], 1)) * 1000), round(float(np.percentile(P[:, 1], 99)) * 1000)],
        "xy_at_lowest_5pct_mm": np.round(np.median(P[z <= np.percentile(z, 5)][:, :2], axis=0) * 1000).tolist(),
    }
    out["arms"][name] = arm
    print(f"\n[{name}] gripper state range {arm['gripper_state_range']}, closed thr {g_thr:.3f}")
    print("  z percentiles (mm):", arm["z_percentiles_mm"])
    print("  per-episode min z (mm):", arm["z_min_per_episode_mm"])
    print("  z when gripper closed (mm):", arm["z_when_closed_mm"])
    print("  x range (mm):", arm["x_range_mm"], " y range (mm):", arm["y_range_mm"],
          " xy at lowest 5%:", arm["xy_at_lowest_5pct_mm"])

(HERE / "aloha_reference.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\nsaved", HERE / "aloha_reference.json")
