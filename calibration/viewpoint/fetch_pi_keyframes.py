"""PI ALOHA 데이터에서 카메라 4개 프레임 + 같은 순간의 관절값(observation.state)을 받는다.
출력: viewpoint/pi_frames/ep{e}_f{i}_{cam}.png, viewpoint/pi_frames/states.json
"""
import json
import pathlib
import sys

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

OUT = pathlib.Path(__file__).resolve().parent / "pi_frames"
OUT.mkdir(parents=True, exist_ok=True)
CAMS = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
EPISODES = [int(x) for x in (sys.argv[1:] or ["0", "5", "10", "15", "20"])]
FRAMES = [0, 60, 120, 180, 240]
fs = HfFileSystem()
states = {}
for e in EPISODES:
    path = f"datasets/physical-intelligence/aloha_pen_uncap_diverse/data/chunk-000/episode_{e:06d}.parquet"
    with fs.open(path, "rb", block_size=2 * 1024 * 1024) as f:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=[f"observation.images.{c}" for c in CAMS] + ["frame_index", "observation.state"])
            idx = t.column("frame_index").to_pylist()
            for row, fi in enumerate(idx):
                if fi not in FRAMES:
                    continue
                for c in CAMS:
                    cell = t.column(f"observation.images.{c}")[row].as_py()
                    (OUT / f"ep{e}_f{fi}_{c}.png").write_bytes(cell["bytes"] if isinstance(cell, dict) else cell)
                states[f"ep{e}_f{fi}"] = t.column("observation.state")[row].as_py()
    print("episode", e, "done")
(OUT / "states.json").write_text(json.dumps(states, indent=1))
print(len(states), "frames ->", OUT)
