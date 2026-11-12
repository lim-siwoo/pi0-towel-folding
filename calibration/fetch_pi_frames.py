"""PI ALOHA 데이터(aloha_pen_uncap_diverse)에서 카메라 4개 프레임을 부분 읽기로 가져온다.
출력: calibration/pi_aloha_pen/frames/ep{e}_f{i}_{cam}.png
"""

import pathlib
import sys

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

OUT = pathlib.Path(__file__).resolve().parent / "pi_aloha_pen" / "frames"
OUT.mkdir(parents=True, exist_ok=True)
CAMS = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
EPISODES = [int(x) for x in (sys.argv[1:] or ["0", "5"])]
fs = HfFileSystem()
for e in EPISODES:
    path = f"datasets/physical-intelligence/aloha_pen_uncap_diverse/data/chunk-000/episode_{e:06d}.parquet"
    with fs.open(path, "rb", block_size=2 * 1024 * 1024) as f:
        pf = pq.ParquetFile(f)
        print(f"episode {e}: row groups {pf.num_row_groups}, rows {pf.metadata.num_rows}")
        t = pf.read_row_group(0, columns=[f"observation.images.{c}" for c in CAMS] + ["frame_index"])
        idx = t.column("frame_index").to_pylist()
        for row in (0, min(len(idx) - 1, 120)):
            for c in CAMS:
                cell = t.column(f"observation.images.{c}")[row].as_py()
                data = cell["bytes"] if isinstance(cell, dict) else cell
                (OUT / f"ep{e}_f{idx[row]}_{c}.png").write_bytes(data)
print("saved to", OUT)
