"""PI 의 ALOHA 데이터(physical-intelligence/aloha_pen_uncap_diverse)에서 관절 열만 부분 읽기로 가져온다.

에피소드 파일 하나가 영상 포함 약 580MB 라서, parquet 열 단위 범위 요청으로 state/action 과
첫 프레임 cam_high 이미지 한 장만 받는다. 출력: calibration/pi_aloha_pen/{states.npz, cam_high_ep0.png}
"""

import io
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = pathlib.Path(__file__).resolve().parent / "pi_aloha_pen"
OUT.mkdir(exist_ok=True)
fs = HfFileSystem()
base = "datasets/physical-intelligence/aloha_pen_uncap_diverse/data/chunk-000"

states, actions, eps = [], [], []
for e in range(N_EP):
    with fs.open(f"{base}/episode_{e:06d}.parquet", "rb", block_size=2 * 1024 * 1024) as f:
        pf = pq.ParquetFile(f)
        if e == 0:
            print("columns:", pf.schema_arrow.names)
        t = pf.read(columns=["observation.state", "action"])
        s = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))
        a = np.stack(t.column("action").to_numpy(zero_copy_only=False))
        if e == 0:  # 첫 프레임 cam_high 한 장 (시점 비교용)
            img_col = [c for c in pf.schema_arrow.names if "cam_high" in c]
            if img_col:
                rg = pf.read_row_group(0, columns=img_col)
                cell = rg.column(0)[0].as_py()
                data = cell["bytes"] if isinstance(cell, dict) else cell
                (OUT / "cam_high_ep0.png").write_bytes(data)
    states.append(s); actions.append(a); eps.append(np.full(len(s), e))
    print(f"episode {e}: {len(s)} frames")
np.savez(OUT / "states.npz", state=np.concatenate(states), action=np.concatenate(actions), episode=np.concatenate(eps))
print("saved", OUT)
