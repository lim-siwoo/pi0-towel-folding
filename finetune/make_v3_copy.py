"""LeRobot π0.5 학습용: 관제센터 녹화(v3.0)를 복사해 과제 문장을 모두 "fold towel" 로 바꾼다.
원본은 건드리지 않는다. data/meta 는 새로 쓰고, videos 는 원본 폴더를 가리키는 심볼릭 링크.
학습/검증 에피소드 목록(원본 번호)도 저장한다: 검증 8개는 openpi 학습과 같은 에피소드, 1프레임짜리 3개는 제외.
출력: ~/pi0_aloha_towel/datasets/local/trossen_towel_cc_v3, finetune/lerobot_episodes.json
"""
import glob
import json
import pathlib
import shutil

import pandas as pd

SRC = pathlib.Path.home() / "control_center/data/datasets/task-e2499de8617c"
DST = pathlib.Path.home() / "pi0_aloha_towel/datasets/local/trossen_towel_cc_v3"
TASK = "fold towel"
if DST.exists():
    shutil.rmtree(DST)
(DST / "meta").mkdir(parents=True)
for f in glob.glob(str(SRC / "data/*/*.parquet")):
    out = DST / pathlib.Path(f).relative_to(SRC); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(f); df["task_index"] = 0; df.to_parquet(out, index=False)
for f in glob.glob(str(SRC / "meta/episodes/*/*.parquet")):
    out = DST / pathlib.Path(f).relative_to(SRC); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(f); df["tasks"] = [[TASK] for _ in range(len(df))]
    for c in [c for c in df.columns if c.startswith("stats/task_index/")]:
        df[c] = [[0.0] * len(v) if hasattr(v, "__len__") else 0.0 for v in df[c]] if not c.endswith("count") else df[c]
    df.to_parquet(out, index=False)
pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task")).to_parquet(DST / "meta/tasks.parquet")
info = json.loads((SRC / "meta/info.json").read_text()); info["total_tasks"] = 1
(DST / "meta/info.json").write_text(json.dumps(info, indent=4))
shutil.copy(SRC / "meta/stats.json", DST / "meta/stats.json")
(DST / "videos").symlink_to(SRC / "videos")

conv = [json.loads(l) for l in (pathlib.Path.home() / "pi0_aloha_towel/datasets/local/trossen_towel_cc/meta/episodes.jsonl").read_text().splitlines()]
VAL = [10, 30, 50, 70, 90, 110, 130, 150]  # finetune/split_val.py 와 같은 번호 (변환본 기준)
val = [conv[i]["source_episode_index"] for i in VAL]
train = [c["source_episode_index"] for c in conv if c["source_episode_index"] not in val]
pathlib.Path.home().joinpath("pi0_aloha_towel/finetune/lerobot_episodes.json").write_text(json.dumps({"train": train, "val": val}))
print("train", len(train), "val", val)
