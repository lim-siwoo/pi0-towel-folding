"""학습 데이터 과제 문장(프롬프트)을 관제센터 '수건 접기' 과제의 영어 지시문으로 바꾼다.
(반으로 접기 → 90도 돌리기 → 다시 반으로 접기 → 내려놓고 놓기)
대상: LeRobot v3 복사본(trossen_towel_cc_v3, π0.5 학습) + v2.1 변환본(openpi 학습: 전체/학습/검증). 원본은 건드리지 않는다.
"""
import json
import pathlib

import pandas as pd

TASK = "fold towel"  # 2026-11-11 사용자 지시: 모든 수건 지시문을 fold towel 로 통일
D = pathlib.Path.home() / "pi0_aloha_towel/datasets/local"

v3 = D / "trossen_towel_cc_v3"
pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task")).to_parquet(v3 / "meta/tasks.parquet")
for f in (v3 / "meta/episodes").glob("*/*.parquet"):
    df = pd.read_parquet(f); df["tasks"] = [[TASK] for _ in range(len(df))]; df.to_parquet(f, index=False)
print("v3:", pd.read_parquet(v3 / "meta/tasks.parquet").index[0][:60])

for name in ("trossen_towel_cc", "trossen_towel_cc_train", "trossen_towel_cc_val"):
    m = D / name / "meta"
    (m / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    rows = [json.loads(l) for l in (m / "episodes.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r["tasks"] = [TASK]
    (m / "episodes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(name, len(rows), "episodes")
(pathlib.Path.home() / "pi0_aloha_towel/finetune/PROMPT.txt").write_text(TASK + "\n")
