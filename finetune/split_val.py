"""변환한 데이터(local/trossen_towel_cc)를 학습용/검증용으로 나눈다. 검증 에피소드는 학습에 쓰지 않는다.
- 검증: 녹화 기간 전체에 고르게 퍼진 8개 (VAL 번호, trossen_towel_cc 기준)
- 출력: local/trossen_towel_cc_train (148개), local/trossen_towel_cc_val (8개)
  parquet·meta 는 에피소드 번호/전역 index 를 다시 매겨 새로 쓰고, 영상은 원본 파일에 하드링크(복사 없음).
"""
import json
import os
import pathlib

import numpy as np
import pandas as pd

HOME = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", pathlib.Path.home() / "pi0_aloha_towel/datasets"))
SRC = HOME / "local/trossen_towel_cc"
VAL = [10, 30, 50, 70, 90, 110, 130, 150]


def jl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write(name, keep):
    dst = HOME / f"local/{name}"
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    info = json.loads((SRC / "meta/info.json").read_text())
    eps = {e["episode_index"]: e for e in jl(SRC / "meta/episodes.jsonl")}
    sts = {e["episode_index"]: e for e in jl(SRC / "meta/episodes_stats.jsonl")}
    cams = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    rows, srows, g = [], [], 0
    for new, old in enumerate(keep):
        df = pd.read_parquet(SRC / f"data/chunk-000/episode_{old:06d}.parquet")
        n = len(df)
        df["episode_index"] = new
        df["index"] = np.arange(g, g + n)
        g += n
        out = dst / f"data/chunk-000/episode_{new:06d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        for c in cams:
            v = dst / f"videos/chunk-000/{c}/episode_{new:06d}.mp4"
            v.parent.mkdir(parents=True, exist_ok=True)
            if v.exists():
                v.unlink()
            os.link(SRC / f"videos/chunk-000/{c}/episode_{old:06d}.mp4", v)
        e = dict(eps[old]); e["episode_index"] = new; e["cc_index"] = old
        rows.append(e)
        s = sts[old]["stats"]
        s["episode_index"] = {k: ([float(new)] if k != "count" else s["episode_index"]["count"]) for k in s["episode_index"]}
        s["index"] = {"min": [float(g - n)], "max": [float(g - 1)], "mean": [float(g - n + (n - 1) / 2)],
                      "std": s["index"]["std"], "count": s["index"]["count"]}
        srows.append({"episode_index": new, "stats": s})
    info.update(total_episodes=len(keep), total_frames=g, total_videos=len(keep) * len(cams),
                splits={"train": f"0:{len(keep)}"})
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    (dst / "meta/tasks.jsonl").write_text((SRC / "meta/tasks.jsonl").read_text())
    (dst / "meta/episodes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (dst / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in srows))
    print(name, len(keep), "episodes", g, "frames")


n = json.loads((SRC / "meta/info.json").read_text())["total_episodes"]
write("trossen_towel_cc_train", [i for i in range(n) if i not in VAL])
write("trossen_towel_cc_val", VAL)
