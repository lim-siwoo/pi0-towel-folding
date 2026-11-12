"""관제센터 수건 접기 녹화(LeRobot v3.0)를 openpi 학습용 LeRobot v2.1 형식으로 변환한다.

원본(읽기만): ~/control_center/data/datasets/task-e2499de8617c
출력: $HF_LEROBOT_HOME/local/trossen_towel_cc  (기본 HF_LEROBOT_HOME = ~/pi0_aloha_towel/datasets)
- 1프레임짜리(저장 실패) 에피소드는 뺀다. 에피소드 번호·전역 index 는 0부터 다시 매긴다.
- 과제 문장은 전부 "fold towel" (학습 config 도 default_prompt="fold towel").
- 영상: 이어 붙은 v3 파일에서 에피소드 구간을 잘라 libx264(-g 2, crf 18)로 다시 인코딩한다.
  (v2.1 은 에피소드마다 파일 하나, -g 2 는 학습 중 임의 프레임 읽기를 빠르게 한다)
사용: python convert_cc_to_v21.py [--workers 12] [--limit N]
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd

SRC = pathlib.Path.home() / "control_center/data/datasets/task-e2499de8617c"
HOME = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", pathlib.Path.home() / "pi0_aloha_towel/datasets"))
DST = HOME / "local/trossen_towel_cc"
TASK = "fold towel"
CAMS = ["observation.images.cam_high", "observation.images.cam_low",
        "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
MIN_LEN = 30


def cut_video(src, start, n, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.6f}", "-i", str(src), "-frames:v", str(n),
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "2", "-pix_fmt", "yuv420p",
                    "-r", "30", str(tmp)], check=True)
    got = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                              "stream=nb_read_frames", "-of", "csv=p=0", str(tmp)],
                             capture_output=True, text=True, check=True).stdout.strip())
    if got != n:
        raise RuntimeError(f"{out}: 프레임 {got} != {n}")
    tmp.rename(out)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    info = json.loads((SRC / "meta/info.json").read_text())
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SRC / "meta/episodes/*/*.parquet")))])
    eps = eps.sort_values("episode_index")
    eps = eps[eps["length"] >= MIN_LEN]
    if args.limit:
        eps = eps.head(args.limit)
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SRC / "data/*/*.parquet")))])
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "meta").mkdir(exist_ok=True)

    jobs, ep_rows, ep_stats, gidx = [], [], [], 0
    for new_i, (_, e) in enumerate(eps.iterrows()):
        old = int(e["episode_index"])
        df = data[data["episode_index"] == old].sort_values("frame_index").reset_index(drop=True)
        n = len(df)
        assert n == int(e["length"]) and (df["frame_index"].values == np.arange(n)).all(), old
        df["episode_index"] = new_i
        df["index"] = np.arange(gidx, gidx + n)
        df["task_index"] = 0
        df["timestamp"] = (np.arange(n) / info["fps"]).astype(np.float32)
        gidx += n
        out = DST / f"data/chunk-000/episode_{new_i:06d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        ep_rows.append({"episode_index": new_i, "tasks": [TASK], "length": n, "source_episode_index": old})
        st = {k: stats(np.stack(df[k].values)) for k in ("action", "observation.state")}
        for k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            st[k] = stats(df[k].values)
        for c in CAMS:
            st[c] = {s: np.asarray([np.asarray(v, float).ravel()[0] for v in e[f"stats/{c}/{s}"]]).reshape(3, 1, 1).tolist()
                     for s in ("min", "max", "mean", "std")}
            st[c]["count"] = [int(np.asarray(e[f"stats/{c}/count"]).ravel()[0])]
            src = SRC / f"videos/{c}/chunk-{int(e[f'videos/{c}/chunk_index']):03d}/file-{int(e[f'videos/{c}/file_index']):03d}.mp4"
            dst = DST / f"videos/chunk-000/{c}/episode_{new_i:06d}.mp4"
            if not dst.exists():
                jobs.append((src, float(e[f"videos/{c}/from_timestamp"]), n, dst))
        ep_stats.append({"episode_index": new_i, "stats": st})

    feats = info["features"]
    for c in CAMS:
        feats[c]["info"]["video.codec"] = "h264"
        feats[c]["info"]["video.pix_fmt"] = "yuv420p"
    new_info = {"codebase_version": "v2.1", "robot_type": info["robot_type"], "total_episodes": len(ep_rows),
                "total_frames": gidx, "total_tasks": 1, "total_videos": len(ep_rows) * len(CAMS), "total_chunks": 1,
                "chunks_size": 1000, "fps": info["fps"], "splits": {"train": f"0:{len(ep_rows)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": feats}
    (DST / "meta/info.json").write_text(json.dumps(new_info, indent=4))
    (DST / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    (DST / "meta/episodes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ep_rows))
    (DST / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep_stats))
    print(f"에피소드 {len(ep_rows)}, 프레임 {gidx}, 영상 인코딩 {len(jobs)}개", flush=True)

    done = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(cut_video, *j) for j in jobs]
        for f in cf.as_completed(futs):
            f.result()
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print(f"영상 {done}/{len(jobs)}", flush=True)
    print("완료:", DST)


if __name__ == "__main__":
    main()
