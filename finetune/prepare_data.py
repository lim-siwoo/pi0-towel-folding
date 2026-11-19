"""관제센터에 쌓인 수건 접기 시연을 openpi 학습용으로 한 번에 준비한다.

하는 일 (이미 끝난 부분은 건너뛴다):
  1) 변환   관제센터 녹화(LeRobot v3.0) 중 아직 변환하지 않은 에피소드만 v2.1 로 바꿔
            local/trossen_towel_cc 뒤에 덧붙인다. 이미 변환한 에피소드의 번호는 바뀌지 않는다.
  2) 분리   local/trossen_towel_cc_train / _val 을 다시 만든다.
            - 검증: 변환본 번호가 20으로 나눠 10이 남는 에피소드 (기존 검증 8개 = 10, 30, …, 150 과 같은 규칙)
            - 관제센터에서 '제외'로 표시한 에피소드는 학습·검증 모두에서 뺀다 (--keep-excluded 로 끌 수 있음)
  3) 정규화 학습 데이터가 바뀌었을 때만 정규화 값을 다시 계산한다 (12~17분, GPU 를 쓰지 않음).

사용:
  cd ~/pi0_aloha_towel
  trossen_openpi/.venv/bin/python finetune/prepare_data.py                 # 기본: '수건 접기' 작업
  trossen_openpi/.venv/bin/python finetune/prepare_data.py --dry-run       # 무엇이 바뀔지만 본다
  trossen_openpi/.venv/bin/python finetune/prepare_data.py --tasks "수건 접기" task-xxxxxxxxxxxx

원본(관제센터 데이터)은 읽기만 한다. 관제센터에서 에피소드 번호가 바뀐 흔적이 보이면 멈춘다.
"""
import argparse
import concurrent.futures as cf
import datetime
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from convert_cc_to_v21 import CAMS, MIN_LEN, TASK, cut_video, stats  # noqa: E402

PROJECT = pathlib.Path.home() / "pi0_aloha_towel"
CENTER = pathlib.Path.home() / "control_center/data"
DEFAULT_TASK = "task-e2499de8617c"  # 관제센터 '수건 접기'
CONFIG = "pi0_trossen_towel_cc"
STORE, TRAIN, VAL = "trossen_towel_cc", "trossen_towel_cc_train", "trossen_towel_cc_val"
VAL_EVERY, VAL_REMAINDER = 20, 10
STATE = pathlib.Path(__file__).resolve().parent / "prepare_state.json"


def jl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.rename(path)


def resolve_tasks(names, center):
    known = {t["id"]: t for t in center["tasks"]}
    out = []
    for name in names:
        match = [t for t in known.values() if name in (t["id"], t["name"])]
        if len(match) != 1:
            sys.exit(f"관제센터에서 작업을 찾지 못했습니다: {name!r}\n있는 작업: "
                     + ", ".join(f"{t['name']} ({t['id']})" for t in known.values()))
        out.append(match[0])
    return out


def load_source(task_id):
    src = CENTER / "datasets" / task_id
    info = json.loads((src / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        sys.exit(f"{task_id}: 관제센터 저장 형식이 v3.0 이 아닙니다 ({info.get('codebase_version')}). "
                 "변환 코드를 먼저 확인해야 합니다.")
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(src / "meta/episodes/*/*.parquet")))])
    return src, info, eps.sort_values("episode_index").set_index("episode_index", drop=False)


def same_features(a, b):
    keys = ["action", "observation.state", *CAMS]
    pick = lambda f: {k: (f[k]["dtype"], list(f[k]["shape"]), f[k].get("names")) for k in keys}
    return pick(a) == pick(b)


# ---------------------------------------------------------------- 1) 변환
def convert(tasks, home, workers, dry):
    store = home / "local" / STORE
    if not (store / "meta/info.json").exists():
        sys.exit(f"변환본이 없습니다: {store}\n처음에는 finetune/convert_cc_to_v21.py 로 만들어야 합니다.")
    info = json.loads((store / "meta/info.json").read_text())
    rows = jl(store / "meta/episodes.jsonl")
    srows = jl(store / "meta/episodes_stats.jsonl")
    for r in rows:
        r.setdefault("source_task", DEFAULT_TASK)
    have = {(r["source_task"], r["source_episode_index"]): r for r in rows}
    gidx = int(info["total_frames"])
    assert gidx == sum(r["length"] for r in rows) and len(rows) == info["total_episodes"], "변환본 meta 가 맞지 않습니다"

    new_rows, new_stats, jobs, parquet = [], [], [], []
    for t in tasks:
        src, sinfo, eps = load_source(t["id"])
        if sinfo["fps"] != info["fps"] or not same_features(sinfo["features"], info["features"]):
            sys.exit(f"{t['name']} ({t['id']}): 카메라·관절 구성이나 fps 가 변환본과 다릅니다. 섞을 수 없습니다.")
        # 이미 변환한 에피소드가 원본과 여전히 같은 번호인지 (길이로) 확인한다.
        for (task_id, old), r in have.items():
            if task_id == t["id"] and (old not in eps.index or int(eps.loc[old, "length"]) != r["length"]):
                sys.exit(f"{t['name']}: 원본 에피소드 {old} 가 변환했을 때와 다릅니다 "
                         f"(변환본 길이 {r['length']}). 관제센터에서 번호가 바뀐 것 같습니다. 멈춥니다.")
        todo = [int(i) for i in eps.index if (t["id"], int(i)) not in have and int(eps.loc[i, "length"]) >= MIN_LEN]
        short = [int(i) for i in eps.index if int(eps.loc[i, "length"]) < MIN_LEN]
        print(f"[변환] {t['name']} ({t['id']}): 원본 {len(eps)}개, 변환됨 "
              f"{sum(k[0] == t['id'] for k in have)}개, 새로 변환 {len(todo)}개"
              + (f", 저장 실패로 건너뜀 {short}" if short else ""))
        if dry:  # 번호와 길이만 채워서 분리 계획을 보여 줄 수 있게 한다
            base = len(rows) + len(new_rows)
            new_rows += [{"episode_index": base + k, "length": int(eps.loc[i, "length"]),
                          "source_episode_index": i, "source_task": t["id"]} for k, i in enumerate(todo)]
            continue
        if not todo:
            continue
        data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(src / "data/*/*.parquet")))])
        for old in todo:
            e = eps.loc[old]
            new_i = len(rows) + len(new_rows)
            df = data[data["episode_index"] == old].sort_values("frame_index").reset_index(drop=True)
            n = len(df)
            assert n == int(e["length"]) and (df["frame_index"].values == np.arange(n)).all(), old
            df["episode_index"] = new_i
            df["index"] = np.arange(gidx, gidx + n)
            df["task_index"] = 0
            df["timestamp"] = (np.arange(n) / info["fps"]).astype(np.float32)
            gidx += n
            parquet.append((store / f"data/chunk-000/episode_{new_i:06d}.parquet", df))
            new_rows.append({"episode_index": new_i, "tasks": [TASK], "length": n,
                             "source_episode_index": old, "source_task": t["id"]})
            st = {k: stats(np.stack(df[k].values)) for k in ("action", "observation.state")}
            for k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                st[k] = stats(df[k].values)
            for c in CAMS:
                st[c] = {s: np.asarray([np.asarray(v, float).ravel()[0] for v in e[f"stats/{c}/{s}"]])
                         .reshape(3, 1, 1).tolist() for s in ("min", "max", "mean", "std")}
                st[c]["count"] = [int(np.asarray(e[f"stats/{c}/count"]).ravel()[0])]
                video = (src / f"videos/{c}/chunk-{int(e[f'videos/{c}/chunk_index']):03d}"
                         / f"file-{int(e[f'videos/{c}/file_index']):03d}.mp4")
                jobs.append((video, float(e[f"videos/{c}/from_timestamp"]), n,
                             store / f"videos/chunk-000/{c}/episode_{new_i:06d}.mp4"))
            new_stats.append({"episode_index": new_i, "stats": st})

    if dry or not new_rows:
        return rows + new_rows, len(new_rows)

    # 영상과 parquet 를 먼저 쓰고, 전부 끝난 뒤에 meta 를 바꾼다 (중간에 끊겨도 변환본은 이전 상태 그대로).
    for path, df in parquet:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for f in cf.as_completed([ex.submit(cut_video, *j) for j in jobs]):
            f.result()
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print(f"[변환] 영상 {done}/{len(jobs)}", flush=True)
    backup = store / "meta" / f"backup-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    backup.mkdir()
    for name in ("info.json", "episodes.jsonl", "episodes_stats.jsonl"):
        shutil.copy2(store / "meta" / name, backup / name)
    rows, srows = rows + new_rows, srows + new_stats
    info.update(total_episodes=len(rows), total_frames=gidx, total_videos=len(rows) * len(CAMS),
                splits={"train": f"0:{len(rows)}"})
    write_atomic(store / "meta/episodes_stats.jsonl", "".join(json.dumps(r) + "\n" for r in srows))
    write_atomic(store / "meta/episodes.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_atomic(store / "meta/info.json", json.dumps(info, indent=4))
    print(f"[변환] 변환본 {len(rows)}개, {gidx} 프레임 (이전 meta 는 {backup.name} 에 보관)")
    return rows, len(new_rows)


# ---------------------------------------------------------------- 2) 분리
def plan_split(rows, center, keep_excluded):
    labels = center.get("labels", {})
    quality = lambda r: labels.get(f"{r['source_task']}:{r['source_episode_index']}", {}).get("quality", "success")
    excluded = [] if keep_excluded else [r["episode_index"] for r in rows if quality(r) == "exclude"]
    usable = [r["episode_index"] for r in rows if r["episode_index"] not in set(excluded)]
    val = [i for i in usable if i % VAL_EVERY == VAL_REMAINDER]
    return [i for i in usable if i not in set(val)], val, excluded


def current_split(home, name):
    path = home / "local" / name / "meta/episodes.jsonl"
    return [r["cc_index"] for r in jl(path)] if path.exists() else None


def write_split(home, name, keep):
    """split_val.py 와 같은 방식: parquet·meta 는 번호를 다시 매겨 새로 쓰고, 영상은 변환본에 하드링크."""
    src, dst, tmp = home / "local" / STORE, home / "local" / name, home / "local" / f".{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "meta").mkdir(parents=True)
    info = json.loads((src / "meta/info.json").read_text())
    eps = {e["episode_index"]: e for e in jl(src / "meta/episodes.jsonl")}
    sts = {e["episode_index"]: e for e in jl(src / "meta/episodes_stats.jsonl")}
    rows, srows, g = [], [], 0
    for new, old in enumerate(keep):
        df = pd.read_parquet(src / f"data/chunk-000/episode_{old:06d}.parquet")
        n = len(df)
        df["episode_index"] = new
        df["index"] = np.arange(g, g + n)
        g += n
        out = tmp / f"data/chunk-000/episode_{new:06d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        for c in CAMS:
            v = tmp / f"videos/chunk-000/{c}/episode_{new:06d}.mp4"
            v.parent.mkdir(parents=True, exist_ok=True)
            os.link(src / f"videos/chunk-000/{c}/episode_{old:06d}.mp4", v)
        e = dict(eps[old]); e["episode_index"] = new; e["cc_index"] = old
        rows.append(e)
        s = sts[old]["stats"]
        s["episode_index"] = {k: ([float(new)] if k != "count" else s["episode_index"]["count"])
                              for k in s["episode_index"]}
        s["index"] = {"min": [float(g - n)], "max": [float(g - 1)], "mean": [float(g - n + (n - 1) / 2)],
                      "std": s["index"]["std"], "count": s["index"]["count"]}
        srows.append({"episode_index": new, "stats": s})
    info.update(total_episodes=len(keep), total_frames=g, total_videos=len(keep) * len(CAMS),
                splits={"train": f"0:{len(keep)}"})
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=4))
    (tmp / "meta/tasks.jsonl").write_text((src / "meta/tasks.jsonl").read_text())
    (tmp / "meta/episodes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (tmp / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in srows))
    if dst.exists():
        shutil.rmtree(dst)  # 변환본에서 다시 만들 수 있는 파생 데이터 (영상은 하드링크라 변환본은 그대로)
    tmp.rename(dst)
    return g


# ---------------------------------------------------------------- 3) 정규화
def norm_stats(home, workers):
    openpi = PROJECT / "trossen_openpi"
    out = openpi / "assets" / CONFIG / "local" / TRAIN / "norm_stats.json"
    if out.exists():
        keep = out.with_name(f"norm_stats.{datetime.datetime.now():%Y%m%d-%H%M%S}.bak.json")
        shutil.copy2(out, PROJECT / "finetune/backups" / keep.name)
        print(f"[정규화] 이전 값 보관: finetune/backups/{keep.name}")
    env = dict(os.environ, HF_LEROBOT_HOME=str(home), OPENPI_DATA_HOME=str(PROJECT / "openpi_data"),
               JAX_PLATFORMS="cpu")  # 계산은 numpy 로만 한다. 정책 서버가 GPU 를 쓰고 있어도 돌 수 있게.
    print("[정규화] 계산 중… (12~17분)", flush=True)
    subprocess.run([str(openpi / ".venv/bin/python"), str(PROJECT / "finetune/compute_norm_stats_fast.py"),
                    CONFIG, str(workers)], cwd=openpi, env=env, check=True)
    return out


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # numpy 의 안내문 (결과와 무관)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=[DEFAULT_TASK], help="관제센터 작업 이름 또는 ID (여러 개 가능)")
    ap.add_argument("--keep-excluded", action="store_true", help="관제센터에서 '제외'로 표시한 에피소드도 학습에 넣는다")
    ap.add_argument("--skip-norm-stats", action="store_true", help="정규화 값을 다시 계산하지 않는다")
    ap.add_argument("--dry-run", action="store_true", help="아무것도 쓰지 않고 무엇이 바뀔지만 보여 준다")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--home", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get("HF_LEROBOT_HOME", PROJECT / "datasets")))
    args = ap.parse_args()
    home = args.home
    center = json.loads((CENTER / "center.json").read_text())
    tasks = resolve_tasks(args.tasks, center)

    rows, added = convert(tasks, home, args.workers, args.dry_run)

    train, val, excluded = plan_split(rows, center, args.keep_excluded)
    frames = {r["episode_index"]: r["length"] for r in rows}
    minutes = lambda ids: sum(frames[i] for i in ids) / 30 / 60
    print(f"[분리] 학습 {len(train)}개 ({minutes(train):.1f}분), 검증 {len(val)}개 ({minutes(val):.1f}분)"
          + (f", 관제센터 '제외' 표시로 뺀 것 {len(excluded)}개 {excluded}" if excluded else ""))
    changed = {}
    for name, keep in ((TRAIN, train), (VAL, val)):
        before = current_split(home, name)
        changed[name] = before != keep
        if not changed[name]:
            print(f"[분리] {name}: 그대로 ({len(keep)}개)")
        else:
            was = "없음" if before is None else f"{len(before)}개"
            if before is not None:
                gone, new = sorted(set(before) - set(keep)), sorted(set(keep) - set(before))
                was += (f", 빠지는 것 {gone}" if gone else "") + (f", 새로 들어가는 것 {new}" if new else "")
            print(f"[분리] {name}: {was} → {len(keep)}개" + (" (dry-run: 쓰지 않음)" if args.dry_run else ""))
            if not args.dry_run:
                write_split(home, name, keep)

    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    key = str(home)
    # 처음 실행이고 학습 데이터가 그대로면, 지금 있는 정규화 값이 이 데이터로 계산한 것이다.
    stats_for = state.get(key, {}).get("norm_stats_train", None if changed[TRAIN] else train)
    need_stats = stats_for != train
    if args.dry_run:
        print("[정규화] " + ("다시 계산해야 함 (dry-run: 하지 않음)" if need_stats else "그대로"))
        return
    if need_stats and args.skip_norm_stats:
        print("[정규화] 학습 데이터가 바뀌었지만 --skip-norm-stats 라서 계산하지 않았습니다. 이대로 학습하면 안 됩니다.")
    elif need_stats and home != PROJECT / "datasets":
        print("[정규화] --home 이 기본 위치가 아니라서 계산하지 않았습니다 (정규화 값은 한 곳에만 저장됨).")
    elif need_stats:
        out = norm_stats(home, args.workers)
        stats_for = train
        print(f"[정규화] 저장: {out}")
    else:
        print("[정규화] 그대로 (학습 데이터가 바뀌지 않음)")
    state[key] = {"norm_stats_train": stats_for, "updated": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                  "tasks": [t["id"] for t in tasks]}
    write_atomic(STATE, json.dumps(state, indent=1))

    print("\n다음 단계 (새 데이터로 학습할 때):")
    print("  기존 실험(towel_cc_lora_split, 40000 체크포인트)을 덮어쓰지 않도록 새 이름으로 학습하세요.")
    print(f"  cd ~/pi0_aloha_towel && EXP=towel_cc_lora_{datetime.datetime.now():%m%d} ./finetune/train.sh")
    if changed[TRAIN]:
        print("  주의: 학습 데이터와 정규화 값이 바뀌었으므로 이전 실험을 --resume 으로 잇지 마세요.")


if __name__ == "__main__":
    main()
