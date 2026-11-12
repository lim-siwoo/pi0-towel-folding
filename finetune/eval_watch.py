"""학습 중 체크포인트 자동 검증 + 계속 학습 여부 판정.

5,000 스텝마다 남는 체크포인트가 생기면 eval_val.py 를 CPU 로 돌린다(학습이 GPU 를 다 쓰므로).
결과를 모아 finetune/eval/report.md 와 trend.png 를 갱신하고, 판정을 한 줄로 출력한다.

판정 기준 (검증 에피소드 = 학습에 쓰지 않은 8개)
  - 기준선: 지금 자세 그대로 있기. 검증 팔 오차가 이것보다 15% 이상 작아야 "학습됨"으로 본다.
  - 계속: 검증 팔 오차가 지금까지 최저보다 2% 넘게 좋아짐
  - 과적합 의심 → 중단 권장: 검증 오차가 최저 대비 연속 2번 나빠지고, 그동안 학습 에피소드 오차는 좋아짐
  - 정체 → 중단 권장: 연속 3번(15,000 스텝) 동안 최저 대비 2% 넘게 좋아지지 않음
  - 방향: 팔 이동 방향 일치(코사인 > 0.7) 비율, 그리퍼 열고 닫는 방향 일치 비율을 함께 본다
사용 (trossen_openpi 폴더): .venv/bin/python ~/pi0_aloha_towel/finetune/eval_watch.py [--exp towel_cc_lora_split]
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import report_html  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="towel_cc_lora_split")
ap.add_argument("--period", type=int, default=5000)
ap.add_argument("--every", type=int, default=90)
ap.add_argument("--config", default="pi0_trossen_towel_cc")
ap.add_argument("--kind", choices=["openpi", "lerobot"], default="openpi")
ap.add_argument("--lerobot-run", default="pi05_towel_lora")
args = ap.parse_args()

ROOT = pathlib.Path.home() / "pi0_aloha_towel"
CK = (ROOT / f"trossen_openpi/checkpoints/{args.config}/{args.exp}" if args.kind == "openpi"
      else ROOT / f"finetune/lerobot_runs/{args.lerobot_run}/checkpoints")
TITLE = "π0 base + LoRA (openpi, pi0_trossen_towel_cc)" if args.kind == "openpi" else "π0.5 base + LoRA (LeRobot pi05, 학습 단계 RTC)"
LEROBOT_PY = ROOT / "lerobot_pi05/.venv/bin/python"
EV = ROOT / "finetune/eval"
EV.mkdir(exist_ok=True)
SUM = EV / f"{args.exp}_summary.json"
summary = json.loads(SUM.read_text()) if SUM.exists() else {}


def row(step, r):
    v, t = r["val"], r["train"]
    return {"step": step,
            "val_arm": v["model"]["arm_deg_mean"], "val_grip": v["model"]["grip_mm_mean"],
            "hold_arm": v["hold"]["arm_deg_mean"], "hold_grip": v["hold"]["grip_mm_mean"],
            "train_arm": t["model"]["arm_deg_mean"],
            "val_arm_50": v["model"]["arm_deg_at_step"]["50"],
            "dir_ok": v["arm_direction"]["same_direction_ratio"], "dir_cos": v["arm_direction"]["cos_median"],
            "move_ratio": v["arm_direction"]["move_size_ratio_median"],
            "grip_dir": v["gripper_direction"]["same_direction"] / max(1, v["gripper_direction"]["events"])}


def verdict(rows):
    rows = sorted(rows, key=lambda r: r["step"])
    last = rows[-1]
    learned = last["val_arm"] < 0.85 * last["hold_arm"]
    best_i, best = 0, rows[0]["val_arm"]
    no_improve, worse = 0, 0
    for i, r in enumerate(rows):
        if r["val_arm"] < best * 0.98:
            best_i, best, no_improve, worse = i, r["val_arm"], 0, 0
        else:
            no_improve += 1
            worse = worse + 1 if r["val_arm"] > best * 1.02 else 0
    train_better = len(rows) >= 3 and rows[-1]["train_arm"] < rows[-3]["train_arm"]
    best_step = rows[best_i]["step"]
    if worse >= 2 and train_better:
        v = f"중단 권장(과적합 의심): 검증 오차가 {best_step} 스텝 최저 뒤로 연속 나빠지고 학습 오차만 좋아짐 → {best_step} 체크포인트 사용"
    elif no_improve >= 3:
        v = f"중단 권장(정체): {best_step} 스텝 이후 {no_improve * args.period} 스텝 동안 2% 넘는 개선 없음 → {best_step} 체크포인트 사용"
    elif not learned:
        v = "계속(아직 기준선보다 충분히 좋지 않음)"
    else:
        v = f"계속: 검증 오차 개선 중 (최저 {best:.2f}도 @ {best_step})"
    return v, best_step, learned


def report(rows):
    rows = sorted(rows, key=lambda r: r["step"])
    v, best_step, learned = verdict(rows)
    lines = [f"# 검증 결과 — {args.exp}", "", f"판정: **{v}**", "",
             "검증 = 학습에 쓰지 않은 8개 에피소드, 50스텝(1.67초) 예측 vs 시연. 기준선 = 지금 자세 그대로.", "",
             "| 스텝 | 검증 팔 오차(도) | 기준선 | 50스텝째 | 학습 에피소드 팔 오차 | 그리퍼 오차(mm) | 기준선 | 팔 방향 일치 | 이동량 비 | 그리퍼 방향 일치 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = " ★" if r["step"] == best_step else ""
        lines.append(f"| {r['step']}{mark} | {r['val_arm']:.2f} | {r['hold_arm']:.2f} | {r['val_arm_50']:.2f} | {r['train_arm']:.2f} | "
                     f"{r['val_grip']:.2f} | {r['hold_grip']:.2f} | {r['dir_ok']:.0%} | {r['move_ratio']:.2f} | {r['grip_dir']:.0%} |")
    (EV / f"{args.exp}_report.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = [r["step"] for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(s, [r["val_arm"] for r in rows], "o-", label="val model"); ax[0].plot(s, [r["train_arm"] for r in rows], "s--", label="train model")
        ax[0].plot(s, [r["hold_arm"] for r in rows], ":", c="gray", label="hold baseline"); ax[0].set_title("arm error (deg, 50-step mean)"); ax[0].legend()
        ax[1].plot(s, [r["val_grip"] for r in rows], "o-", label="val model"); ax[1].plot(s, [r["hold_grip"] for r in rows], ":", c="gray", label="hold")
        ax[1].set_title("gripper error (mm)"); ax[1].legend()
        ax[2].plot(s, [r["dir_ok"] for r in rows], "o-", label="arm direction ok"); ax[2].plot(s, [r["grip_dir"] for r in rows], "s-", label="gripper direction ok")
        ax[2].set_ylim(0, 1.05); ax[2].set_title("direction agreement (val)"); ax[2].legend()
        for a in ax:
            a.set_xlabel("step")
        fig.tight_layout(); fig.savefig(EV / f"{args.exp}_trend.png", dpi=70)
    except Exception as ex:  # noqa: BLE001
        print("plot skipped:", ex, file=sys.stderr)
    return v


env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", TRAIN_CONFIG=args.config)
# 없으면 local/... 데이터셋을 허브에서 찾다가 404 로 실패한다 (2026-11-11 10000 스텝 검증 실패 원인)
env.setdefault("HF_LEROBOT_HOME", str(ROOT / "datasets"))
print(f"감시 시작: {CK} (매 {args.period} 스텝)", flush=True)
while True:
    done = []
    if CK.exists():
        for d in sorted(CK.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
            if not d.name.isdigit() or int(d.name) % args.period or str(int(d.name)) in summary:
                continue
            ready = (d / "_CHECKPOINT_METADATA").exists() if args.kind == "openpi" else \
                (d / "pretrained_model/adapter_model.safetensors").exists() and (d / "training_state").exists()
            if not ready:
                continue
            step = str(int(d.name))
            t0 = time.time()
            print(f"[{time.strftime('%H:%M')}] 스텝 {d.name} 검증 시작", flush=True)
            if args.kind == "openpi":
                r = subprocess.run([sys.executable, str(ROOT / "finetune/eval_val.py"), str(d), "--cpu", "--every", str(args.every)],
                                   env=env, capture_output=True, text=True)
            else:  # GPU 여유(학습 약 12GB)가 있어 GPU 로 검증, 영상용 예측도 함께 만든다
                r = subprocess.run([str(LEROBOT_PY), str(ROOT / "finetune/eval_lerobot.py"), str(d / "pretrained_model"),
                                    args.exp, step, "--device", "cuda", "--every", str(args.every)],
                                   capture_output=True, text=True, cwd=str(ROOT / "lerobot_pi05"))
            js = EV / f"{args.exp}_{step}.json"
            if r.returncode or not js.exists():
                print(f"스텝 {step} 검증 실패:\n{r.stderr[-1500:]}", flush=True)
                summary[step] = None
            else:
                summary[step] = row(int(d.name), json.loads(js.read_text()))
                v = report([x for x in summary.values() if x])
                x = summary[step]
                print(f"[{time.strftime('%H:%M')}] 스텝 {d.name} ({(time.time() - t0) / 60:.0f}분): 검증 팔 {x['val_arm']:.2f}도 "
                      f"(기준선 {x['hold_arm']:.2f}), 그리퍼 {x['val_grip']:.2f}mm, 팔 방향 {x['dir_ok']:.0%}, 그리퍼 방향 {x['grip_dir']:.0%} → {v}", flush=True)
            SUM.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
            # 검증 영상 (같은 검증 에피소드, CPU 추론 → MuJoCo 렌더링)
            if summary[step]:
                t1 = time.time()
                npz = EV / f"{args.exp}_{step}_val.npz"; mp4 = EV / f"{args.exp}_{step}_val.mp4"
                if args.kind == "openpi":
                    r1 = subprocess.run([sys.executable, str(ROOT / "finetune/val_video_infer.py"), str(d), str(npz), "--cpu"],
                                        env=env, capture_output=True, text=True)
                else:  # eval_lerobot.py 가 이미 만들었다
                    r1 = subprocess.CompletedProcess([], 0 if npz.exists() else 1, "", "영상 예측 npz 없음")
                r2 = r1.returncode == 0 and subprocess.run(
                    [str(pathlib.Path.home() / "towel_sim/.venv/bin/python"), str(ROOT / "finetune/val_video_render.py"),
                     str(npz), str(mp4), f"step {step}"], capture_output=True, text=True)
                if r1.returncode or not r2 or r2.returncode:
                    print(f"스텝 {d.name} 영상 실패:\n{(r1.stderr if r1.returncode else r2.stderr)[-1200:]}", flush=True)
                else:
                    print(f"[{time.strftime('%H:%M')}] 스텝 {d.name} 검증 영상 ({(time.time() - t1) / 60:.0f}분): {mp4}", flush=True)
            done.append(d.name)
    good = [x for x in summary.values() if x]
    v, best = (verdict(good)[:2] if good else (None, None))
    try:
        report_html.write(EV, args.exp, good, v, best, args.period, kind=args.kind, title=TITLE)
    except Exception as ex:  # noqa: BLE001
        print("dashboard 갱신 실패:", ex, flush=True)
    if not done:
        time.sleep(60)
