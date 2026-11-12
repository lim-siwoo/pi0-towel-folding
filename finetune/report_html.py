"""학습 대시보드 HTML (finetune/eval/dashboard.html) 생성. eval_watch.py 가 1분마다 부른다.

내용: 판정, 학습 진행(스텝·속도·남은 시간), 학습 손실 곡선, 체크포인트별 검증 표·추세 그림,
최근 체크포인트의 검증 예측 대 시연 궤적. 그림은 PNG 를 HTML 안에 넣어 파일 하나로 열린다.
"""
import base64
import datetime as dt
import glob
import html
import io
import json
import pathlib
import re

ROOT = pathlib.Path.home() / "pi0_aloha_towel"
TRAIN_LOG = ROOT / "finetune/train.log"
OPENPI = ROOT / "trossen_openpi"
TOTAL = 150_000
LEROBOT_LOG = ROOT / "finetune/train_pi05.log"
LEROBOT_TOTAL = 30_000
_LR = re.compile(r"INFO \d{4}-\d\d-\d\d (\d\d:\d\d:\d\d) \S+ step:([\d.]+)(K?) .*?smpl:([\d.]+)(K?) .*?loss:([\d.]+).*?step_s:([\d.]+)")
_TQDM = re.compile(r"Training:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[")


def _num(v, k):
    return float(v) * (1000 if k else 1)


def lerobot_lines():
    try:
        txt = LEROBOT_LOG.read_bytes().decode("utf-8", "ignore").replace("\r", "\n")
    except FileNotFoundError:
        return [], ""
    rows = [(t, _num(sv, sk), _num(nv, nk), float(l), float(r)) for t, sv, sk, nv, nk, l, r in _LR.findall(txt)]
    # 스텝: 로그는 1K 단위로 반올림되므로 tqdm 진행(이어 학습이면 남은 스텝 기준)으로 정확히 계산
    m = None
    for m in _TQDM.finditer(txt[-200000:]):
        pass
    exact = (LEROBOT_TOTAL - int(m.group(2)) + int(m.group(1))) if m else None
    return rows, txt[-20000:], exact


def training_status(kind="openpi"):
    """학습 로그 마지막 진행: 스텝, 초/스텝, 마지막 갱신 시각."""
    if kind == "lerobot":
        rows, tail, exact = lerobot_lines()
        if not rows:
            return None
        t, step, _, _, rate = rows[-1]
        return {"time": t, "step": int(exact if exact is not None else step), "rate": float(rate), "remaining": "",
                "total": LEROBOT_TOTAL,
                "error": "Traceback" in tail or "CUDA out of memory" in tail}
    try:
        tail = TRAIN_LOG.read_bytes()[-20000:].decode("utf-8", "ignore").replace("\r", "\n")
    except FileNotFoundError:
        return None
    m = None
    for m in re.finditer(r"(\d\d:\d\d:\d\d)\.\d+ \[I\] Progress on: ([\d.]+)(k?)it/150kit rate:([\d.]+)s/it remaining:([\d:]+)", tail):
        pass
    if not m:
        return None
    step = float(m.group(2)) * (1000 if m.group(3) else 1)
    err = "Traceback" in tail or "RESOURCE_EXHAUSTED" in tail
    return {"time": m.group(1), "step": int(step), "rate": float(m.group(4)), "remaining": m.group(5), "error": err,
            "total": TOTAL}


def loss_history(kind="openpi"):
    if kind == "lerobot":
        # 손실 기록 스텝 = 본 샘플 수 / batch 8 (로그가 K 단위로 반올림돼 근사)
        return [(int(smpl / 8), l) for _, _, smpl, l, _ in lerobot_lines()[0]]
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal import datastore
    except ImportError:
        return []
    runs = sorted(glob.glob(str(OPENPI / "wandb/offline-run-*/run-*.wandb")), key=lambda p: pathlib.Path(p).stat().st_mtime)
    if not runs:
        return []
    rows = []
    try:
        ds = datastore.DataStore(); ds.open_for_scan(runs[-1])
        while True:
            data = ds.scan_data()
            if data is None:
                break
            rec = wandb_internal_pb2.Record(); rec.ParseFromString(data)
            if rec.WhichOneof("record_type") == "history":
                d = {it.key: it.value_json for it in rec.history.item}
                if "loss" in d:
                    rows.append((int(d["_step"]), float(json.loads(d["loss"]))))
    except Exception:  # noqa: BLE001  (학습 중 쓰는 파일이라 끝이 잘려 있을 수 있음)
        pass
    return rows


def _png(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def _loss_png(loss):
    if len(loss) < 2:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    s, v = np.array(loss).T
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(s, v, lw=0.8, alpha=0.4, color="#4c78a8")
    if len(v) > 10:
        k = 10
        ax.plot(s[k - 1:], np.convolve(v, np.ones(k) / k, "valid"), lw=1.6, color="#4c78a8", label="moving avg (10)")
        ax.legend()
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title("training loss (train episodes only)")
    ax.grid(alpha=0.3)
    s_ = _png(fig); plt.close(fig)
    return s_


def write(ev_dir, exp, rows, verdict, best_step, period, kind="openpi", title=""):
    st = training_status(kind)
    loss = loss_history(kind)
    rows = sorted(rows, key=lambda r: r["step"])
    trend = ev_dir / f"{exp}_trend.png"
    last_traj = ev_dir / f"{exp}_{rows[-1]['step']}.png" if rows else None
    best_traj = ev_dir / f"{exp}_{best_step}.png" if best_step else None
    b64 = lambda p: base64.b64encode(p.read_bytes()).decode() if p and p.exists() else None  # noqa: E731
    lp = _loss_png(loss)

    if st:
        total = st["total"]
        pct = st["step"] / total * 100
        eta = dt.datetime.now() + dt.timedelta(seconds=(total - st["step"]) * st["rate"])
        next_eval = ((st["step"] // period) + 1) * period
        prog = (f"<div class='bar'><div style='width:{pct:.2f}%'></div></div>"
                f"<p><b>{st['step']:,}</b> / {total:,} 스텝 ({pct:.1f}%) · {st['rate']:.2f}초/스텝 · "
                f"끝나는 예상 시각 <b>{eta:%m/%d %H:%M}</b> · 다음 검증 {next_eval:,} 스텝 "
                f"(약 {dt.datetime.now() + dt.timedelta(seconds=(next_eval - st['step']) * st['rate']):%H:%M})"
                f" · 로그 갱신 {st['time']}</p>")
        if st["error"]:
            prog += "<p class='bad'>학습 로그에 오류가 있습니다. 터미널을 확인하세요.</p>"
    else:
        prog = "<p>학습 로그를 아직 읽지 못했습니다.</p>"

    head = ["스텝", "검증 팔 오차(도)", "기준선", "50스텝째", "학습 에피소드 팔", "그리퍼(mm)", "기준선", "팔 방향 일치", "이동량 비", "그리퍼 방향 일치"]
    trs = []
    for r in rows:
        good = r["val_arm"] < 0.85 * r["hold_arm"]
        trs.append("<tr" + (" class='best'" if r["step"] == best_step else "") + ">" + "".join(
            f"<td>{c}</td>" for c in [
                f"{r['step']:,}" + (" ★" if r["step"] == best_step else ""),
                f"<span class='{'ok' if good else 'warn'}'>{r['val_arm']:.2f}</span>", f"{r['hold_arm']:.2f}",
                f"{r['val_arm_50']:.2f}", f"{r['train_arm']:.2f}", f"{r['val_grip']:.2f}", f"{r['hold_grip']:.2f}",
                f"{r['dir_ok']:.0%}", f"{r['move_ratio']:.2f}", f"{r['grip_dir']:.0%}"]) + "</tr>")
    table = ("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
             + ("".join(trs) or f"<tr><td colspan='{len(head)}'>첫 검증은 {period:,} 스텝 체크포인트가 저장된 뒤 실행됩니다.</td></tr>")
             + "</tbody></table>")

    vids = []
    for r in reversed(rows):
        mp4 = ev_dir / f"{exp}_{r['step']}_val.mp4"
        if mp4.exists():
            tag = " ★ 최저 오차" if r["step"] == best_step else ""
            vids.append(f"<details{' open' if not vids else ''}><summary>스텝 {r['step']:,}{tag}</summary>"
                        f"<video controls preload='metadata' src='{mp4.name}' style='width:100%'></video></details>")
    vids = "".join(vids) or "<p class='muted'>아직 없음 (검증 직후 만들어집니다)</p>"
    img = lambda b, alt: f"<img alt='{alt}' src='data:image/png;base64,{b}'>" if b else "<p class='muted'>아직 없음</p>"  # noqa: E731
    vclass = "bad" if verdict and verdict.startswith("중단") else "ok"
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="300">
<title>수건 LoRA 학습</title><style>
:root{{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e0;--accent:#4c78a8;--ok:#2e7d32;--warn:#b26a00;--bad:#c62828;--best:#eef5ff}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#151517;--card:#1f1f22;--fg:#ececee;--muted:#9a9aa0;--line:#333338;--accent:#7aa6d6;--ok:#66bb6a;--warn:#ffb74d;--bad:#ef5350;--best:#1d2a3a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Noto Sans KR",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:20px 16px 48px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:0 0 10px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0;overflow-x:auto}}
.muted{{color:var(--muted)}} .ok{{color:var(--ok)}} .warn{{color:var(--warn)}} .bad{{color:var(--bad)}}
.verdict{{font-size:17px;font-weight:600}}
.bar{{height:10px;background:var(--line);border-radius:5px;overflow:hidden}} .bar div{{height:100%;background:var(--accent)}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
tr.best td{{background:var(--best)}}
img{{max-width:100%;height:auto;background:#fff;border-radius:6px}}
ul{{margin:6px 0;padding-left:20px}} details{{margin:6px 0}} summary{{cursor:pointer;font-weight:600}} video{{border-radius:6px;background:#000}}
</style></head><body><main>
<h1>수건 접기 LoRA 학습 — {html.escape(exp)}</h1>
<p class="muted">{html.escape(title or 'π0 base + LoRA (openpi, pi0_trossen_towel_cc)')} · 학습 148 에피소드 / 검증 8 에피소드(학습에 쓰지 않음) · 이 페이지 생성 {dt.datetime.now():%m/%d %H:%M:%S} (5분마다 새로고침)</p>
<div class="card"><h2>판정</h2><p class="verdict {vclass}">{html.escape(verdict or '첫 검증 전')}</p>
<ul class="muted"><li>"학습됨" 기준: 검증 팔 오차가 기준선(지금 자세 그대로)보다 15% 이상 작음</li>
<li>계속: 검증 오차가 지금까지 최저보다 2% 넘게 좋아짐 · 과적합 의심: 검증 오차가 연속 2번 나빠지는데 학습 에피소드 오차만 좋아짐 · 정체: 연속 3번(15,000 스텝) 개선 없음</li>
<li>방향 일치: 팔 이동 방향 코사인 &gt; 0.7 비율, 그리퍼 열고 닫는 방향이 시연과 같은 비율. 이동량 비 1.0 = 시연만큼 움직임</li></ul></div>
<div class="card"><h2>학습 진행</h2>{prog}{img(lp, 'loss')}</div>
<div class="card"><h2>체크포인트별 검증 ({period:,} 스텝마다)</h2>{table}<p class="muted">오차 = 50스텝(1.67초) 예측 전체 평균. ★ = 검증 팔 오차 최저(실물 테스트 추천 체크포인트)</p></div>
<div class="card"><h2>추세</h2>{img(b64(trend), 'trend')}</div>
<div class="card"><h2>검증 영상</h2><p class="muted">학습에 쓰지 않은 검증 에피소드(매번 같은 에피소드). 위: 실제 시연 카메라 · 아래 왼쪽: 시뮬레이터에서 불투명 = 시연 동작, 빨간 반투명 = 모델 예측(0.5초마다 다시 예측해 이어 붙임) · 아래 오른쪽: 그리퍼 열림(검정 시연, 빨강 모델). 입력은 시연 당시 영상이라 실제 로봇이 이 예측대로 움직였을 때와는 다를 수 있습니다.</p>
{vids}</div>
<div class="card"><h2>최근 체크포인트 {rows[-1]['step'] if rows else ''}: 검증 예측(빨강) 대 시연(검정)</h2>{img(b64(last_traj), 'latest')}</div>
{f"<div class='card'><h2>최저 오차 체크포인트 {best_step}</h2>{img(b64(best_traj), 'best')}</div>" if best_step and rows and best_step != rows[-1]['step'] else ''}
</main></body></html>"""
    (ev_dir / ("dashboard.html" if kind == "openpi" else f"dashboard_{exp}.html")).write_text(page)
