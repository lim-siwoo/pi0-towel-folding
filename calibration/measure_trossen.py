"""2단계: Trossen(WidowX AI) 팔로워 실측 캘리브레이션.

팔을 한 쪽씩 '손으로 움직일 수 있는 상태(외력 0, 중력 보상)'로 풀고, 사람이 그리퍼를 책상에 대 가며
Enter 로 관절값을 기록한다. 기록이 끝난 팔은 그 자리에서 다시 고정한다. 마지막에 양팔을 준비 -> 휴식
자세로 천천히 돌려보낸다.

측정 항목 (팔마다):
  1) table  : 그리퍼를 닫고 손가락 끝을 책상에 댄 점 (4곳 이상, 넓게 흩어서)  -> 책상 평면
  2) grasp  : 책상 위 얇은 물건을 집는 자세 (위에서 아래로, 손가락 끝이 책상에 닿게, 2곳 이상)
             -> 이 로봇 그리퍼 기준점의 '집는 높이' (ALOHA 의 약 42mm 와 비교)
  3) common : 책상에 표시한 공통 지점 p1, p2, ... 에 손가락 끝을 댄 점 (양팔이 같은 지점)
             -> 두 팔 베이스의 상대 위치/방향

실행 (관제센터·정책 클라이언트가 팔을 쓰고 있지 않을 때, 로봇 앞에서):
    cd ~/pi0_aloha_towel/trossen_openpi/examples/trossen_ai
    .venv/bin/python ~/pi0_aloha_towel/calibration/measure_trossen.py
기록만 다시 계산:
    .venv/bin/python ~/pi0_aloha_towel/calibration/measure_trossen.py --compute-only

입력: Enter = 기록, u = 마지막 기록 취소, n = 다음 항목, q = 중단(팔 고정 후 기본 자세 복귀)
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trossen_aloha_real"))
import retarget as R  # noqa: E402

STATION = json.loads((pathlib.Path.home() / "trossen_setup/stationary-teleop.json").read_text())["robot"]
ARMS = (("left", STATION["left_arm_ip_address"]), ("right", STATION["right_arm_ip_address"]))
STAGED = [0.0, math.pi / 3, math.pi / 6, math.pi / 5, 0.0, 0.0, 0.0]
RAW_FILE = HERE / "trossen_measurements_raw.json"
OUT_FILE = HERE / "trossen_measurements.json"
ALOHA_GRASP_Z_M = 0.042  # 1단계: PI ALOHA 펜 집기, 그리퍼 기준점 최저 높이 중앙값
HOME_S = 5.0
SECTIONS = (
    ("table", 4, "그리퍼를 닫고, 손가락 끝을 책상에 살짝 대세요 (작업 영역 안에서 넓게 흩어서 {n}곳 이상)"),
    ("grasp", 2, "책상 위 얇은 물건을 집는 자세로 놓으세요: 그리퍼가 위에서 아래로, 손가락 끝이 책상에 닿게 ({n}곳 이상)"),
    ("common", 3, "책상에 표시한 공통 지점에 손가락 끝을 대세요. 두 팔 모두 같은 순서로 (p1, p2, ... {n}곳 이상)"),
)


# ------------------------------------------------------------------ robot
class Arm:
    def __init__(self, side, ip):
        import trossen_arm as t
        self.t, self.side = t, side
        self.d = t.TrossenArmDriver()
        self.d.configure(t.Model.wxai_v0, t.StandardEndEffector.wxai_v0_follower, ip, False, 5.0)
        if self.d.get_error_information() != "No error":
            raise RuntimeError(f"{side} 팔 드라이버 오류: {self.d.get_error_information()}")

    def q(self):
        q = np.asarray(self.d.get_all_positions(), dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError(f"{self.side} 관절값을 읽지 못했습니다")
        return q

    def hold(self):
        """현재 자세 그대로 위치 제어로 고정."""
        q = list(self.q())
        self.d.set_all_modes(self.t.Mode.position)
        self.d.set_all_positions(q, goal_time=0.1, blocking=False)

    def free(self):
        """외력 0 (중력 보상) — 손으로 움직일 수 있는 상태."""
        self.d.set_all_modes(self.t.Mode.external_effort)
        self.d.set_all_external_efforts([0.0] * 7, goal_time=0.0, blocking=False)

    def sample(self, seconds=0.3, n=15):
        qs = []
        for _ in range(n):
            qs.append(self.q())
            time.sleep(seconds / n)
        qs = np.array(qs)
        return qs.mean(axis=0), float(np.abs(qs - qs.mean(axis=0))[:, :6].max())


def go_home(arms):
    for label, pose in (("준비", STAGED), ("휴식", [0.0] * 7)):
        print(f"[home] 양팔 {label} 자세로 이동 · {HOME_S:.0f}초", flush=True)
        for a in arms:
            a.d.set_all_modes(a.t.Mode.position)
            a.d.set_all_positions(pose, goal_time=HOME_S, blocking=False)
        time.sleep(HOME_S + 0.2)


def record_arm(arm, raw):
    print("\n" + "=" * 64)
    print(f" {arm.side} 팔 측정. 이 팔을 손으로 받쳐 잡으세요. 힘이 풀리면 무게를 손으로 지탱해야 합니다.")
    print("=" * 64)
    if input(f"{arm.side} 팔을 잡았으면 'go' 입력: ").strip() != "go":
        raise KeyboardInterrupt
    arm.free()
    print(f"[{arm.side}] 손으로 움직일 수 있습니다.")
    data = raw.setdefault(arm.side, {})
    for name, minimum, guide in SECTIONS:
        pts = data.setdefault(name, [])
        print(f"\n--- {name}: {guide.format(n=minimum)}")
        while True:
            label = f"p{len(pts) + 1}" if name == "common" else f"#{len(pts) + 1}"
            cmd = input(f"[{arm.side}/{name}] {label} 위치에 대고 Enter (u=취소, n=다음, q=중단): ").strip().lower()
            if cmd == "q":
                raise KeyboardInterrupt
            if cmd == "u":
                if pts:
                    print("   취소:", pts.pop()["label"])
                continue
            if cmd == "n":
                if len(pts) < minimum:
                    print(f"   최소 {minimum}곳이 필요합니다 (지금 {len(pts)}곳)")
                    continue
                break
            q, jitter = arm.sample()
            p = R.fk(q[:6], native=True)[0]
            pts.append({"label": label, "q": q.round(6).tolist(), "jitter_rad": round(jitter, 5)})
            warn = "  ← 움직이는 중이었음, 다시 기록 권장(u)" if jitter > 0.01 else ""
            print(f"   기록 {label}: 그리퍼 기준점 (x, y, z) = {np.round(p * 1000, 1)} mm{warn}")
        RAW_FILE.write_text(json.dumps(raw, indent=1, ensure_ascii=False))
    arm.hold()
    print(f"[{arm.side}] 측정 완료 · 그 자리에 고정했습니다.")


# ------------------------------------------------------------------ compute
def kabsch(a, b):
    """b ≈ R a + t 인 회전 R, 평행이동 t (점 3개 이상)."""
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1, 1, d]) @ u.T
    return rot, cb - rot @ ca


def compute(raw):
    out = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
           "frame": "각 WidowX AI 팔 베이스 좌표계, retarget.fk(native=True) 그리퍼 기준점, mm",
           "aloha_grasp_z_mm": ALOHA_GRASP_Z_M * 1000, "arms": {}}
    for side in ("left", "right"):
        d = raw.get(side, {})
        P = {k: np.array([R.fk(np.array(r["q"][:6]), native=True)[0] for r in d.get(k, [])]).reshape(-1, 3)
             for k in ("table", "grasp", "common")}
        arm = {}
        if len(P["table"]) >= 3:
            A = np.c_[P["table"][:, :2], np.ones(len(P["table"]))]
            coef, *_ = np.linalg.lstsq(A, P["table"][:, 2], rcond=None)
            resid = P["table"][:, 2] - A @ coef
            center = P["table"][:, :2].mean(0)
            arm["table_plane"] = {
                "z_at_points_center_mm": round(float(coef[0] * center[0] + coef[1] * center[1] + coef[2]) * 1000, 1),
                "tilt_deg": round(math.degrees(math.atan(math.hypot(coef[0], coef[1]))), 2),
                "rms_mm": round(float(np.sqrt(np.mean(resid ** 2))) * 1000, 1),
                "points_mm": np.round(P["table"] * 1000, 1).tolist()}
        if len(P["grasp"]):
            zg = float(np.median(P["grasp"][:, 2]))
            arm["grasp_z_mm"] = round(zg * 1000, 1)
            arm["grasp_points_mm"] = np.round(P["grasp"] * 1000, 1).tolist()
            arm["suggested_dz_mm"] = round((zg - ALOHA_GRASP_Z_M) * 1000, 1)
        arm["common_points_mm"] = np.round(P["common"] * 1000, 1).tolist()
        out["arms"][side] = arm
    cl, cr = (np.array(out["arms"][s]["common_points_mm"]) / 1000 for s in ("left", "right"))
    n = min(len(cl), len(cr))
    if n >= 3:
        rot, t = kabsch(cr[:n], cl[:n])  # 오른팔 좌표 -> 왼팔 좌표
        err = np.linalg.norm((rot @ cr[:n].T).T + t - cl[:n], axis=1)
        out["right_base_in_left_frame"] = {
            "translation_mm": np.round(t * 1000, 1).tolist(),
            "distance_mm": round(float(np.linalg.norm(t)) * 1000, 1),
            "yaw_deg": round(math.degrees(math.atan2(rot[1, 0], rot[0, 0])), 2),
            "fit_error_mm": np.round(err * 1000, 1).tolist()}
    dzs = [out["arms"][s].get("suggested_dz_mm") for s in ("left", "right") if "suggested_dz_mm" in out["arms"][s]]
    if dzs:
        out["suggested_dz_mm"] = round(float(np.mean(dzs)), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-only", action="store_true")
    ap.add_argument("--arms", default="left,right")
    args = ap.parse_args()
    raw = json.loads(RAW_FILE.read_text()) if RAW_FILE.exists() and args.compute_only else {}
    if not args.compute_only:
        sides = [s.strip() for s in args.arms.split(",")]
        arms = [Arm(side, ip) for side, ip in ARMS]
        try:
            for a in arms:
                a.hold()  # 시작 시 양팔 고정
            for a in arms:
                if a.side in sides:
                    record_arm(a, raw)
        except KeyboardInterrupt:
            print("\n[중단] 양팔을 그 자리에 고정합니다.")
        finally:
            for a in arms:
                try:
                    a.hold()
                except Exception as e:  # noqa: BLE001
                    print("고정 실패 — 비상정지를 누르세요:", e)
            try:
                go_home(arms)
            except BaseException as e:  # noqa: BLE001
                print("기본 자세 복귀 실패:", e)
            for a in arms:
                try:
                    a.d.set_all_modes(a.t.Mode.idle)
                    a.d.cleanup()
                except Exception:  # noqa: BLE001
                    pass
            RAW_FILE.write_text(json.dumps(raw, indent=1, ensure_ascii=False))
    out = compute(raw)
    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n" + json.dumps({k: v for k, v in out.items() if k != "arms"}, ensure_ascii=False, indent=1))
    for s, a in out["arms"].items():
        print(s, {k: v for k, v in a.items() if not k.endswith("points_mm")})
    print("저장:", OUT_FILE)


if __name__ == "__main__":
    main()
