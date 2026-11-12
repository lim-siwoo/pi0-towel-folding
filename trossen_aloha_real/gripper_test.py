"""그리퍼 단독 시험: 준비 자세에서 양팔 그리퍼를 완전히 열었다가 닫는다 (정책 없음).

순서: 연결·상태 확인 -> 'go' 확인 -> 준비 자세(5초) -> 열기 40mm(1초) -> 유지 -> 닫기 0mm(1초) -> 유지
      -> 다시 열기 -> 준비 자세 -> 휴식 자세(5초) -> 모터 해제
각 단계에서 측정한 그리퍼 열림(mm)을 출력한다. Ctrl-C 면 그 자리에서 고정 후 기본 자세로 복귀.

    cd ~/pi0_aloha_towel/trossen_openpi/examples/trossen_ai
    .venv/bin/python ~/pi0_aloha_towel/trossen_aloha_real/gripper_test.py
"""

import json
import math
import pathlib
import time

STATION = json.loads((pathlib.Path.home() / "trossen_setup/stationary-teleop.json").read_text())["robot"]
ARMS = (("left", STATION["left_arm_ip_address"]), ("right", STATION["right_arm_ip_address"]))
STAGED = [0.0, math.pi / 3, math.pi / 6, math.pi / 5, 0.0, 0.0]
OPEN, CLOSED = 0.04, 0.0


def main():
    import trossen_arm as t
    drivers = {}
    for side, ip in ARMS:
        d = t.TrossenArmDriver()
        d.configure(t.Model.wxai_v0, t.StandardEndEffector.wxai_v0_follower, ip, False, 5.0)
        err = d.get_error_information()
        if err != "No error":
            raise SystemExit(f"{side} 팔 드라이버 오류: {err}")
        drivers[side] = d

    def grip_mm():
        return {s: round(d.get_all_positions()[6] * 1000, 1) for s, d in drivers.items()}

    def move(grip, seconds, label):
        print(f"[{label}] 그리퍼 목표 {grip * 1000:.0f}mm · {seconds:.0f}초", flush=True)
        for d in drivers.values():
            d.set_all_modes(t.Mode.position)
            d.set_all_positions(STAGED + [grip], goal_time=seconds, blocking=False)
        time.sleep(seconds + 0.3)

    def hold_and_report(label, seconds=1.5):
        time.sleep(seconds)
        print(f"   측정 열림 ({label}): {grip_mm()}", flush=True)

    print("\n" + "=" * 60)
    print(" 그리퍼 시험: 준비 자세로 이동한 뒤 그리퍼를 열고 닫습니다.")
    print("  - 그리퍼 근처에 손가락을 두지 마세요 (최대 100N).")
    print("  - 중단: Ctrl-C 또는 물리 비상정지")
    print("=" * 60)
    print("현재 그리퍼 열림:", grip_mm())
    if input("계속하려면 'go' 입력: ").strip() != "go":
        print("취소했습니다. 팔은 움직이지 않았습니다.")
        for d in drivers.values():
            d.cleanup()
        return
    try:
        cur = {s: d.get_all_positions()[6] for s, d in drivers.items()}
        move(max(cur.values()), 5.0, "준비 자세")
        move(OPEN, 1.0, "열기")
        hold_and_report("열림")
        move(CLOSED, 1.0, "닫기")
        hold_and_report("닫힘")
        move(OPEN, 1.0, "다시 열기")
        hold_and_report("열림")
    except KeyboardInterrupt:
        print("\n[중단] 그 자리에서 고정합니다.")
        for d in drivers.values():
            q = list(d.get_all_positions())
            d.set_all_modes(t.Mode.position)
            d.set_all_positions(q, goal_time=0.1, blocking=False)
    finally:
        for label, pose in (("준비", STAGED + [OPEN]), ("휴식", [0.0] * 7)):
            print(f"[복귀] {label} 자세 · 5초", flush=True)
            for d in drivers.values():
                d.set_all_modes(t.Mode.position)
                d.set_all_positions(pose, goal_time=5.0, blocking=False)
            time.sleep(5.2)
        for d in drivers.values():
            d.set_all_modes(t.Mode.idle)
            d.cleanup()
        print("완료.")


if __name__ == "__main__":
    main()
