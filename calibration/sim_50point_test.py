"""실제 실행 코드(env.TrossenAlohaEnvironment)의 관절 변환을 시뮬레이터 50점으로 검증한다.

  1) ALOHA -> Trossen (apply_action 경로): PI 데이터의 ALOHA 자세 50개를 ALOHA 1 시뮬레이터에 넣은 손가락 끝과,
     env.apply_action 이 만든 Trossen 관절을 Trossen 시뮬레이터에 넣은 손가락 끝을 작업대 좌표계에서 비교.
  2) Trossen -> ALOHA (관측 경로): 실제 실행의 Trossen 자세 50개를 Trossen 시뮬레이터에 넣은 손가락 끝과,
     env._native_to_aloha 가 만든 ALOHA 관절을 ALOHA 1 시뮬레이터에 넣은 손가락 끝을 비교.
수식만 검증하기 위해 Trossen 책상은 시뮬레이터 책상(평평, 베이스 -19.1mm)으로 둔다.
depth-shift 는 0 과 50mm 두 경우를 본다 (50mm 면 '앞쪽(-v)으로 50mm' 차이가 나야 정상).
"""

import logging
import pathlib
import sys
import types

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trossen_aloha_real"))
sys.path.insert(0, str(HERE))
logging.disable(logging.CRITICAL)
import retarget as R  # noqa: E402
import sim_transform_test as ST  # noqa: E402  (ALOHA 1 / Trossen 시뮬레이터, 작업대 좌표계)

KEYS = [f"{s}_{j}.pos" for s in ("left", "right") for j in
        ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "left_carriage_joint")]


class FakeArm:
    def __init__(self, robot, off): self.robot, self.off = robot, off
    def get_observation(self):
        return {f"{j}.pos": self.robot.q[self.off + i] for i, j in enumerate(
            ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "left_carriage_joint"))}


class FakeRobot:
    action_features = {k: float for k in KEYS}
    def __init__(self):
        self.q = np.array(list(R.WXAI_STAGED) + [0.02] + list(R.WXAI_STAGED) + [0.02])
        self.left_arm, self.right_arm, self.cameras = FakeArm(self, 0), FakeArm(self, 7), {}
    def connect(self): pass
    def send_action(self, a): self.q = np.array([a[k] for k in KEYS]); return a


import lerobot.robots as lr  # noqa: E402
lr.make_robot_from_config = lambda cfg: FakeRobot()
import env as E  # noqa: E402


def make_env(depth_mm):
    env = E.TrossenAlohaEnvironment(dry_run=False, depth_shift_m=depth_mm / 1000, workspace=False)
    sim_table = ST.T_RIG[0][2] - ST.T_BASE["left"][2]
    for side in ("left", "right"):
        env._cal.params[side] = (np.array([0.0, 0.0, sim_table]), -0.0010)
    return env


rng = np.random.default_rng(1)
pi = np.load(HERE / "pi_aloha_pen/states.npz")["state"]
aloha_pts = pi[rng.choice(len(pi), 50, replace=False)].astype(float)
run = np.load(pathlib.Path.home() / "pi0_aloha_towel/trossen_aloha_real/runs/20261105-110453/log.npz")["native_cmd"]
trossen_pts = run[rng.choice(len(run), 50, replace=False)].astype(float)


def report(name, errs):
    e = np.array(errs) * 1000
    n = np.linalg.norm(e, axis=1)
    print(f"{name}: n={len(e)}  평균 오차(u,v,z) {np.round(e.mean(0), 1)} mm  |오차| 중앙 {np.median(n):.1f} 최대 {n.max():.1f} mm")


for depth in (0, 50):
    env = make_env(depth)
    # 1) ALOHA -> Trossen : apply_action 을 그대로 쓴다 (직전 명령 = 같은 자세에서 IK 시작)
    errs = []
    for a in aloha_pts:
        a14 = a.copy()
        env._last_aloha = None
        # 시작 자세: 이 ALOHA 자세를 대략 흉내 낸 Trossen 자세 (IK 시작값)
        for side, off in (("left", 0), ("right", 7)):
            p, rot = R.fk(a14[off:off + 6], native=False)
            q0, _ = E._approx_ik(env._cal.to_native(side, p, rot), rot, R.WXAI_STAGED, max_pos=1, max_rot_deg=180)
            env._last_cmd[off:off + 6] = q0
        env.apply_action({"actions": a14})
        qn = env._last_cmd
        for side, off in (("left", 0), ("right", 7)):
            errs.append(ST.trossen_tip(side, qn[off:off + 6]) - ST.aloha_tip(side, a14[off:off + 6]))
    report(f"[depth {depth}mm] ALOHA→Trossen (apply_action, 50자세 x 2팔)", errs)
    # 2) Trossen -> ALOHA : _native_to_aloha
    errs = []
    for q in trossen_pts:
        al = env._native_to_aloha(q)
        for side, off in (("left", 0), ("right", 7)):
            errs.append(ST.aloha_tip(side, al[off:off + 6].astype(float)) - ST.trossen_tip(side, q[off:off + 6]))
    report(f"[depth {depth}mm] Trossen→ALOHA (관측, 50자세 x 2팔)", errs)
