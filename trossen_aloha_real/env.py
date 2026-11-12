"""openpi examples/aloha_real/env.py 를 Trossen AI Stationary(WidowX AI) 용으로 옮긴 환경.

upstream 과 같은 인터페이스(openpi_client Environment)와 같은 관측/행동 형식을 쓴다.
  - 관측 state : ALOHA(ViperX 300 S) 관절 6개 + 그리퍼(0 닫힘 ~ 1 열림, 선형 위치) x 2
  - 관측 images: cam_high / cam_low / cam_left_wrist / cam_right_wrist, 224x224 패딩 리사이즈, uint8 CHW
  - 행동 actions: ALOHA 절대 관절 6개 + 그리퍼(0 닫힘 ~ 1 열림, 정규화 관절각) x 2
다른 점은 하드웨어 계층뿐이다.
  - ROS/Interbotix 대신 lerobot BiWidowXAIFollower 로 팔·카메라를 다룬다.
  - ALOHA <-> WidowX AI 관절 변환(retarget.py: 팔 끝 자세 기준 FK/IK, 그리퍼 링크 기하 변환)을 거친다.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import List, Optional  # noqa: UP035

import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from typing_extensions import override

import retarget as R
from height_calib import HeightCalibration
from workspace import Workspace
import gripper_map as G
from viewpoint import ViewWarp
from collision import ArmCollisionGuard

STATION_FILE = pathlib.Path.home() / "trossen_setup/stationary-teleop.json"
CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
STAGED_NATIVE = [0.0, 1.0471975511965976, 0.5235987755982988, 0.6283185307179586, 0.0, 0.0, 0.0]  # lerobot staged_positions
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]  # upstream real_env.py 기본값 (ALOHA 관절)


# ALOHA reset_pose [0,-1.5,1.5,0,0,0] 은 WidowX AI 정상 자세 계열(베이스≈0)에서 가장 가까운 해가
# 약 20mm / 10도 차이 난다 (정확히 맞추는 해는 베이스 -90도 + 손목 -90도의 비정상 자세뿐).
MAX_APPROX_POS_M = 0.025   # 근사 해 허용 위치 오차
MAX_APPROX_ROT_DEG = 12.0  # 근사 해 허용 방향 오차
MAX_STALE_S = 2.0          # 카메라 새 프레임이 이 시간 넘게 없으면 정지 (그 전엔 마지막 프레임 사용)
HOME_MOVE_S = 5.0          # 종료·오류 시 준비 -> 휴식 자세 복귀, 단계별 이동 시간
MAX_IK_JUMP_RAD = 0.25     # 모델 동작 변화가 작을 때 변환 결과의 스텝당 허용 변화 (넘으면 IK 해 전환으로 보고 정지)
MAX_STEP_RAD = 0.6         # 모델이 크게 움직여도 한 스텝 명령 변화는 이 이하 (비정상 출력 정지)
REG_WEIGHT = 0.02          # 근사 해를 직전 자세 근처에 붙잡는 항의 가중치 (m/rad)


def _approx_ik(position, rotation, seed, *, native=True, max_pos=MAX_APPROX_POS_M, max_rot_deg=MAX_APPROX_ROT_DEG,
               reg=REG_WEIGHT):
    """ALOHA 자세를 도달 가능한 가장 가까운 WidowX AI 자세로 근사 변환.

    ALOHA 의 reset_pose [0,-1.5,1.5,0,0,0] 처럼 팔을 바짝 접은 자세는 WidowX AI 어깨 한계(0 rad)
    때문에 정확한 해가 없다(약 3mm 부족). 이때 최소 오차 해를 쓰되, seed 에서 멀리 떨어진 다른
    해(예: 베이스 -90도 + 손목으로 보정)로 빠지지 않도록 seed 와의 차이에 작은 벌점을 준다.
    위치 15mm / 방향 10도를 넘으면 거절한다.
    """
    lower, upper = (R.WXAI_LOWER, R.WXAI_UPPER) if native else (R.ALOHA_LOWER, R.ALOHA_UPPER)
    seed = np.clip(np.asarray(seed, float), lower + 1e-6, upper - 1e-6)

    def error(q):
        p, r = R.fk(q, native=native)
        return np.concatenate((p - position, 0.15 * Rotation.from_matrix(rotation.T @ r).as_rotvec(),
                               reg * (q - seed)))

    res = least_squares(error, seed, bounds=(lower, upper), max_nfev=300)
    p, r = R.fk(res.x, native=native)
    pos_err = float(np.linalg.norm(p - position))
    rot_err = float(np.degrees(Rotation.from_matrix(rotation.T @ r).magnitude()))
    if pos_err > max_pos or rot_err > max_rot_deg:
        raise R.RetargetError(
            f"{'WidowX AI' if native else 'ALOHA'} 로 변환할 수 없는 자세 "
            f"(가장 가까운 해 오차 {pos_err * 1000:.1f}mm, {rot_err:.1f}도)")
    return res.x, pos_err


def _reset_seed(aloha_q6):
    """초기 자세 IK 시작값: 베이스는 ALOHA 베이스 각도 그대로(두 팔 모두 1:1 대응), 나머지는 팔을 접은 자세."""
    return np.array([aloha_q6[0], 0.05, 0.6, -0.5, 0.0, aloha_q6[5]])


class TrossenAlohaEnvironment(_environment.Environment):
    """An ALOHA-convention environment backed by a Trossen AI Stationary (WidowX AI) robot."""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
        reset_move_s: float = 3.0,
        dry_run: bool = False,
        obs_every: int = 1,
        goal_time_s: float = 0.1,
        height_calib: bool = True,
        floor_margin_m: float = 0.002,
        extra_dz_m: float = 0.0,
        calib_mode: str = "fingertip",
        depth_shift_m: float = 0.0,
        workspace: bool = True,
        workspace_margin_m: float = 0.02,
        workspace_z_max_m: float = 0.35,
        reset_pose: str = "staged",
        gripper_map: str = "pi",
        view_warp=(),
        view_mask: bool = False,
        native: bool = False,
        collision_stop_m: float | None = 0.025,
    ) -> None:
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.robots import make_robot_from_config
        from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig

        station = json.loads(STATION_FILE.read_text())["robot"]
        cams = station["cameras"]
        config = BiWidowXAIFollowerRobotConfig(
            id="bimanual_follower",
            left_arm_ip_address=station["left_arm_ip_address"],
            right_arm_ip_address=station["right_arm_ip_address"],
            # lerobot 은 명령마다 goal_time = multiplier / loop_rate 안에 목표에 도달하게 한다.
            # 0.06초(3/50)는 20ms 명령마다 팔이 툭툭 꺾여서, 기본 0.1초로 늘려 부드럽게 따라가게 한다.
            min_time_to_move_multiplier=goal_time_s * 50,
            loop_rate=50,
            cameras={
                name: RealSenseCameraConfig(
                    serial_number_or_name=cams[name]["serial_number_or_name"],
                    width=640, height=480, fps=30, use_depth=False,
                )
                for name in CAMERAS
            },
        )
        self._robot = make_robot_from_config(config)
        self._keys = list(self._robot.action_features.keys())  # left 7 + right 7
        self._robot.connect()  # lerobot: 팔을 준비(staged) 자세로 이동
        self._mapper = R.PoseRetargeter()
        self._cal = HeightCalibration(enabled=height_calib, floor_margin_m=floor_margin_m, extra_dz_m=extra_dz_m,
                                      mode=calib_mode, depth_shift_m=depth_shift_m)
        self._ws = Workspace(enabled=workspace, margin_m=workspace_margin_m, z_max_m=workspace_z_max_m)
        self._reset_mode = reset_pose
        # native: 이 로봇(WidowX AI) 관절로 학습한 모델 (관제센터 녹화 파인튜닝). ALOHA 변환 없이 관절을 그대로 주고받는다.
        self._native = native
        # 양팔 충돌 방지 (collision.py, 시뮬레이터 충돌 형상 기반 부위별 멈춤 간격 + collision_stop_m). None 이면 끔
        self._guard = ArmCollisionGuard(enabled=collision_stop_m is not None,
                                        stop_m=collision_stop_m if collision_stop_m is not None else 0.0)
        self._view = ViewWarp(view_warp, mask=view_mask)  # 카메라 시점 보정 (viewpoint.py)
        self._gripper_map = gripper_map  # pi: PI 그리퍼 기준 직선 대응 / viperx: 기존 retarget(ViperX 링크) 해석
        self._reset_position = list(reset_position[:6]) if reset_position else DEFAULT_RESET_POSITION
        self._reset_move_s = reset_move_s
        self._render_height = render_height
        self._render_width = render_width
        self._dry_run = dry_run
        # ActionChunkBroker 는 action_horizon 스텝마다 한 번만 관측을 서버로 보낸다. 나머지 스텝의 관측은
        # 쓰이지 않으므로 계산을 건너뛰어 50Hz 제어 주기를 맞춘다 (obs_every = action_horizon).
        # obs_every=0 이면 주기 계산을 끄고, 브로커가 request_fresh_obs() 로 요청할 때만 새로 계산한다.
        self._obs_every = max(0, obs_every)
        self._fresh_requested = True
        self._obs_count = 0
        self._cached_obs = None
        self._cam_last_ok = {}
        self._last_aloha = None
        self._last_cmd = self._native_state()
        self.steps = 0
        self.approx_steps = 0
        logging.info("Trossen ALOHA env ready (dry_run=%s), reset_position(ALOHA)=%s", dry_run, self._reset_position)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _clip_small(native14, tol=0.01):
        """측정 오차 수준(관절 0.01rad, 그리퍼 0.01m 이하)의 범위 이탈만 경계로 맞춘다. 더 크면 그대로 둬서 검증에서 정지."""
        native14 = np.asarray(native14, dtype=np.float64).copy()
        lo = np.concatenate([R.WXAI_LOWER, [0.0], R.WXAI_LOWER, [0.0]])
        hi = np.concatenate([R.WXAI_UPPER, [R.GRIPPER_STROKE_M], R.WXAI_UPPER, [R.GRIPPER_STROKE_M]])
        clipped = np.clip(native14, lo, hi)
        small = np.abs(clipped - native14) <= tol
        native14[small] = clipped[small]
        return native14

    def _native_state(self, obs=None) -> np.ndarray:
        if obs is None:  # 팔 상태만 읽는다 (카메라 지연과 무관하게)
            obs = {}
            for side, arm in (("left", self._robot.left_arm), ("right", self._robot.right_arm)):
                obs.update({f"{side}_{k}": v for k, v in arm.get_observation().items()})
        return self._clip_small(np.array([obs[k] for k in self._keys], dtype=np.float64))

    def _send(self, native14) -> None:
        native14 = R.validate_native_action(self._clip_small(native14))  # WidowX AI 관절/그리퍼 범위
        for off, side in ((0, "왼팔"), (7, "오른팔")):  # 테이블 평면 아래 목표 거절
            p = R.fk(native14[off:off + 6], native=True)[0]
            if self._native:  # native: 바닥 보호(측정 책상면 - floor_margin) 아래로 5mm 넘게 가면 거절
                floor = self._cal._table("left" if off == 0 else "right", p) - self._cal.floor_margin - 0.005
                if p[2] < floor:
                    raise R.RetargetError(f"{side} {self.steps}번째 목표 z={p[2] * 1000:.1f}mm 가 바닥 한계 "
                                          f"{floor * 1000:.1f}mm 아래여서 거절합니다.")
            else:
                R.validate_tool_height(p, side, self.steps)
        self._last_cmd = native14.copy()
        if self._dry_run:
            return
        self._robot.send_action({k: float(v) for k, v in zip(self._keys, native14)})

    def _move(self, target14, seconds, hz=50) -> None:
        start = self._native_state() if not self._dry_run else self._last_cmd.copy()
        n = max(1, int(seconds * hz))
        for i in range(1, n + 1):
            self._send(start + (target14 - start) * (i / n))
            time.sleep(1 / hz)

    # ------------------------------------------------------------ Environment API
    @override
    def reset(self) -> None:
        # upstream: 팔을 reset_position 으로 이동(1초) -> 그리퍼 닫기(1초) -> 열기(0.5초)
        current = self._native_state()
        target = current.copy()
        if self._reset_mode == "staged":
            # Trossen 준비 자세 (자연스러운 자세). ALOHA reset_pose 는 팔을 바짝 접은 자세라 이 로봇으로
            # 흉내 내면 어깨를 세우고 손목을 꺾은 부자연스러운 자세가 된다.
            for off in (0, 7):
                target[off:off + 6] = STAGED_NATIVE[:6]
            logging.info("Reset: Trossen 준비 자세 %s", np.round(target[:6], 3))
        else:
            for off in (0, 7):
                side = "left" if off == 0 else "right"
                pos, rot = R.fk(np.asarray(self._reset_position, float), native=False)
                q, err = _approx_ik(self._cal.to_native(side, pos, rot), rot, _reset_seed(self._reset_position))
                target[off:off + 6] = q
            logging.info("Reset: ALOHA %s -> WidowX AI %s (근사 오차 %.1fmm)", self._reset_position,
                         np.round(target[:6], 3), err * 1000)
        self._move(target, self._reset_move_s)
        closed, opened = target.copy(), target.copy()
        closed[[6, 13]] = 0.0
        opened[[6, 13]] = R.GRIPPER_STROKE_M
        self._move(closed, 1.0)
        self._move(opened, 0.5)
        self._mapper = R.PoseRetargeter()
        self.steps = 0
        self.approx_steps = 0
        self._obs_count = 0
        self._cached_obs = None
        self._fresh_requested = True
        self._last_aloha = None

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def request_fresh_obs(self) -> None:
        """다음 get_observation() 에서 관측을 새로 계산하게 한다 (broker.SmoothChunkBroker 가 호출)."""
        self._fresh_requested = True

    def _read_robot(self):
        """팔 상태 + 카메라 4대.

        cam_high(260522277170)가 실물에서 1초 넘게 프레임을 못 보내는 일이 반복됐다(lerobot 백그라운드
        스레드 500ms 읽기 실패 연속). 새 프레임이 100ms 안에 안 오면 그 카메라의 마지막 프레임을 쓰고,
        MAX_STALE_S 넘게 새 프레임이 없을 때만 멈춘다.
        """
        obs = {}
        for side, arm in (("left", self._robot.left_arm), ("right", self._robot.right_arm)):
            obs.update({f"{side}_{k}": v for k, v in arm.get_observation().items()})
        now = time.monotonic()
        for name, cam in self._robot.cameras.items():
            try:
                obs[name] = cam.async_read(timeout_ms=100)
                self._cam_last_ok[name] = now
            except TimeoutError:
                with cam.frame_lock:
                    frame = cam.latest_frame
                last_ok = self._cam_last_ok.get(name, now)
                if frame is None or now - last_ok > MAX_STALE_S:
                    raise TimeoutError(f"카메라 {name} 새 프레임 없음 {now - last_ok:.1f}초 — 정지")
                logging.warning("카메라 %s 프레임 지연 %.2f초 — 마지막 프레임 사용", name, now - last_ok)
                obs[name] = frame
        return obs

    def _native_to_aloha(self, native14) -> np.ndarray:
        """WidowX AI 측정 자세 -> ALOHA 관절 (높이 보정 반영). retarget.PoseRetargeter.observation 대체."""
        native14 = R.vector(native14, 14, "WidowX AI 상태")
        out = np.zeros(14)
        for side_i, (off, side) in enumerate(((0, "left"), (7, "right"))):
            pos, rot = R.fk(native14[off:off + 6], native=True)
            pos = self._cal.to_aloha(side, pos, rot)
            seed = self._mapper.aloha_seeds[side_i]
            try:
                q = R.ik(pos, rot, seed, native=False)
            except R.RetargetError:
                q, _ = _approx_ik(pos, rot, seed, native=False)
            self._mapper.aloha_seeds[side_i] = q
            out[off:off + 6] = q
            if self._gripper_map == "pi":
                out[off + 6] = G.native_to_state(native14[off + 6])
            else:
                out[off + 6] = np.clip(native14[off + 6] / R.GRIPPER_STROKE_M, 0, 1)  # 기존 retarget 방식
        return out.astype(np.float32)

    @override
    def get_observation(self) -> dict:
        periodic = self._obs_every > 0 and self._obs_count % self._obs_every == 0
        fresh = self._cached_obs is None or self._fresh_requested or periodic
        self._obs_count += 1
        if not fresh:
            return self._cached_obs  # 브로커가 이번 스텝에는 관측을 쓰지 않음
        self._fresh_requested = False
        obs = self._read_robot()
        native = self._native_state(obs) if not self._dry_run else self._last_cmd.copy()
        state = native.astype(np.float32) if self._native else self._native_to_aloha(native)
        self.last_obs_native = native  # 기록용 (recorder.py)
        images = {}
        for cam in CAMERAS:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(np.asarray(obs[cam]), self._render_height, self._render_width)
            )
            img = self._view(cam, img, src_size=np.asarray(obs[cam]).shape[1::-1])  # 시점 보정 (끄면 그대로)
            images[cam] = einops.rearrange(img, "h w c -> c h w")
        self._cached_obs = {"state": state, "images": images}
        return self._cached_obs

    @override
    def _apply_native(self, a: np.ndarray) -> None:
        """WidowX AI 관절 행동을 그대로 보낸다. 안전 장치: 관절 범위, 바닥 보호, 작업 영역, 스텝당 변화 상한."""
        prev = self._last_cmd
        native = prev.copy()
        for side, off in (("left", 0), ("right", 7)):
            q = np.clip(a[off:off + 6], R.WXAI_LOWER, R.WXAI_UPPER)
            p, rot = R.fk(q, native=True)
            t = p.copy()
            floor = self._cal._table(side, p) - self._cal.floor_margin
            if t[2] < floor:
                t[2] = floor
                self._cal.floor_clamps += 1
            t = self._ws.clamp(side, t)
            if np.linalg.norm(t - p) > 1e-6:  # 잘린 위치로 다시 IK (방향은 그대로)
                try:
                    q = R.ik(t, rot, q, native=True)
                except R.RetargetError:
                    q, _ = _approx_ik(t, rot, q)
                    self.approx_steps += 1
            jump = float(np.abs(q - prev[off:off + 6]).max())
            if jump > MAX_STEP_RAD:
                raise R.RetargetError(f"{side} 명령이 한 스텝에 {jump:.2f} rad 변함 (상한 {MAX_STEP_RAD}) — 정지")
            native[off:off + 6] = q
            native[off + 6] = float(np.clip(a[off + 6], 0.0, G.STROKE))
        self._send(self._guard.filter(native, prev))  # 양팔 충돌 방지 (collision.py)
        self.steps += 1

    def apply_action(self, action: dict) -> None:
        if self._native:
            self._apply_native(np.asarray(action["actions"], dtype=np.float64)[:14])
            return
        aloha = np.asarray(action["actions"], dtype=np.float64)
        prev = self._last_cmd
        prev_aloha = self._last_aloha if self._last_aloha is not None else aloha
        native = prev.copy()
        for off in (0, 7):
            seed = prev[off:off + 6]
            # 모델 자체가 크게 움직인 만큼(묶음 경계 불연속 등)은 upstream 처럼 그대로 따른다.
            # 변환 결과가 모델 움직임보다 훨씬 크게 튀면 IK 해 전환으로 본다.
            aloha_jump = float(np.abs(aloha[off:off + 6] - prev_aloha[off:off + 6]).max())
            allowed = min(MAX_STEP_RAD, max(MAX_IK_JUMP_RAD, 1.5 * aloha_jump + 0.05))
            q = None
            side = "left" if off == 0 else "right"
            pos, rot = R.fk(aloha[off:off + 6], native=False)
            pos = self._ws.clamp(side, self._cal.to_native(side, pos, rot))  # 높이 보정 + 바닥 보호 + 작업 영역
            try:  # 1) 정확한 IK (직전 명령에서 시작)
                q = R.ik(pos, rot, seed, native=True)
                if np.abs(q - seed).max() > allowed:
                    q = None  # 다른 해로 갈아탐 -> 2) 로
            except R.RetargetError:
                pass
            if q is None:  # 2) 직전 자세 근처의 가장 가까운 해 (작업 공간 경계·해 갈아타기 방지)
                q, _ = _approx_ik(pos, rot, seed)
                self.approx_steps += 1
                if self.approx_steps in (1, 10, 100) or self.approx_steps % 500 == 0:
                    logging.info("근사 변환 %d회 (작업 공간 경계 또는 IK 해 전환 회피)", self.approx_steps)
            jump = float(np.abs(q - seed).max())
            if jump > allowed:
                raise R.RetargetError(
                    f"관절 변환 결과가 직전 명령에서 {jump:.2f} rad 튐 (모델 동작 변화 {aloha_jump:.2f} rad, "
                    f"허용 {allowed:.2f}) — 정지")
            native[off:off + 6] = q
            native[off + 6] = (G.action_to_native(aloha[off + 6]) if self._gripper_map == "pi"
                               else R.aloha_action_gripper_to_native(aloha[off + 6]))
        self._send(self._guard.filter(native, prev))  # 양팔 충돌 방지 (collision.py)
        self._last_aloha = aloha
        self.steps += 1

    # ------------------------------------------------------------ shutdown
    def _go_home(self, seconds: float = HOME_MOVE_S) -> None:
        """양팔을 동시에 준비(staged) 자세 -> 휴식(0) 자세로 천천히 이동."""
        import trossen_arm as t
        arms = (self._robot.left_arm, self._robot.right_arm)
        for label, poses in (("준비", [list(a.config.staged_positions) for a in arms]),
                             ("휴식", [[0.0] * 7 for _ in arms])):
            logging.info("기본 자세 복귀: %s 자세로 이동 · %.0f초", label, seconds)
            for arm, pose in zip(arms, poses):
                arm.driver.set_all_modes(t.Mode.position)
                arm.driver.set_all_positions(pose, goal_time=seconds, blocking=False)
            time.sleep(seconds + 0.2)

    def close(self, *, error: bool) -> None:
        """종료. 정상·오류·Ctrl-C 모두 기본 자세(준비 -> 휴식)로 돌아간 뒤 끈다.

        공중에서 브레이크를 걸면 모터가 자세를 버티느라 부하가 걸리므로, 오류 시에도 먼저 복귀를 시도한다.
        복귀 자체가 실패하면(드라이버 오류 등)·복귀 중 다시 Ctrl-C 를 누르면 그 자리에서 idle 로 멈춘다.
        """
        import trossen_arm as t
        arms = (self._robot.left_arm, self._robot.right_arm)
        if error:
            logging.warning("오류로 종료 — 기본 자세로 복귀합니다. 위험하면 물리 비상정지를 누르세요.")
        try:
            self._go_home()
        except BaseException as e:  # noqa: BLE001 — Ctrl-C 포함
            logging.error("기본 자세 복귀 실패 (%r) — 그 자리에서 idle 로 멈춥니다.", e)
        for arm in arms:  # 휴식 자세에서(또는 복귀 실패 시 그 자리에서) 모터 해제
            try:
                arm.driver.set_all_modes(t.Mode.idle)
            except Exception as e:  # noqa: BLE001
                logging.error("idle 전환 실패 — 물리 비상정지를 누르세요: %s", e)
            try:
                arm.driver.cleanup()
            except Exception:  # noqa: BLE001
                pass
        for cam in self._robot.cameras.values():
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001
                pass
