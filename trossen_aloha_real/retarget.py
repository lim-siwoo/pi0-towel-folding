"""Model-based ALOHA <-> WXAI pose retargeting; never imports a robot driver.

Copied from /home/dit/control_center/pi0_retarget.py (2026-11-04) so this
project runs independently of the control center. Only change: the
pi0_execution import was replaced by a local constant.

The arm geometries are not interchangeable joint vectors.  These FK chains
come from the VX300S URDF and the installed WXAI SDK kinematic specification.
End-effector poses are expressed in each arm's own base frame, metres/radians.
This is an experimental geometric adapter, not a camera/scene calibration or
a demonstration of task success.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

EXECUTED_STEPS = 25  # was imported from control_center/pi0_execution.py

VERSION = "vx300s-wxai-pose-v2"
POSITION_TOLERANCE_M = 0.001
ORIENTATION_TOLERANCE_RAD = math.radians(0.5)
GRIPPER_STROKE_M = 0.04
# Standard ALOHA observations normalize finger distance, whereas actions
# normalize the rotary gripper motor angle. Source: ALOHA constants.py and
# Interbotix robot_convert_angular_position_to_linear (metres/radians).
ALOHA_GRIPPER_POSITION_CLOSED_M = 0.01844
ALOHA_GRIPPER_POSITION_OPEN_M = 0.05800
ALOHA_GRIPPER_JOINT_CLOSED_RAD = -0.6213
ALOHA_GRIPPER_JOINT_OPEN_RAD = 1.4910
ALOHA_GRIPPER_HORN_RADIUS_M = 0.022
ALOHA_GRIPPER_ARM_LENGTH_M = 0.036
GRIPPER_MAPPING = "aloha-linkage-relative-stroke-v1"
# User-specified table surface, in each arm's base frame (metres).
# The table is 20 mm below the mounting plane; base z=0 is not a floor.
TABLE_PLANE_Z_M = -0.020
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ACTION_NAMES = tuple(
    f"{side}_{name}.pos"
    for side in ("left", "right")
    for name in (*[f"joint_{i}" for i in range(6)], "left_carriage_joint")
)

# Origins are applied before rotation, as in URDF and StandardJoints in the SDK.
ALOHA_ORIGINS = np.array([
    [0, 0, 0.079], [0, 0, 0.04805], [0.05955, 0, 0.3],
    [0.2, 0, 0], [0.1, 0, 0], [0.069744, 0, 0],
], dtype=float)
ALOHA_AXES = np.array([
    [0, 0, 1], [0, 1, 0], [0, 1, 0],
    [1, 0, 0], [0, 1, 0], [1, 0, 0],
], dtype=float)
ALOHA_LOWER = np.radians([-179.999, -106, -101, -179.999, -107, -179.999])
ALOHA_UPPER = np.radians([179.999, 72, 92, 179.999, 128, 179.999])
ALOHA_TOOL = np.array([0.042825 + 0.025875 + 0.0385, 0, 0])

WXAI_ORIGINS = np.array([
    [0, 0, 0.05725], [0.02, 0, 0.04625], [-0.264, 0, 0],
    [0.245, 0, 0.06], [0.06775, -0.00005, 0.0455],
    [0.02895, 0, -0.0455],
], dtype=float)
WXAI_AXES = np.array([
    [0, 0, 1], [0, 1, 0], [0, -1, 0],
    [0, -1, 0], [0, 0, -1], [1, 0, 0],
], dtype=float)
WXAI_LOWER = np.array([-math.pi, 0, 0, -math.pi / 2, -math.pi / 2, -math.pi])
WXAI_UPPER = np.array([math.pi, math.pi, 3 * math.pi / 4, math.pi / 2, math.pi / 2, math.pi])
WXAI_TOOL = np.array([0.156062, 0, 0])
WXAI_STAGED = np.array([0, math.pi / 3, math.pi / 6, math.pi / 5, 0, 0])
ALOHA_SEED = np.array([0, -0.96, 1.16, 0, -0.3, 0])


class RetargetError(ValueError):
    pass


def vector(value, size, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise RetargetError(f"{name}: 유한한 {size}개 값이 필요합니다.")
    return value


def clip_aloha_gripper_actions(actions):
    """Saturate finite normalized grippers; preserve arm joints and raw input.

    OpenPI's ALOHA output transform normalizes but does not saturate model
    predictions. Values past an endpoint map to fully closed/open on this rig.
    """
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 14 or not np.isfinite(values).all():
        raise RetargetError("ALOHA 예측은 유한한 N×14 배열이어야 합니다.")
    bounded = values.copy()
    grippers = values[:, [6, 13]]
    bounded[:, [6, 13]] = np.clip(grippers, 0.0, 1.0)
    return bounded, int(np.count_nonzero(grippers != bounded[:, [6, 13]]))


def aloha_action_gripper_to_native(value):
    """Map ALOHA motor-angle actions to the WXAI linear carriage stroke.

    Relative opening is preserved, using the same slider-crank geometry that
    produces ALOHA's measured finger positions. The two ALOHA endpoint
    calibrations differ slightly (about 0.11 mm in WXAI stroke at full open),
    so explicit close/open commands retain exact native endpoints.
    """
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise RetargetError("ALOHA 그리퍼 명령은 유한해야 합니다.")
    bounded = np.clip(value, 0, 1)
    angle = (ALOHA_GRIPPER_JOINT_CLOSED_RAD
             + bounded * (ALOHA_GRIPPER_JOINT_OPEN_RAD - ALOHA_GRIPPER_JOINT_CLOSED_RAD))
    lever = ALOHA_GRIPPER_HORN_RADIUS_M * np.sin(angle)
    distance = lever + np.sqrt(
        ALOHA_GRIPPER_ARM_LENGTH_M**2 - ALOHA_GRIPPER_HORN_RADIUS_M**2 + lever**2)
    opening = np.clip(
        (distance - ALOHA_GRIPPER_POSITION_CLOSED_M)
        / (ALOHA_GRIPPER_POSITION_OPEN_M - ALOHA_GRIPPER_POSITION_CLOSED_M), 0, 1)
    opening = np.where(bounded <= 0, 0, np.where(bounded >= 1, 1, opening))
    return opening * GRIPPER_STROKE_M


def fk(q, *, native):
    q = vector(q, 6, "관절")
    origins, axes, tool = (
        (WXAI_ORIGINS, WXAI_AXES, WXAI_TOOL) if native
        else (ALOHA_ORIGINS, ALOHA_AXES, ALOHA_TOOL)
    )
    rotation, position = np.eye(3), np.zeros(3)
    for origin, axis, angle in zip(origins, axes, q, strict=True):
        position += rotation @ origin
        rotation = rotation @ Rotation.from_rotvec(axis * angle).as_matrix()
    position += rotation @ tool
    return position, rotation


def ik(position, rotation, seed, *, native):
    position = vector(position, 3, "팔 끝 위치")
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise RetargetError("팔 끝 방향이 유효하지 않습니다.")
    lower, upper = (WXAI_LOWER, WXAI_UPPER) if native else (ALOHA_LOWER, ALOHA_UPPER)
    seed = np.clip(vector(seed, 6, "역기구학 시작값"), lower + 1e-7, upper - 1e-7)

    def error(q):
        got_position, got_rotation = fk(q, native=native)
        angle = Rotation.from_matrix(rotation.T @ got_rotation).as_rotvec()
        return np.concatenate((got_position - position, 0.15 * angle))

    result = least_squares(
        error, seed, bounds=(lower, upper), max_nfev=80,
        ftol=1e-8, xtol=1e-8, gtol=1e-8,
    )
    got_position, got_rotation = fk(result.x, native=native)
    distance = float(np.linalg.norm(got_position - position))
    angle = float(Rotation.from_matrix(rotation.T @ got_rotation).magnitude())
    if distance > POSITION_TOLERANCE_M or angle > ORIENTATION_TOLERANCE_RAD:
        target = "WidowX AI" if native else "ALOHA"
        raise RetargetError(
            f"{target} 자세 변환 불가: 위치 오차 {distance * 1000:.2f}mm, "
            f"방향 오차 {math.degrees(angle):.2f}도. 명령을 전송하지 않습니다."
        )
    return result.x


class PoseRetargeter:
    def __init__(self):
        self.aloha_seeds = [ALOHA_SEED.copy(), ALOHA_SEED.copy()]

    def observation(self, native_state):
        native_state = vector(native_state, 14, "WidowX AI 상태")
        result = np.zeros(14)
        for side, offset in enumerate((0, 7)):
            q = native_state[offset:offset + 6]
            if np.any(q < WXAI_LOWER - 1e-4) or np.any(q > WXAI_UPPER + 1e-4):
                raise RetargetError("실제 관절 상태가 등록된 WidowX AI 범위를 벗어났습니다.")
            grip = native_state[offset + 6]
            if not -0.001 <= grip <= GRIPPER_STROKE_M + 0.001:
                raise RetargetError("실제 그리퍼 상태가 미터 단위 범위와 다릅니다.")
            position, rotation = fk(q, native=True)
            solved = ik(position, rotation, self.aloha_seeds[side], native=False)
            self.aloha_seeds[side] = solved
            result[offset:offset + 6] = solved
            # Observation is linear finger opening, unlike the action's angle.
            result[offset + 6] = np.clip(grip / GRIPPER_STROKE_M, 0, 1)
        return result.astype(np.float32)

    def action(self, aloha_action, native_seed):
        aloha_action = vector(aloha_action, 14, "ALOHA 모델 출력")
        native_seed = vector(native_seed, 14, "WidowX AI 현재 상태")
        result = np.zeros(14)
        for offset in (0, 7):
            q = aloha_action[offset:offset + 6]
            if np.any(q < ALOHA_LOWER - 1e-4) or np.any(q > ALOHA_UPPER + 1e-4):
                raise RetargetError("모델의 관절 목표가 ALOHA 기구학 범위를 벗어났습니다.")
            grip = aloha_action[offset + 6]
            position, rotation = fk(q, native=False)
            result[offset:offset + 6] = ik(
                position, rotation, native_seed[offset:offset + 6], native=True,
            )
            result[offset + 6] = aloha_action_gripper_to_native(grip)
        return result


def validate_native_action(action):
    action = vector(action, 14, "실물 목표")
    for offset in (0, 7):
        if (np.any(action[offset:offset + 6] < WXAI_LOWER)
                or np.any(action[offset:offset + 6] > WXAI_UPPER)
                or not 0 <= action[offset + 6] <= GRIPPER_STROKE_M):
            raise RetargetError("변환된 명령이 WidowX AI 범위를 벗어났습니다.")
    return action


def validate_tool_height(position, side, index):
    """Reject targets below this station's table, not below its arm mounts."""
    if position[2] < TABLE_PLANE_Z_M:
        raise RetargetError(
            f"{side} {index}번째 팔 끝 목표가 테이블 평면 "
            f"(z={TABLE_PLANE_Z_M * 1000:.1f}mm) 아래여서 거절합니다. "
            f"목표 z={position[2] * 1000:.2f}mm (로봇 베이스 기준).")


def validate_chunk(actions, observation):
    """Validate physical targets; dispatch applies the native driver limiter.

    Raw predictions no longer have the temporary .35rad / 8cm jump heuristics.
    """
    actions = np.asarray(actions, dtype=np.float64)
    observation = vector(observation, 14, "청크 시작 상태")
    if actions.shape != (EXECUTED_STEPS, 14) or not np.isfinite(actions).all():
        raise RetargetError(f"실물 청크는 유한한 {EXECUTED_STEPS}×14 목표여야 합니다.")
    for action in actions:
        validate_native_action(action)
    for offset in (0, 7):
        side = "왼팔" if offset == 0 else "오른팔"
        for index, action in enumerate(actions, start=1):
            position, _ = fk(action[offset:offset + 6], native=True)
            validate_tool_height(position, side, index)
    return actions
