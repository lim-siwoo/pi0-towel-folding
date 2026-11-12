"""그리퍼 변환: pi0(PI) 그리퍼 기준 <-> WidowX AI 직선 그리퍼 (0~40mm).

openpi aloha_policy.py (adapt_to_pi=True) 기준
  - 모델 내부 그리퍼 값 g_pi 는 PI 하드웨어 모터 각도 0.5476~1.6296 rad(엔코더 2405~3110)를
    0(닫힘)~1(열림) 로 정규화한 값이다 (PI 출력은 '라디안, 0.5476 기준' 이라 1.082 가 완전히 열림).
  - 출력 변환 _gripper_from_angular: action = normalize(g + 0.5476, -0.6213, 1.4910)  (일반 ALOHA 관절 정규화)
  - 입력 변환 _gripper_to_angular  : state(일반 ALOHA 손가락 직선 정규화) -> ViperX 링크 -> 라디안 -> (0.5476, 1.6296) 정규화

기존 retarget.py 는 action 을 일반 ALOHA(ViperX) 그리퍼로 해석해서, PI 의 '완전히 닫힘'(action 0.553)이
WidowX AI 24mm(60% 열림)가 됐다. 관측도 같은 식이라 24mm 를 모델은 '완전히 닫힘'으로 봤다.
-> 여기서는 PI 기준 0~1 을 WidowX AI 0~40mm 에 직선으로 대응시킨다.
"""

from __future__ import annotations

import numpy as np

PI_MIN, PI_MAX = 0.5476, 1.6296          # PI 그리퍼 모터 각도 범위 (rad)
JOINT_CLOSE, JOINT_OPEN = -0.6213, 1.4910  # 일반 ALOHA puppet 그리퍼 관절
POS_CLOSE, POS_OPEN = 0.01844, 0.05800     # 일반 ALOHA puppet 손가락 직선 위치
ARM, HORN = 0.036, 0.022                   # Interbotix 그리퍼 링크
STROKE = 0.04                              # WidowX AI 그리퍼 최대 열림 (m)


def action_to_pi_open(action_value):
    """모델 출력(일반 ALOHA 관절 정규화) -> PI 기준 열림 정도 0(닫힘)~1(열림). _gripper_from_angular 의 역."""
    angle_rel = np.asarray(action_value) * (JOINT_OPEN - JOINT_CLOSE) + JOINT_CLOSE - PI_MIN
    return angle_rel / (PI_MAX - PI_MIN)


def action_to_native(action_value):
    """모델 그리퍼 출력 -> WidowX AI 그리퍼 열림 (m)."""
    return float(np.clip(action_to_pi_open(action_value), 0.0, 1.0) * STROKE)


def native_to_state(opening_m):
    """WidowX AI 그리퍼 열림 (m) -> 모델 입력용 일반 ALOHA 손가락 직선 정규화 값.

    openpi 입력 변환(_gripper_to_angular)을 거치면 정확히 PI 기준 열림 정도(opening/STROKE)가 되도록 역산한다.
    """
    pi_open = float(np.clip(opening_m / STROKE, 0.0, 1.0))
    # openpi 입력 변환은 arcsin 을 쓰므로 π/2 (PI 기준 약 0.95) 보다 열린 값은 표현할 수 없다.
    # PI 데이터에서 완전히 열린 상태도 같은 이유로 약 0.95 로 들어간다.
    angle = min(PI_MIN + pi_open * (PI_MAX - PI_MIN), np.pi / 2)
    s = np.sin(angle)
    linear = HORN * s + np.sqrt(HORN ** 2 * s ** 2 - HORN ** 2 + ARM ** 2)
    return float((linear - POS_CLOSE) / (POS_OPEN - POS_CLOSE))


def _openpi_gripper_to_angular(value):
    """검증용: openpi aloha_policy._gripper_to_angular 와 같은 계산."""
    value = value * (POS_OPEN - POS_CLOSE) + POS_CLOSE
    v = (HORN ** 2 + value ** 2 - ARM ** 2) / (2 * HORN * value)
    return (np.arcsin(np.clip(v, -1.0, 1.0)) - PI_MIN) / (PI_MAX - PI_MIN)
