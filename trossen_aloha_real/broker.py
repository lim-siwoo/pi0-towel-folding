"""부드러운 동작 묶음 브로커 (upstream ActionChunkBroker 의 대안, --broker smooth).

upstream ActionChunkBroker 는 action_horizon(25) 스텝을 다 쓴 다음에야 다음 추론을 동기로 호출한다.
그래서 0.5초마다 추론 시간(~115ms) 동안 명령이 끊기고, 새 묶음은 이전 묶음과 이어지지 않아 팔이 튄다.

이 브로커는:
  1) 미리 계산: 묶음이 끝나기 prefetch 스텝 전의 관측으로 다음 추론을 백그라운드에서 시작한다.
     새 묶음은 그 관측 시점 기준이므로, 전환 시점에는 prefetch 만큼 지난 위치(index=prefetch)부터 쓴다.
  2) 이어 붙이기: 전환 후 blend 스텝 동안 "이전 묶음이 예측해 둔 뒷부분"과 "새 묶음"을 선형으로 섞는다.
     (모델은 50 스텝을 예측하므로 이전 묶음의 뒷부분이 남아 있다.)
추론이 제때 끝나지 않으면 기다린다(그 동안은 upstream 과 같이 멈춤).
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Dict

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override


class SmoothChunkBroker(_base_policy.BasePolicy):
    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int = 25, prefetch: int = 8,
                 blend: int = 8, env=None) -> None:
        self._policy = policy
        self._h = action_horizon
        self._prefetch = prefetch
        self._blend = blend
        self._env = env  # 다음 스텝에 새 관측이 필요하다고 알려 주기 위해 사용
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.reset()

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._chunk = None      # 현재 묶음 (50, 14)
        self._start = 0         # 현재 묶음에서 쓰기 시작한 index
        self._idx = 0           # 현재 묶음에서 다음에 쓸 index
        self._prev = None       # 이전 묶음 (이어 붙이기용)
        self._prev_idx = 0      # 이전 묶음에서 대응하는 index
        self._blend_left = 0
        self._future = None
        self.waits = 0
        if self._env is not None:
            self._env.request_fresh_obs()

    def _launch(self, obs: Dict) -> None:
        self._future = self._pool.submit(self._policy.infer, obs)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._chunk is None:  # 첫 묶음: 동기 추론
            self._chunk = np.asarray(self._policy.infer(obs)["actions"])
            self._start = self._idx = 0

        # 미리 계산 시작 (이번 스텝의 관측은 env 가 새로 계산해 둔 것)
        if self._future is None and self._idx == self._start + self._h - self._prefetch:
            self._launch(obs)

        # 묶음 전환
        if self._idx >= self._start + self._h:
            if self._future is None:  # 혹시 시작 못 했으면 지금 동기로
                self._launch(obs)
            if not self._future.done():
                self.waits += 1
            new = np.asarray(self._future.result()["actions"])
            self._future = None
            self._prev, self._prev_idx = self._chunk, self._idx
            self._chunk, self._start, self._idx = new, self._prefetch, self._prefetch
            self._blend_left = self._blend

        a = self._chunk[self._idx].copy()
        if self._blend_left > 0 and self._prev is not None and self._prev_idx < len(self._prev):
            w = (self._blend - self._blend_left + 1) / (self._blend + 1)  # 이전 -> 새 로 점점
            a = (1 - w) * self._prev[self._prev_idx] + w * a
            self._prev_idx += 1
            self._blend_left -= 1
        self._idx += 1

        # 다음 스텝이 미리 계산 시점이면 env 에 새 관측을 요청
        if self._env is not None and self._future is None and self._idx == self._start + self._h - self._prefetch:
            self._env.request_fresh_obs()
        return {"actions": a}

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class RtcChunkBroker(_base_policy.BasePolicy):
    """관제센터 pi0_deployment.Pi0DeploymentPolicy 의 RTC 스케줄(Algorithm 1)을 옮긴 것 (--broker rtc).

    - 첫 묶음만 동기 추론. 이후 묶음 index 가 replan(25) 에 도달하면, 그 시점 관측과
      "이전 묶음의 남은 동작(plan[index:])" 을 rtc_server 로 보내 백그라운드에서 추론한다.
    - 추론 중에도 이전 묶음을 계속 실행한다(멈추지 않음).
    - 결과가 오면 새 묶음의 시작 시점 = 요청 시점으로 두므로, 추론하는 동안 지나간 스텝만큼 앞부분은
      자동으로 건너뛴다. 서버는 그 앞부분을 이전 묶음과 맞추도록 유도하므로 이음매가 이어진다.
    - delay 예측 = 최근 10회 실제 지연(스텝)의 최대값 + 2.
    """

    MARGIN = 2

    def __init__(self, policy: _base_policy.BasePolicy, *, replan: int = 25, rate_hz: int = 50,
                 warm_ms: float = 200.0, env=None, horizon: int = 50) -> None:
        import collections
        import math
        import time

        self._math, self._time = math, time
        self._policy = policy
        self._replan, self._rate, self._h = replan, rate_hz, horizon
        self._env = env
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc")
        self._delays = collections.deque(maxlen=10)
        self._warm_steps = math.ceil(warm_ms * rate_hz / 1000)
        self.reset()

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._plan = None
        self._plan_start = 0
        self._steps = 0
        self._future = None
        self._request = None
        self._delays.clear()
        self._delays.append(self._warm_steps)
        self.stats = dict(inferences=0, last_delay_steps=0, max_delay_steps=0)
        if self._env is not None:
            self._env.request_fresh_obs()

    def _accept_done(self) -> None:
        if self._future is None or not self._future.done():
            return
        future, req = self._future, self._request
        self._future = self._request = None
        actions = np.asarray(future.result()["actions"], dtype=np.float64)
        consumed = self._steps - req["start"]
        if not (0 <= consumed <= req["prefix_length"] and consumed < self._h):
            raise RuntimeError(f"RTC 응답이 너무 늦었습니다 (지난 스텝 {consumed}, 남은 묶음 {req['prefix_length']}) — 정지")
        wall = self._math.ceil((self._time.monotonic() - req["started"]) * self._rate)
        self._delays.append(max(consumed, wall))
        self.stats["last_delay_steps"] = max(consumed, wall)
        self.stats["max_delay_steps"] = max(self.stats["max_delay_steps"], max(consumed, wall))
        self.stats["inferences"] += 1
        self._plan, self._plan_start = actions, req["start"]

    def _launch(self, obs: Dict, index: int) -> None:
        remaining = self._plan[index:]
        delay = max(self._delays) + self.MARGIN
        if delay > len(remaining):
            raise RuntimeError(f"RTC 추론 지연 예측({delay} 스텝)이 남은 묶음({len(remaining)})보다 깁니다 — 정지")
        previous = np.zeros((self._h, self._plan.shape[1]), np.float32)
        previous[:len(remaining)] = remaining
        request_obs = {**obs, "rtc_previous": previous, "rtc_prefix_length": np.int32(len(remaining)),
                       "rtc_delay": np.int32(delay)}
        self._request = dict(start=self._steps, prefix_length=len(remaining), started=self._time.monotonic())
        self._future = self._pool.submit(self._policy.infer, request_obs)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._plan is None:  # 첫 묶음만 동기 추론
            self._plan = np.asarray(self._policy.infer(obs)["actions"], dtype=np.float64)
            self._plan_start = self._steps
            self.stats["inferences"] += 1
        else:
            self._accept_done()
        index = self._steps - self._plan_start
        if not 0 <= index < self._h:
            raise RuntimeError("RTC 동작 버퍼가 소진됐습니다 (추론이 너무 느림) — 정지")
        if index >= self._replan and self._future is None:
            self._launch(obs, index)
        action = self._plan[index].copy()
        self._steps += 1
        # 다음 스텝이 재계획 시점이면 env 에 새 관측을 요청
        if self._env is not None and self._future is None and index + 1 >= self._replan:
            self._env.request_fresh_obs()
        return {"actions": action}

    def close(self) -> None:
        if self._future is not None:
            self._future.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
