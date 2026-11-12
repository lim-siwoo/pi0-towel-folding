"""openpi examples/aloha_real/main.py 와 같은 클라이언트 (Trossen AI Stationary 용).

upstream 과 같은 구성: WebsocketClientPolicy -> ActionChunkBroker(action_horizon=25)
-> PolicyAgent -> Runtime(max_hz=50, num_episodes=1, max_episode_steps=1000).
환경만 env.TrossenAlohaEnvironment (ALOHA <-> WidowX AI 관절 변환 포함) 로 바꿨다.

서버 (openpi aloha_real README 'Towel Task'):
    cd ~/pi0_aloha_towel/trossen_openpi
    uv run scripts/serve_policy.py --default_prompt='fold the towel' policy:checkpoint \
        --policy.config=pi0_aloha_towel \
        --policy.dir=$HOME/pi0_aloha_towel/openpi_data/openpi-assets/checkpoints/pi0_aloha_towel

클라이언트:
    cd ~/pi0_aloha_towel/trossen_openpi/examples/trossen_ai
    .venv/bin/python ~/pi0_aloha_towel/trossen_aloha_real/main.py
"""

import argparse
import logging

import numpy as np
import pathlib
import signal
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from openpi_client import action_chunk_broker  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402
from openpi_client.runtime import runtime as _runtime  # noqa: E402
from openpi_client.runtime.agents import policy_agent as _policy_agent  # noqa: E402

import broker as _broker  # noqa: E402
import env as _env  # noqa: E402
import recorder as _recorder  # noqa: E402


HZ = 50
VIEW_WARP = {"none": (), "high": ("cam_high",), "wrist": ("cam_left_wrist", "cam_right_wrist"),
             "all": ("cam_high", "cam_left_wrist", "cam_right_wrist")}


def main(args) -> None:
    global HZ
    HZ = args.hz or (30 if args.native else 50)
    if args.native:
        if args.goal_time is None:
            args.goal_time = 4 / 30  # Trossen 클라이언트와 같은 목표 도달 시간 (multiplier 4 @ 30Hz)
        # 시연 기준 넉넉한 제한: 행동(leader) 손끝이 측정 책상면 아래 1% 약 55mm (바닥은 40mm 로 제한), 높이 최대 569mm,
        # xy 는 측정 상자 + 60mm 밖이 0.1% 이하 (finetune/README.md)
        if args.floor_margin_mm is None:
            args.floor_margin_mm = 40.0
        if args.workspace_margin_mm is None:
            args.workspace_margin_mm = 60.0
        if args.workspace_z_max_mm is None:
            args.workspace_z_max_mm = 600.0
        logging.info("native 모드: WidowX AI 관절 그대로, %dHz, 바닥 여유 %.0fmm, 작업 영역 여유 %.0fmm, 높이 ≤ %.0fmm",
                     HZ, args.floor_margin_mm, args.workspace_margin_mm, args.workspace_z_max_mm)
    if args.goal_time is None:
        args.goal_time = 0.1
    if args.floor_margin_mm is None:
        args.floor_margin_mm = 2.0
    if args.workspace_margin_mm is None:
        args.workspace_margin_mm = 20.0
    if args.workspace_z_max_mm is None:
        args.workspace_z_max_mm = 350.0
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()
    if args.broker == "rtc" and not metadata.get("rtc"):
        raise SystemExit("--broker rtc 는 rtc_server.py 로 띄운 서버가 필요합니다 (지금 서버 metadata 에 rtc 없음).")
    smooth = args.broker in ("smooth", "rtc")
    environment = _env.TrossenAlohaEnvironment(
        reset_position=metadata.get("reset_pose"),
        dry_run=args.dry_run,
        obs_every=0 if smooth else args.action_horizon,
        goal_time_s=args.goal_time,
        height_calib=not args.no_height_calib,
        floor_margin_m=args.floor_margin_mm / 1000,
        extra_dz_m=args.extra_dz_mm / 1000,
        calib_mode=args.calib_mode,
        depth_shift_m=args.depth_shift_mm / 1000,
        workspace=not args.no_workspace,
        workspace_margin_m=args.workspace_margin_mm / 1000,
        workspace_z_max_m=args.workspace_z_max_mm / 1000,
        reset_pose=args.reset_pose,
        gripper_map=args.gripper_map,
        view_warp=VIEW_WARP[args.view_warp],
        view_mask=args.view_mask,
        native=args.native,
        collision_stop_m=None if args.no_collision_guard else args.collision_stop_mm / 1000,
    )
    if args.broker == "rtc":  # 관제센터 RTC 스케줄 (broker.RtcChunkBroker + rtc_server.py)
        policy = _broker.RtcChunkBroker(
            ws_client_policy, replan=args.action_horizon, rate_hz=HZ,
            warm_ms=float(metadata.get("rtc_warm_guided_ms", 200.0)), env=environment,
        )
    elif args.broker == "smooth":  # 미리 계산 + 묶음 경계 이어 붙이기 (broker.py)
        policy = _broker.SmoothChunkBroker(
            ws_client_policy, action_horizon=args.action_horizon, prefetch=args.prefetch, blend=args.blend,
            env=environment,
        )
    else:  # upstream 그대로
        policy = action_chunk_broker.ActionChunkBroker(policy=ws_client_policy, action_horizon=args.action_horizon)
    logging.info("broker=%s, goal_time=%.2fs, %dHz", args.broker, args.goal_time, HZ)
    recorder = _recorder.Recorder(environment, frame_every=args.record_every)  # 진단용 기록 (upstream 에 없음)
    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(policy=policy),
        subscribers=[recorder],
        max_hz=HZ,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    error = True
    try:
        runtime.run()
        error = False
    finally:
        # 로봇을 먼저 기본 자세로 내려놓고, 그 뒤에 기록을 저장한다.
        # (30fps 녹화는 저장이 수 초~수십 초 걸려서, 그동안 팔이 공중에 멈춰 있지 않게)
        try:
            if smooth:
                policy.close()
                if args.broker == "rtc":
                    logging.info("RTC 통계: %s", policy.stats)
            logging.info("바닥 보호로 높이를 자른 명령 수: %d", environment._cal.floor_clamps)
            logging.info("작업 영역 경계로 자른 명령 수: %s", environment._ws.clamps)
            g = environment._guard
            logging.info("양팔 충돌 회피: 비켜 감 %d %s, 경로 일부만 감 %d, 두 팔 멈춤 %d / 가장 가까웠던 부위 %s (기준까지 %.0fmm)",
                         g.deflected, g.modes, g.partial, g.held_both, g.worst_pair,
                         g.min_seen * 1000 if np.isfinite(g.min_seen) else float("nan"))
            environment.close(error=error)
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)  # 저장 중 Ctrl+C 로 영상이 날아가지 않게
            logging.info("기록 저장 중… 끝날 때까지 기다리세요 (Ctrl+C 를 더 누르지 마세요).")
            recorder.save()  # 오류로 끝나도 그때까지의 기록 저장


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # upstream 은 tyro 를 쓰지만 이 venv 에 없어 같은 이름의 argparse 옵션으로 대신한다.
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--action-horizon", type=int, default=25)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--record-every", type=int, default=5,
                   help="cam_high.mp4 에 몇 스텝마다 한 장 남길지 (1 = 제어 주기 그대로, native 30fps)")
    p.add_argument("--dry-run", action="store_true",
                   help="팔에 모델 동작을 보내지 않음 (연결 시 준비 자세, 종료 시 휴식 자세 이동은 함)")
    p.add_argument("--broker", choices=["rtc", "smooth", "upstream"], default="rtc",
                   help="rtc: 관제센터 RTC (rtc_server.py 필요) / smooth: 미리 계산 + 경계 섞기 / "
                        "upstream: openpi ActionChunkBroker 그대로")
    p.add_argument("--prefetch", type=int, default=8, help="smooth: 묶음 끝나기 몇 스텝 전에 다음 추론 시작")
    p.add_argument("--blend", type=int, default=8, help="smooth: 묶음 경계에서 섞는 스텝 수")
    p.add_argument("--no-height-calib", action="store_true",
                   help="calibration/trossen_measurements_raw.json 높이 보정을 끔 (베이스 기준 높이 그대로)")
    p.add_argument("--no-workspace", action="store_true", help="측정 기반 작업 영역 제한 끄기")
    p.add_argument("--workspace-z-max-mm", type=float, default=None, help="작업 영역 높이 상한 (기본 350, native 600)")
    p.add_argument("--workspace-margin-mm", type=float, default=None,
                   help="측정 때 짚은 영역 밖으로 허용하는 여유 (mm)")
    p.add_argument("--gripper-map", choices=["pi", "viperx"], default="pi",
                   help="pi: PI 그리퍼 기준(0 닫힘~1 열림)을 0~40mm 에 직선 대응 / viperx: 기존 ViperX 링크 해석")
    p.add_argument("--reset-pose", choices=["staged", "aloha"], default="staged",
                   help="staged: Trossen 준비 자세 / aloha: ALOHA reset_pose 근사 (upstream 동작)")
    p.add_argument("--calib-mode", choices=["fingertip", "zshift"], default="fingertip",
                   help="fingertip: 손가락 끝·책상 높이·팔 간격 맞춤(시뮬레이션 검증) / zshift: 이전 높이만 이동")
    p.add_argument("--depth-shift-mm", type=float, default=0.0,
                   help="작업대 깊이 보정: 양수면 두 팔이 수건 먼 쪽으로 과하게 뻗는 것을 그만큼 앞으로 당김")
    p.add_argument("--extra-dz-mm", type=float, default=0.0,
                   help="높이 보정 미세 조정 (음수 = 더 낮게, 예: -10). 책상 누름 방지는 그대로")
    p.add_argument("--floor-margin-mm", type=float, default=None,
                   help="집는 높이보다 이만큼 아래까지만 허용 (책상 누름 방지)")
    p.add_argument("--view-warp", choices=list(VIEW_WARP), default="none",
                   help="카메라 시점을 PI ALOHA 처럼 보정: high(위 카메라) / wrist(손목 2대) / all(위+손목). "
                        "켤 때는 --depth-shift-mm 0 권장 (깊이 차이를 영상에서 보정하므로)")
    p.add_argument("--view-mask", action="store_true",
                   help="--view-warp 에 high 가 있을 때 위 카메라에서 책상 먼 쪽 너머(벽)를 검게 가림 (PI 는 검은 커튼)")
    p.add_argument("--collision-stop-mm", type=float, default=25.0,
                   help="양팔 충돌 회피 여유: 부위별 멈춤 간격(collision.py STOPS)에 더함. + 면 더 일찍 피함 "
                        "(0 은 실물에서 카메라-그리퍼가 닿았음, 20261105-195300)")
    p.add_argument("--no-collision-guard", action="store_true", help="양팔 충돌 방지 끄기")
    p.add_argument("--native", action="store_true",
                   help="이 로봇 관절로 학습한 모델(관제센터 녹화 파인튜닝)용: ALOHA 변환 없이 30Hz, 바닥 여유 20mm")
    p.add_argument("--hz", type=int, default=0, help="제어 주기 (기본: native 30, 아니면 50)")
    p.add_argument("--goal-time", type=float, default=None, help="명령마다 목표 도달 시간(초). 클수록 부드럽고 느림")
    main(p.parse_args())
