# openpi `aloha_real` Towel Task — Trossen AI Stationary 버전

[openpi examples/aloha_real](https://github.com/Physical-Intelligence/openpi/tree/main/examples/aloha_real)의 Towel Task를 같은 방식으로 실행한다.
서버 명령은 README 그대로이고, 클라이언트 구조(Runtime, ActionChunkBroker 25스텝, 50Hz, 최대 1000스텝)도 같다.
upstream과 다른 것은 하드웨어 계층뿐이다.

| | upstream `aloha_real` | 이 폴더 |
|---|---|---|
| 로봇 연결 | ROS Noetic + Interbotix (ViperX 300 S, USB) | lerobot BiWidowXAIFollower (WidowX AI, 이더넷) |
| 관절 | ALOHA 관절 그대로 | `retarget.py`로 ALOHA ↔ WidowX AI 변환 (팔 끝 자세 기준 FK/IK, 그리퍼 링크 기하 변환) |
| 카메라 | ROS realsense publisher | lerobot RealSense (`~/trossen_setup/stationary-teleop.json`의 시리얼) |
| 초기 자세 | `reset_pose` [0,-1.5,1.5,0,0,0], 1초 | 같은 ALOHA 자세를 WidowX AI 정상 자세 계열의 가장 가까운 자세(약 2cm·10° 차이)로 근사, 3초 |
| CLI | tyro | argparse (옵션 이름 같음) |

## 실행

전제: 관제센터가 꺼져 있어야 한다(카메라·팔 공유 불가).

```bash
systemctl --user stop control-center.service
```

터미널 1 — 정책 서버 (aloha_real README의 Towel Task 체크포인트와 프롬프트):

```bash
cd ~/pi0_aloha_towel/trossen_openpi && uv run scripts/serve_policy.py --default_prompt='fold the towel' policy:checkpoint --policy.config=pi0_aloha_towel --policy.dir=$HOME/pi0_aloha_towel/openpi_data/openpi-assets/checkpoints/pi0_aloha_towel
```

터미널 2 — 로봇 클라이언트. 서버 탭이 아닌 **다른 터미널**에서 실행한다. 로봇 앞에서, 비상정지를 쥐고:

> 클라이언트 환경의 `trossen-arm`은 컨트롤러 펌웨어(v1.11.1)에 맞춰 1.11.0으로 바꿔 두었다. Trossen 포크 기본값은 1.10.0이고, 이 버전으로는 버전 불일치로 연결이 거절된다.
> `uv run`은 lock 파일 기준으로 1.10.0을 다시 설치하므로, 반드시 `.venv/bin/python`으로 실행한다.

```bash
cd ~/pi0_aloha_towel/trossen_openpi/examples/trossen_ai && .venv/bin/python ~/pi0_aloha_towel/trossen_aloha_real/main.py
```

동작 순서는 upstream과 같다.
1. 연결하면 준비 자세로 이동한다(lerobot 기본 동작).
2. 초기 자세로 이동한 뒤 그리퍼를 닫았다 연다.
3. 1000스텝(50Hz 기준 20초) 동안 정책을 실행한다.
4. 다시 초기 자세로 이동한다.
5. 종료하면 준비 자세를 거쳐 휴식 자세로 간다.

옵션:
- `--max-episode-steps 300`: 짧게 시험할 때
- `--dry-run`: 모델 동작과 초기 자세 이동을 팔에 보내지 않는다. 연결·종료 시 lerobot의 준비·휴식 자세 이동은 그대로 한다.

마치면 관제센터를 다시 켠다.

```bash
systemctl --user start control-center.service
```

## RTC (기본값 `--broker rtc`) — 관제센터에서 옮김

관제센터가 11/3~11/4 오전 실물 실행에 쓴 Real-Time Chunking을 옮겼다(30Hz, 25스텝마다 재계획).

- `rtc_model.py`: `control_center/pi0_rtc_model.py`를 그대로 복사했고, 상수 import만 바꿨다. 새 묶음을 샘플링할 때 이전 묶음의 남은 동작과 맞도록 유도한다(VJP guidance, Black et al. 2025).
- `rtc_server.py`: openpi websocket 프로토콜 그대로이고, 관측과 함께 `rtc_previous / rtc_prefix_length / rtc_delay`를 받는다. 관제센터 RTC를 검증한 커밋과 같은 openpi 215abfb 환경(`~/pi0_aloha_towel/openpi`)에서 실행한다.
- `broker.RtcChunkBroker`: `control_center/pi0_deployment.py`의 스케줄을 옮겼다.
  - 25스텝마다 백그라운드에서 추론하고, 그동안 이전 묶음을 계속 실행한다.
  - 응답이 오면 지나간 스텝만큼 건너뛴다.
  - 지연 예측은 최근 10회 최대값 + 2스텝이다.

실행:

```bash
# 터미널 1 — RTC 정책 서버 (유도 추론 약 93ms)
cd ~/pi0_aloha_towel/openpi && .venv/bin/python ../trossen_aloha_real/rtc_server.py
# 터미널 2 — 클라이언트 (기본 --broker rtc)
cd ~/pi0_aloha_towel/trossen_openpi/examples/trossen_ai && .venv/bin/python ~/pi0_aloha_towel/trossen_aloha_real/main.py --max-episode-steps 300
```

`--broker smooth`나 `--broker upstream`은 RTC 서버에서도 동작한다. 관측에 RTC 키가 없으면 일반 추론을 한다.

## π0.5 실행 기준 (2026-11-10 사용자 지시)

`test_pi05_new31.sh`는 Trossen의 OpenPI 예제 기준으로 `--native --broker upstream --action-horizon 50`을 사용한다. 30Hz로 행동 50개를 순서대로 실행한 뒤 다음 추론을 요청하며, 사전 추론과 행동 혼합은 사용하지 않는다. 기존 바닥·작업 영역·충돌 보호와 종료 시 홈 복귀는 유지한다.

근거: https://docs.trossenrobotics.com/trossen_arm/main/tutorials/openpi.html 의 `rate_of_inference = 50`, temporal ensembling `None`. 일반 ALOHA 예제의 25스텝·50Hz와 구분한다.

## 기존 smooth 구현 (π0.5 기본 실행에서는 사용하지 않음)

upstream ActionChunkBroker는 실물에서 팔이 0.5초마다 "팍팍" 끊기며 움직였다. 원인은 세 가지였다.
- 25스텝마다 추론하는 약 115ms 동안 명령이 끊긴다.
- 새 동작 묶음이 이전 묶음과 이어지지 않아 최대 0.11rad 튄다.
- 명령마다 목표 도달 시간이 0.06초라 동작이 툭툭 꺾인다.

`broker.py`의 SmoothChunkBroker로 다음과 같이 바꿨다.
- 묶음이 끝나기 8스텝 전에 다음 추론을 미리 시작한다.
- 묶음이 바뀔 때 이전 묶음이 예측해 둔 뒷부분과 새 묶음을 8스텝 동안 섞는다.
- 목표 도달 시간은 `--goal-time 0.1`(초)이다.

가짜 로봇에서 300스텝을 비교한 결과:

| | upstream | smooth |
|---|---|---|
| 명령당 최대 관절 변화 | 0.118rad | 0.038rad |
| 2차 차분 최대(덜컹거림) | 0.116 | 0.037 |
| 가장 긴 명령 공백 | 149ms | 79ms |

upstream과 똑같이 돌리려면 `--broker upstream --goal-time 0.06`을 쓴다.

카메라 프레임이 200ms 넘게 늦으면 lerobot이 TimeoutError를 낸다. 실물에서 cam_high(260522277170)에서 실제로 발생했다. 이 경우 바로 멈추지 않고 최대 5번 다시 읽는다.

## upstream에 없는 보호 장치 (변환 계층 때문에 필요한 것)

- 변환 결과가 WidowX AI 관절·그리퍼 범위를 벗어나면 정지한다.
- 테이블 평면(z = -20mm) 아래를 향하는 목표는 정지한다.
- 정확한 IK 해가 없으면(작업 공간 경계) 직전 자세 근처의 가장 가까운 해를 쓴다. 그래도 25mm·12°를 넘으면 정지한다.
- 변환 결과가 직전 명령에서 0.25rad 넘게 튀면(IK 해 전환) 정지한다.
- 정상 종료, 오류, Ctrl-C 모두 양팔을 동시에 준비 자세 → 휴식 자세로 각 5초씩 천천히 돌려보낸 뒤 모터를 해제(idle)한다. 공중에서 브레이크를 걸면 모터에 부하가 걸리기 때문이다. 복귀 자체가 실패하거나(드라이버 오류 등) 복귀 중 Ctrl-C를 한 번 더 누르면 그 자리에서 idle로 멈춘다.
- 측정값이 범위 경계를 아주 조금 벗어나는 경우(관절 0.01rad, 그리퍼 0.01m 이하)는 경계 값으로 맞춘다. 첫 실물 실행에서 리셋 첫 명령이 이 문제로 정지했다.

## 오프라인 검증 (2026-11-04, 가짜 로봇 + 실제 정책 서버 + 녹화된 실제 카메라 프레임)

- 150스텝 동안 관절 변환 실패나 정지가 없었다.
  - 명령당 최대 관절 변화는 0.079rad였다.
  - 0.15rad 넘게 튄 명령은 0건이었다.
- 제어 속도는 약 39Hz였다(목표 50Hz). 25스텝마다 한 번 있는 추론 구간에서 약 150ms가 걸린다.
- 수정 전에는 문제가 있었다. ALOHA 초기 자세를 IK로 근사하면서 베이스 −90°·손목 −90°의 비정상 해로 빠졌고, 명령 간 관절이 최대 1.17rad 튀었다. 근사 해를 직전 자세 근처로 붙잡는 항과 튐 검사를 넣어 해결했다.

## 작업 영역 제한·시작 자세 (2026-11-04 22:20)

- `workspace.py`: 캘리브레이션 때 손으로 짚은 점들의 범위에 여유 20mm를 더해 팔마다 상자를 만든다. 두 팔은 대칭이라, 반대 팔의 점을 y 방향으로 뒤집어 함께 쓴다. 그리퍼 기준점이 상자 밖이나 z 350mm 위로 가면 경계로 자른다. 정지하지는 않는다.
  - 현재 상자: 왼팔 x 141~709, y −318~285mm / 오른팔 x 141~709, y −285~318mm
  - 기물에 부딪힐 뻔한 실행(`20261104-221619`)에서는 두 팔이 y −408 / +447mm까지 나갔다. 지금 기준이면 그 실행 명령의 약 6%가 경계에서 잘린다.
  - 끄려면 `--no-workspace`, 여유는 `--workspace-margin-mm`로 바꾼다.
- 시작 자세: 기본값을 `--reset-pose staged`(Trossen 준비 자세)로 바꿨다. ALOHA reset_pose를 근사하면 어깨를 세우고 손목을 꺾은 자세가 나오기 때문이다. upstream 동작대로 하려면 `--reset-pose aloha`를 쓴다.
- 아직 없는 것: 두 팔끼리의 충돌 방지. 상자 x 범위가 책상 반대편까지라 두 팔이 가운데서 만날 수 있다.

## 그리퍼 변환 수정 (2026-11-04 22:35, `gripper_map.py`, 기본 `--gripper-map pi`)

pi0(PI) 그리퍼 기준은 PI 하드웨어 모터 각도 0.5476~1.6296rad(엔코더 2405~3110)를 0(닫힘)~1(열림)로 쓴다(openpi `aloha_policy.py`).
기존 `retarget.py`는 모델 출력을 일반 ALOHA ViperX 그리퍼 링크로 해석했다. 그래서 다음 문제가 생겼다.
- 모델의 "완전히 닫힘"(출력 0.553)이 WidowX AI 24mm(60% 열림)가 됐다.
- 관측도 반대로 어긋나서, 24mm를 모델은 "완전히 닫힘"으로 봤다.
- 실제로 모든 실행에서 모델 그리퍼 명령 최솟값이 정확히 0.55였다. 모델은 닫으라고 하는데 그리퍼는 반쯤 열린 채였다.

수정: PI 기준 0~1을 WidowX AI 0~40mm에 직선으로 대응시킨다. openpi의 실제 변환 함수로 검증했다.

| | 이전 | 수정 후 |
|---|---|---|
| 모델 "닫힘" | 24.0mm | 0.0mm |
| 모델 "반" | 34.9mm | 18.5mm |
| 0mm를 모델이 보는 값 | −1.08 | 0.00 |
| 24mm를 모델이 보는 값 | 0.00 | 0.60 |

예전 방식은 `--gripper-map viperx`로 쓸 수 있다.

## 카메라 시점 보정 (2026-11-05, `viewpoint.py`, `--view-warp`, 기본 끔)

PI ALOHA와 이 로봇은 카메라 위치·화각이 다르다. 그래서 모델 입력 직전에 영상을 PI 시점처럼 바꾼다. 추정 과정은 `calibration/README.md` 7단계, 보정 값은 `view_warp.json`에 있다.

| 옵션 | 보정하는 카메라 |
|---|---|
| `--view-warp high` | cam_high (책상면 기준) |
| `--view-warp wrist` | 손목 2대 (손가락 끝 앞 20mm 평면 기준) |
| `--view-warp all` | 위 + 손목 (cam_low는 보정하지 않음) |
| `--view-mask` | cam_high에서 책상 먼 쪽 너머(벽)를 검게 가림 (PI는 검은 커튼) |

- 영상으로 깊이 차이를 보정하므로 `--depth-shift-mm 0`과 함께 쓴다. depth-shift는 시점 차이 때문에 먼 쪽으로 과하게 뻗던 것을 막으려고 넣은 임시 보정이었다.
- 기록(`runs/*/model_inputs.mp4`)에는 보정된 모델 입력이 저장된다.

```bash
cd ~/pi0_aloha_towel/trossen_openpi/examples/trossen_ai && .venv/bin/python ~/pi0_aloha_towel/trossen_aloha_real/main.py --view-warp all --view-mask --depth-shift-mm 0 --max-episode-steps 15000
```
