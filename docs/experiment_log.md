> 실험하면서 적어 둔 작업 기록을 그대로 옮긴 것입니다. 문서에 나오는 `HANDOFF.md`, `control_center`(별도 관제 프로그램), `base_model_run`, 체크포인트 경로는 이 저장소에 들어 있지 않습니다.

# Trossen AI Stationary 수건 접기: 모델 정리 (2026-11-10, 새 π0.5 학습 완료 확인)

모든 모델의 작업은 관제센터 "수건 접기"(task-e2499de8617c)다. 펼친 수건을 반으로 접고, 90도 돌려 한 번 더 접는다.
실물 결과는 관제센터 평가 기록(`control_center/data/center.json` assessments)과 이 세션의 실행 기록을 따른다.

새 접기 방식 시연은 **수건 접기 새 방식 (2026-11-06)** 작업(`task-e2b573c17dc9`, 지시문 `fold towel`)에 31개 저장됐다. F·G는 이 데이터로 새로 수행하는 π0.5 LoRA 실험이다. 8개는 양 단계 검증용으로 제외하고 23개를 학습한다. A~E는 새 데이터를 학습하지 않았다. 진행 상태는 `finetune/new31_pi05/status.json`, 상세 설정은 `HANDOFF.md` §1 참고.

## 1. 이 세션에서 만든 것과 쓴 것 (openpi / LeRobot π0·π0.5)

| # | 모델 | 시작 가중치 | 데이터 | 방식 | 상태 | 위치 | 결과 |
|---|---|---|---|---|---|---|---|
| A | `pi0_aloha_towel` (PI 공개) | — | PI 내부 ALOHA 수건 데이터 | 완성 모델, 학습 없음 | 사용 가능 | `openpi_data/openpi-assets/checkpoints/pi0_aloha_towel` | 실물: ALOHA↔WidowX 관절 변환과 카메라 시점 보정을 거쳐 실행. 수건 근처까지는 가지만 잡기 높이와 위치가 어긋남 (접기 실패) |
| B | π0 LoRA, 전체 156개 | `pi0_base` (openpi) | 관제센터 156개 (검증 분리 없음) | openpi LoRA | **폐기 대상**. 1,120 스텝에서 멈춤 (검증 분리 전) | `trossen_openpi/checkpoints/pi0_trossen_towel_cc/towel_cc_lora/1000` (8.7GB) | — |
| C | π0 LoRA, 학습 148 / 검증 8 | `pi0_base` (openpi) | 관제센터 148개, 프롬프트 `fold towel` | openpi LoRA, GitHub `pi0_trossen_organize_tools` 값 | 5,000 / 150,000 스텝에서 멈춤, 이어서 학습 가능 | `.../towel_cc_lora_split/5000` (8.7GB) | 검증: 팔 4.67도 (기준선 7.57), 그리퍼 1.72mm, 팔 방향 79%, 그리퍼 방향 91%. 과거 실물 충돌 후 회피 추가. 11/06 13:23·13:25·13:28 실행 모델은 서버 기록으로 추정. 사용자 총평: **나쁘지 않았음**. 재시험 `20261106-134558`은 **한쪽 모서리 잡기 실패로 수건이 꼬임** (사용자 확인). 영상·로그 분석은 `HANDOFF.md` §1 참고 |
| D | π0.5 LoRA, 학습 148 / 검증 8 | `lerobot/pi05_base` | 관제센터 148개. 5,000까지 `fold towel`, 이후 새 프롬프트(`finetune/PROMPT.txt`) | LeRobot pi05 + PEFT LoRA(r=16, 학습 파라미터 1.35M), 학습 단계 RTC | **완료** (30,000 스텝, 11/06 05:20) | `finetune/lerobot_runs/pi05_towel_lora/checkpoints/030000/pretrained_model` | 검증 30000: 팔 6.50도 (기준선 7.57), 그리퍼 2.96mm (기준선 2.61), 팔 방향 70%, 그리퍼 방향 83%. 실물 11/06 13:17·13:20은 서버 기록상 이 모델로 추정. 사용자 총평: **쓰기 힘들었음**. 영상 표본·로그 1차 분석 완료 (`HANDOFF.md` §1), 개별 성공 여부 미확인. 대시보드 `finetune/eval/dashboard_pi05_towel_lora.html` |
| E | π0.5 LoRA (openpi) | `pi05_base` (openpi) | 관제센터 148개 | openpi LoRA | 설정만 준비 (`pi05_trossen_towel_cc_lora`), 학습 안 함 | — | — |
| F | π0.5 새 방식 + 공개 접기 | `lerobot/pi05_base` | 새 시연 학습 23 / 검증 8 + mo0821 공개 접기 30, 프레임 비율 약 1:1 | language backbone·action expert q/v 및 입출력 LoRA, r=16/alpha=16, 학습 가능 3,195,904개 전부 LoRA | **완료** (1단계 10,000, 11/06 21:54) | `finetune/lerobot_runs/pi05_new31_mixed_lora/checkpoints/{005000,010000}` | 5000: 팔 6.81도·그리퍼 2.98mm, 10000: 팔 6.41도·그리퍼 2.81mm. 팔은 기준선 6.30도보다 나쁨 |
| G | π0.5 새 방식만 추가 학습 | F의 10,000 LoRA | 새 시연 학습 23 / 검증 8, 공개 데이터 제외 | F와 같은 LoRA를 이어서 학습, 기준 가중치 고정 | **완료** (추가 10,000, 누적 20,000, 11/07 01:34) | `finetune/lerobot_runs/pi05_new31_native_lora/checkpoints/{005000,010000}` | 추가5000: 팔 6.08도·그리퍼 2.75mm. 추가10000: 팔 **5.82도**·그리퍼 **2.67mm**, 팔 기준선보다 7.5% 개선. 새 4개 중 우선 실물 시험 후보, 실물 성능 미확인 |

## 2. 관제센터에서 예전에 학습한 모델 (LeRobot, 2026-10 ~ 11)

체크포인트는 `control_center/data/jobs/<job>/training/checkpoints/<step>/pretrained_model`에 있다.

| job | 정책 | 시연 수 | 스텝 / batch | 실물 평가 |
|---|---|---|---|---|
| **20261019-131608-2caf0f** | **SmolVLA** (smolvla_base) | 98 | 50,000 / 8 | **050000: 성공 5/5** ← 지금까지 최고 |
| 20261015-185954-fda907 | SmolVLA | 71 | 50,000 / 4 | 050000: 성공 2/3 (실패 1: 왼손 놓침) |
| 20261014-185847-4a66a1 | SmolVLA | 40 | 20,000 / 4 | 020000: 성공 2/4 (두 번째 접기에서 왼팔 놓침) |
| 20261014-152042-eee119 | SmolVLA | 24 | 10,000 / 4 | 010000: 성공 2/4 |
| 20261020-180022-6d3f9e | SmolVLA | 129 | 50,000 / 4 | 050000: 성공 1/2 |
| 20261014-173744-ce76ca | SmolVLA | 33 | 5,000 / 4 | 005000: 성공 1/1, 002500: 실패 1/1 |
| 20261016-175641-76fe73 | SmolVLA | 98 | 50,000 / 8 | 평가 기록 없음 |
| 20261019-185514-30e100 | SmolVLA | 98 | 100,000 / 4 | 평가 기록 없음 |
| 20261016-172218-c8fb57 | SmolVLA | 98 | 5,000 / 8 | 평가 기록 없음 |
| 20261016-133356-c779a8 | ACT | 74 | 50,000 / 8 | 평가 기록 없음 |
| 20261104-140434-47ea6e | π0.5 (lerobot/pi05_base) | 142 | 20,000 / **1** | 평가 기록 없음 (체크포인트 21개, 각 약 14MB = 어댑터) |
| 20261014-141428-3519e2 | π0 (lerobot/pi0_base) | 21 | 5,000 / 1 | 평가 기록 없음 |
| 20261015-111839-e7d91d | π0 (lerobot/pi0_base) | 41 | 40,000 / 1 (15,233에서 멈춤) | 평가 기록 없음 |

나머지는 초기 소규모 학습이다(SmolVLA 1,000~10,000 스텝, 시연 5~16개). 평가는 모두 실패이거나 기록이 없다.

## 3. 받아 둔 기본 모델 (학습 시작점)

| 모델 | 형식 | 위치 |
|---|---|---|
| `pi0_base` | openpi JAX | `openpi_data/openpi-assets/checkpoints/pi0_base` (12GB) |
| `pi05_base` | openpi JAX | `openpi_data/openpi-assets/checkpoints/pi05_base` (12GB) |
| `lerobot/pi05_base` | LeRobot PyTorch | `~/.cache/huggingface/hub/models--lerobot--pi05_base` (14GB) |
| `lerobot-pi0-aloha-towel` | `pi0_aloha_towel`을 LeRobot 형식으로 변환한 것 (학습 아님) | `~/.boim-work/public-towel-training-20261104/lerobot-pi0-aloha-towel` |

## 4. 실행 방법 요약

새 F·G 시험: `./trossen_aloha_real/serve_pi05_new31.sh <mixed|native> <5000|10000>`로 서버를 선택한다. 2026-11-10에는 **native 10000(누적20000)**을 서버로 준비하고 저장된 관측의 비구동 추론 3회를 확인했다(274~277ms). 실물은 사용자가 `./trossen_aloha_real/test_pi05_new31.sh`로 실행한다. 기본 10분, `Ctrl+C` 종료, 새 4개 모델은 모두 지시문 `fold towel`과 `--broker upstream --action-horizon 50`을 사용한다. 사용자 지시로 smooth를 해제하고 Trossen OpenPI 예제의 30Hz·50개 순차 실행·temporal ensembling 해제 기준으로 맞췄다. 한 번에 정책 서버 하나만 띄우고 관제센터는 자동으로 켜지 않는다.

G의 native 추가10000은 관제센터에도 등록했다. 작업 **수건 접기 새 방식 (2026-11-06)**에서 **PI05 · 20261107-013422-pi05-new31-native · 010000 스텝**을 선택한다. 관제센터 호환 사본의 가중치·정규화 값은 원본과 같고, 구형 설정·토크나이저 경로만 호환 처리했다. 관제센터 로더의 기본 가중치 로딩·LoRA 152개 텐서·기록 관측 추론을 확인했다. 관제센터와 터미널 모두 50개 행동을 순차 실행하지만 보호·기록 경로는 다르며, 해당 UI 경로에서의 실물 동작은 이번 등록 시 실행하지 않았다.

- **A (원래 π0 수건 모델):** `base_model_run/README.md`. ALOHA 변환을 쓰고 50Hz로 돈다.
- **C (π0 파인튜닝):** `trossen_aloha_real/rtc_server.py --config pi0_trossen_towel_cc --checkpoint <ckpt> --default-prompt "fold towel"`로 서버를 띄우고, 클라이언트는 사용자가 `main.py --native --broker rtc --max-episode-steps 2700`으로 실행한다. 30Hz, 충돌 회피가 켜져 있다. 5000에는 반드시 학습 당시 프롬프트 `fold towel`을 쓴다.
- **D (π0.5 파인튜닝):** 아래 30000 실행 명령을 쓴다. 학습은 완료됐으며, 추가 학습은 사용자 결정 전 시작하지 않는다. 이 추론 서버는 RTC 입력을 무시하므로 클라이언트에는 `--broker upstream --action-horizon 50`을 지정한다.
- **관제센터 모델:** 관제센터 UI에서 평가하고 실행한다.

π0.5 30000 서버 (다른 정책 서버와 포트 8000을 동시에 사용하지 않는다):

```bash
cd ~/pi0_aloha_towel/lerobot_pi05
.venv/bin/python ../trossen_aloha_real/lerobot_server.py --checkpoint ../finetune/lerobot_runs/pi05_towel_lora/checkpoints/030000/pretrained_model
```

π0.5 클라이언트 (**사용자만 실행**, 에이전트 실행 금지):

```bash
cd ~/pi0_aloha_towel/trossen_openpi/examples/trossen_ai
.venv/bin/python ~/pi0_aloha_towel/trossen_aloha_real/main.py --host 127.0.0.1 --native --broker upstream --action-horizon 50 --max-episode-steps 2700
```

π0.5 5000을 쓸 때는 체크포인트를 `005000/pretrained_model`로 바꾸고 서버에 `--task "fold towel"`을 추가한다. 10000~30000은 서버 기본값인 `finetune/PROMPT.txt`의 긴 프롬프트를 쓴다. 전체 인계 내용과 모델별 실행 명령은 `HANDOFF.md` 참고.

## 5. 정리 제안
- B(8.7GB)는 검증 분리 전 시험이라 지워도 된다.
- 관제센터 SmolVLA `20261019-131608-2caf0f/050000`이 실물 5/5로 기준선 역할을 한다. 새 모델(C, D)은 이 모델과 같은 조건(관제센터 평가)에서 비교해야 의미가 있다.
