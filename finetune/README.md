# 관제센터 수건 접기 데이터 — pi0 LoRA 파인튜닝

## 데이터
- 원본(읽기만): `~/control_center/data/datasets/task-e2499de8617c` (LeRobot v3.0, 159 에피소드)
- 변환본: `~/pi0_aloha_towel/datasets/local/trossen_towel_cc` (LeRobot v2.1, 8GB)
  - `convert_cc_to_v21.py`: 1프레임짜리 저장 실패 에피소드 3개를 뺀 156 에피소드, 140,848 프레임(30fps, 약 78분)
  - 과제 문장은 모두 `fold towel`. 원본 과제 문장(영어 107개, 한국어 52개)은 쓰지 않는다.
  - 영상은 에피소드별로 잘라 h264로 다시 인코딩했다. 원본 프레임과의 차이는 평균 0.9/255이다.
  - 값은 이 로봇(WidowX AI)의 관절 그대로다: 절대 관절각(rad), 그리퍼(m). ALOHA 변환을 하지 않는다.

## 학습 설정 `pi0_trossen_towel_cc` (trossen_openpi `src/openpi/training/config.py`)
GitHub의 `pi0_trossen_organize_tools`(긴 과제, π0 base, 카메라 4대, LoRA)를 복사했다. 바꾼 것은 아래 셋뿐이다.

| 항목 | GitHub | 여기 |
|---|---|---|
| repo_id | TrossenRoboticsCommunity/trossen_ai_stationary_organize_tools | local/trossen_towel_cc |
| default_prompt | Put all tools on table into the pen holder | fold towel |
| 정규화 값 | pi0_base assets 의 `trossen` | 이 데이터에서 계산 (`compute_norm_stats.py`) |

- 정규화 값을 바꾼 이유: base의 `trossen` 값은 구형 ALOHA형 Trossen(델타 행동, 그리퍼 0~1) 것이다.
  - 이 데이터를 그 값으로 정규화하면 행동이 ±15까지 나온다(정상은 약 ±1).
  - 그리퍼는 전체 범위가 0.08밖에 움직이지 않아 거의 학습되지 않는다.
- 그대로인 값
  - 모델: `gemma_2b_lora` + `gemma_300m_lora`, LoRA freeze filter
  - 가중치: `pi0_base`
  - 학습: 150,000 스텝, batch 8, EMA 끔
  - 그 밖의 기본값: AdamW(clip 1.0), cosine lr(warmup 1000, peak 2.5e-5, 30,000 스텝에 걸쳐 2.5e-6까지), 1,000 스텝마다 저장, 5,000 스텝마다 보관
  - 행동 형식: `use_delta_joint_actions=False`, `adapt_to_pi=False`

## 시연을 더 모은 뒤: 데이터 준비 한 번에 (2026-11-19 추가)
관제센터에서 시연을 더 모았으면 아래 한 줄로 변환 → 학습/검증 분리 → 정규화 값 계산까지 한다.
```bash
cd ~/pi0_aloha_towel
trossen_openpi/.venv/bin/python finetune/prepare_data.py --dry-run   # 무엇이 바뀔지만 본다
trossen_openpi/.venv/bin/python finetune/prepare_data.py             # 실제로 준비한다
```
- 아직 변환하지 않은 에피소드만 변환해 `trossen_towel_cc` 뒤에 덧붙인다 (에피소드 6개에 약 20초). 이미 변환한 에피소드의 번호는 바뀌지 않는다.
- 검증 에피소드는 변환본 번호가 20으로 나눠 10이 남는 것이다. 기존 8개(10, 30, …, 150)는 그대로이고, 20개가 늘 때마다 1개씩 추가된다.
- **관제센터에서 '제외'로 표시한 에피소드는 학습·검증에서 뺀다.** 기존 156개 중 14개가 '제외' 표시인데 40000 체크포인트는 이들을 포함해 학습했다. 예전처럼 넣으려면 `--keep-excluded`.
- 정규화 값은 학습 데이터가 바뀌었을 때만 다시 계산한다 (12~17분, GPU 를 쓰지 않으므로 정책 서버를 켜 둔 채로 가능). 이전 값은 `finetune/backups/`에 남긴다. 이미 저장된 체크포인트는 자기 정규화 값을 따로 갖고 있어 영향받지 않는다.
- 다른 관제센터 작업을 함께 쓰려면 `--tasks "수건 접기" "작업 이름 또는 ID"`. 카메라·관절 구성이 다르면 거절한다. 지시문은 모두 `fold towel`이 된다.
- 원본은 읽기만 한다. 관제센터 쪽 에피소드 번호가 바뀐 흔적(길이 불일치)이 있으면 멈춘다.
- 검증(2026-11-19): 변환본을 150개로 줄인 사본에서 실행해 나머지 6개를 새로 변환했고, 변환본·학습·검증 세 데이터셋의 값·meta·영상 화면이 기존 것과 모두 같았다. 정규화 단계도 정책 서버를 켠 채로 돌려 기존 값과 바이트 단위로 같은 파일이 나왔다.

## 실행
```bash
~/pi0_aloha_towel/finetune/train.sh                              # 처음부터 (--overwrite)
~/pi0_aloha_towel/finetune/train.sh --resume                     # 이어서
EXP=towel_cc_lora_1201 ~/pi0_aloha_towel/finetune/train.sh       # 새 데이터로 다시 학습 (새 실험 이름)
```
- 이미 체크포인트가 있는 실험 이름으로 처음부터 학습하려 하면 거절한다 (40000 체크포인트가 지워지는 것을 막기 위해). 정말 지우려면 `FORCE=1`.
- 데이터를 바꾼 뒤에는 이전 실험을 `--resume`으로 잇지 않는다 (정규화 값이 달라진다).
- 새 실험 이름으로 학습한 체크포인트는 `eval_watch.py --exp <이름>`으로 검증한다. `serve_pi0_ckpt.sh`는 아직 `towel_cc_lora_split`만 본다.
- GPU(32GB)를 거의 다 쓰므로 RTC 정책 서버를 먼저 꺼야 한다.
- 체크포인트는 `trossen_openpi/checkpoints/pi0_trossen_towel_cc/towel_cc_lora/<스텝>`에 저장된다.
- 이 모델은 WidowX AI 관절을 그대로 입·출력한다. 그래서 실행은 ALOHA 변환 클라이언트(`trossen_aloha_real`)가 아니라 Trossen 클라이언트 방식(`examples/trossen_ai/main.py`)으로 한다.
