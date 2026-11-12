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

## 실행
```bash
~/pi0_aloha_towel/finetune/train.sh            # 처음부터 (--overwrite)
~/pi0_aloha_towel/finetune/train.sh --resume   # 이어서
```
- GPU(32GB)를 거의 다 쓰므로 RTC 정책 서버를 먼저 꺼야 한다.
- 체크포인트는 `trossen_openpi/checkpoints/pi0_trossen_towel_cc/towel_cc_lora/<스텝>`에 저장된다.
- 이 모델은 WidowX AI 관절을 그대로 입·출력한다. 그래서 실행은 ALOHA 변환 클라이언트(`trossen_aloha_real`)가 아니라 Trossen 클라이언트 방식(`examples/trossen_ai/main.py`)으로 한다.
