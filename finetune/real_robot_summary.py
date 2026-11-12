"""실제 로봇 수건 접기 시험 점수 요약 (finetune/eval/real_robot/pi0_towel_trials.csv).

단계: 0 못 잡음 / 1 잡음 / 2 첫 반 접기 / 3 돌리기 / 4 두 번째 반 접기 완료. 표준 라이브러리만 쓴다.
"""
import csv
import pathlib
from collections import defaultdict

path = pathlib.Path(__file__).resolve().parent / "eval/real_robot/pi0_towel_trials.csv"
rows = defaultdict(list)
with path.open() as f:
    for r in csv.DictReader(f):
        rows[int(r["step"])].append(r)
print("\n체크포인트별 실제 로봇 결과 (각 단계까지 도달한 횟수/시도)")
print(f"{'스텝':>7} {'시도':>4} {'잡기':>7} {'1차접기':>7} {'돌리기':>7} {'2차접기':>7} {'깔끔':>6} {'평균단계':>7}")
for step in sorted(rows):
    r = rows[step]
    n = len(r)
    stage = [int(x["stage_reached"]) for x in r]
    reach = [sum(s >= k for s in stage) for k in (1, 2, 3, 4)]
    tidy = sum(x["tidy"] == "y" for x in r)
    cells = " ".join(f"{c:>3}/{n:<3}" for c in reach)
    print(f"{step:>7} {n:>4} {cells} {tidy:>3}/{n:<3} {sum(stage) / n:>6.2f}")
