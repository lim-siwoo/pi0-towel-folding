"""오프라인 wandb 기록(.wandb)에서 학습 지표(loss, grad_norm, param_norm)를 꺼낸다.
사용 (trossen_openpi 폴더): .venv/bin/python ~/pi0_aloha_towel/finetune/read_wandb_loss.py [run.wandb] [--csv out.csv]"""
import glob
import json
import sys

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

path = next((a for a in sys.argv[1:] if a.endswith(".wandb")), None) or sorted(glob.glob("wandb/offline-run-*/run-*.wandb"))[-1]
csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
ds = datastore.DataStore()
ds.open_for_scan(path)
rows = []
while True:
    data = ds.scan_data()
    if data is None:
        break
    rec = wandb_internal_pb2.Record()
    rec.ParseFromString(data)
    if rec.WhichOneof("record_type") == "history":
        d = {it.key or "/".join(it.nested_key): json.loads(it.value_json) for it in rec.history.item}
        rows.append(d)
keys = ["loss", "grad_norm", "param_norm"]
rows = [r for r in rows if "loss" in r]
print("step," + ",".join(keys))
lines = [f"{r.get('_step')}," + ",".join(f"{r.get(k, float('nan')):.5f}" for k in keys) for r in rows]
print("\n".join(lines))
if csv:
    open(csv, "w").write("step," + ",".join(keys) + "\n" + "\n".join(lines) + "\n")
