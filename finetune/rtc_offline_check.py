"""Robot-free check of the control-center pi0.5 RTC engine on a recorded episode.

Replays episode frames at 30 Hz through FrameRTCEngine exactly like worker.py does
(one get_action per tick), and reports per-tick latency (stalls), arm error vs demo,
and that reset() discards queued actions.
"""
import sys, time, json
import numpy as np

sys.path.insert(0, '/home/dit/control_center')
import native_policy_runtime as rt  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402

CK = sys.argv[1]
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 20
ROOT = '/home/dit/control_center/data/datasets/task-e2b573c17dc9'

ds = LeRobotDataset('local/towel_cc_new', root=ROOT, episodes=[EP], video_backend='pyav')
feat = {k: dict(v) for k, v in ds.meta.features.items()}
# Same feature dict shape the worker passes (images HWC uint8).
features = {k: v for k, v in feat.items() if k.startswith('observation.') or k == 'action'}

loaded = rt.load_policy(CK)
engine = rt.inference_engine(loaded, features, 'fold towel', 'bi_widowx_ai')
print('engine', type(engine).__name__, flush=True)

def frame(i):
    f = ds[i]
    out = {'observation.state': f['observation.state'].numpy()}
    for k, v in f.items():
        if k.startswith('observation.images.'):
            out[k] = (v.numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return out, f['action'].numpy()

# warm-up like PolicyCache.prepare
dummy = {k: np.zeros(v['shape'], np.uint8 if v['dtype'] == 'video' else np.float32)
         for k, v in features.items() if k.startswith('observation.')}
engine.get_action(dummy); engine.reset()

n = min(int(SECONDS * 30), len(ds))
frames = [frame(i) for i in range(n)]
lat, errs, acts = [], [], []
start = time.monotonic()
for i, (obs, demo) in enumerate(frames):
    tick = time.monotonic()
    a = engine.get_action(obs).numpy()
    acts.append(a)
    lat.append(time.monotonic() - tick)
    errs.append(np.degrees(np.abs(a[[0,1,2,3,4,5,7,8,9,10,11,12]] - demo[[0,1,2,3,4,5,7,8,9,10,11,12]])).mean())
    time.sleep(max(0, start + (i + 1) / 30 - time.monotonic()))
lat = np.array(lat) * 1000
ARM = [0,1,2,3,4,5,7,8,9,10,11,12]
acts = np.array(acts)
first_gap = np.abs(acts[0][ARM] - frames[0][0]['observation.state'][ARM])
step = np.abs(np.diff(acts[:, ARM], axis=0)) * 30  # rad/s if each tick is 1/30 s
print(json.dumps({
    'first_cmd_vs_state_max_rad': round(float(first_gap.max()), 3),
    'cmd_speed_max_rad_s_per_joint': [round(float(x), 2) for x in step.max(0)],
    'cmd_speed_max_rad_s': round(float(step.max()), 2),
}), flush=True)
print(json.dumps({
    'ticks': n, 'first_ms': round(lat[0], 1),
    'after_first_ms_p50_p99_max': [round(float(x), 1) for x in np.percentile(lat[1:], [50, 99, 100])],
    'ticks_over_100ms': int((lat[1:] > 100).sum()),
    'arm_err_deg_mean_vs_demo': round(float(np.mean(errs)), 2),
}), flush=True)
engine.reset()
assert engine.engine is None, 'reset must drop the RTC thread and queue'
a = engine.get_action(frames[0][0])
print('reset_then_first_action_ok', a.shape, flush=True)
engine.reset()
