"""trossen_openpi scripts/compute_norm_stats.py 와 같은 계산을 데이터 로더 worker 수만 늘려서 돌린다.
(원본은 config.num_workers=2 를 써서 영상 디코딩 때문에 1.5시간 넘게 걸림. 결과·저장 위치는 같다)
사용 (trossen_openpi 폴더에서): .venv/bin/python ~/pi0_aloha_towel/finetune/compute_norm_stats_fast.py <config> <workers>
"""
import sys

sys.path.insert(0, "scripts")
import compute_norm_stats as C  # noqa: E402
import numpy as np  # noqa: E402
import tqdm  # noqa: E402

from openpi.shared import normalize  # noqa: E402
from openpi.training import config as _config  # noqa: E402

def main():
    name, workers = sys.argv[1], int(sys.argv[2])
    config = _config.get_config(name)
    data_config = config.data.create(config.assets_dirs, config.model)
    loader, n = C.create_torch_dataloader(data_config, config.model.action_horizon, config.batch_size, config.model, workers, None)
    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=n, desc="Computing stats", mininterval=30):
        for k in stats:
            stats[k].update(np.asarray(batch[k]))
    out = config.assets_dirs / data_config.repo_id
    print("Writing stats to:", out)
    normalize.save(out, {k: s.get_statistics() for k, s in stats.items()})


if __name__ == "__main__":  # 데이터 로더가 spawn 방식이라 필요
    main()
