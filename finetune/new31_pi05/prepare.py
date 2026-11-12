"""Prepare immutable native snapshot and resized public-image cache; no robot imports."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import pyarrow.parquet as pq

PROJECT = Path('/home/dit/pi0_aloha_towel')
WORK = Path(__file__).resolve().parent
NATIVE_SOURCE = Path('/home/dit/control_center/data/datasets/task-e2b573c17dc9')
NATIVE = PROJECT / 'datasets/local/towel_new31_pi05_v3'
PUBLIC = PROJECT / 'datasets/public_raw/mo0821__aloha_fold_towel'
CACHE = PROJECT / 'datasets/cache/mo0821_towel_30hz_224'
CAMS = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
PUBLIC_CAMS = ['camera_high', 'camera_low', 'camera_wrist_left', 'camera_wrist_right']


def image224(rgb):
    h, w = rgb.shape[:2]
    scale = 224 / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    out = np.zeros((224, 224, 3), dtype=np.uint8)
    out[(224-nh)//2:(224-nh)//2+nh, (224-nw)//2:(224-nw)//2+nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out.transpose(2, 0, 1)


def exact_stats(x):
    x = np.asarray(x, np.float64)
    out = {k: fn(x, axis=0).tolist() for k, fn in [('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)]}
    out['count'] = [len(x)]
    for q in [1, 10, 50, 90, 99]:
        out[f'q{q:02}'] = np.percentile(x, q, axis=0).tolist()
    return out


def public_episode(path):
    cv2.setNumThreads(1)
    path = Path(path)
    stem = path.stem
    out = CACHE / (stem + '.npz')
    image_path = CACHE / (stem + '.images.npy')
    if out.exists() and image_path.exists():
        with np.load(out) as d:
            return {'episode': stem, 'frames': len(d['state']), 'source': str(path)}
    table = pq.read_table(path)
    state, action = [np.stack(table[k].to_numpy(zero_copy_only=False)).astype(np.float32) for k in ['observation.state', 'action']]
    n = int((len(state)-1) * 30 / 50) + 1
    target_t, source_t = np.arange(n)/30, np.arange(len(state))/50
    resample = lambda x: np.stack([np.interp(target_t, source_t, x[:, j]) for j in range(14)], axis=1).astype(np.float32)
    source_rows = np.minimum(np.round(target_t * 50).astype(int), len(state)-1)
    tmp = image_path.with_suffix('.tmp.npy')
    images = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(n, 4, 3, 224, 224))
    for c, cam in enumerate(PUBLIC_CAMS):
        column = table['observation.images.' + cam]
        for i, row in enumerate(source_rows):
            cell = column[int(row)].as_py()
            pixels = cv2.imdecode(np.frombuffer(cell['bytes'], np.uint8), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError(f'Image decode failed: {path}, {cam}, {row}')
            images[i, c] = image224(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
    images.flush()
    del images
    tmp.rename(image_path)
    np.savez(out, state=resample(state), action=resample(action), source_rows=source_rows)
    print(f'Public cache ready: {stem}, {n} frames', flush=True)
    return {'episode': stem, 'frames': n, 'source': str(path)}


def main():
    WORK.mkdir(exist_ok=True)
    info_path = NATIVE_SOURCE / 'meta/info.json'
    info = json.loads(info_path.read_text())
    assert info['total_episodes'] == 31 and info['total_frames'] == 57860
    assert info['fps'] == 30
    assert pq.read_table(NATIVE_SOURCE/'meta/tasks.parquet').to_pandas().index.tolist() == ['fold towel']
    rows = [r for p in sorted(NATIVE_SOURCE.glob('meta/episodes/**/*.parquet')) for r in pq.read_table(p).to_pylist()]
    rows.sort(key=lambda r:r['episode_index'])
    assert [r['episode_index'] for r in rows] == list(range(31)) and all(r['length'] >= 50 for r in rows)
    val = np.linspace(0, 30, 8).round().astype(int).tolist()
    train = [i for i in range(31) if i not in val]
    if not NATIVE.exists():
        NATIVE.mkdir(parents=True)
        for folder in ['meta', 'data']:
            shutil.copytree(NATIVE_SOURCE/folder, NATIVE/folder)
        (NATIVE/'videos').symlink_to(NATIVE_SOURCE/'videos', target_is_directory=True)
    elif not (NATIVE/'meta/info.json').exists():
        raise RuntimeError('Incomplete snapshot; inspect before rebuilding')
    assert json.loads((NATIVE/'meta/info.json').read_text())['total_frames'] == info['total_frames']
    data = [pq.read_table(p, columns=['episode_index', 'observation.state', 'action']) for p in sorted(NATIVE.glob('data/**/*.parquet'))]
    selected = {k:[] for k in ['observation.state','action']}
    for table in data:
        mask = np.isin(table['episode_index'].to_numpy(), train)
        for k in selected:
            x = np.stack(table[k].to_numpy(zero_copy_only=False))
            assert x.shape[1] == 14 and np.isfinite(x).all()
            selected[k].append(x[mask])
    native_stats = {k:exact_stats(np.concatenate(v)) for k,v in selected.items()}
    stats_path = NATIVE/'meta/stats.json'
    stats = json.loads(stats_path.read_text())
    stats.update(native_stats)
    stats_path.write_text(json.dumps(stats, indent=2)+'\n')
    CACHE.mkdir(parents=True, exist_ok=True)
    files = sorted(PUBLIC.glob('data/**/*.parquet'))
    assert len(files) == 30
    with ProcessPoolExecutor(max_workers=4) as pool:
        public = list(pool.map(public_episode, map(str, files)))
    public_values = {k:[] for k in ['observation.state','action']}
    for ep in public:
        with np.load(CACHE/(ep['episode']+'.npz')) as d:
            public_values['observation.state'].append(d['state'])
            public_values['action'].append(d['action'])
    public_stats = {k:exact_stats(np.concatenate(v)) for k,v in public_values.items()}
    train_frames = sum(rows[i]['length'] for i in train)
    public_frames = sum(ep['frames'] for ep in public)
    repeats = max(1, round(train_frames/public_frames))
    manifest = {'native_source':str(NATIVE_SOURCE),'native_root':str(NATIVE), 'native_repo':'local/towel_new31_pi05_v3',
        'native_source_info_sha256':hashlib.sha256(info_path.read_bytes()).hexdigest(),
        'native_total_episodes':31,'native_total_frames':info['total_frames'], 'train':train,'val':val,
        'native_train_frames':train_frames,'public_source':str(PUBLIC),'public_cache':str(CACHE),
        'public_episodes':public,'public_frames':public_frames,'public_repeats':repeats,
        'native_sampling_fraction':train_frames/(train_frames+repeats*public_frames),
        'native_stats_train_only':native_stats,'public_stats':public_stats,
        'native_task':'fold towel','public_task':'ALOHA robot: fold towel',
        'normalization':'Public native joint values are retained in cache. Per-source q01/q99 normalization is mapped into the native training normalizer coordinate system at sampling time. This is numeric normalization, not ALOHA-to-WidowX kinematic conversion.',
        'public_exclusions':['zhifeishen is towel spreading, not folding','virtualkss download incomplete'],
        'stages':[{'name':'mixed','updates':10000,'save_every':5000},{'name':'native','initialization':'mixed/010000','additional_updates':10000,'save_every':5000}],
        'validation':'Same 8 native episodes excluded from both stages and normalization fitting.'}
    (WORK/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:manifest[k] for k in ['native_train_frames','train','val','public_frames','public_repeats','native_sampling_fraction']},indent=2),flush=True)


if __name__ == '__main__':
    main()
