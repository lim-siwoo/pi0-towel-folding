"""Episode-safe, source-normalized mixture; public joints are never retargeted."""
from bisect import bisect_right
from copy import copy, deepcopy
from functools import lru_cache
import json
from pathlib import Path

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from lerobot.datasets import LeRobotDataset

from prepare import CAMS, WORK, image224


def manifest():
    return json.loads((WORK/'manifest.json').read_text())


class Native(Dataset):
    def __init__(self, episodes):
        m = manifest()
        self.episodes = list(episodes)
        self.ds = LeRobotDataset(m['native_repo'], root=m['native_root'], episodes=self.episodes,
            delta_timestamps={'action':[i/30 for i in range(50)]}, video_backend='pyav', return_uint8=True)
        self.meta = self.ds.meta
        self.lengths = [int(self.meta.episodes[e]['length']) for e in self.episodes]
        self.meta.stats.update({k:{s:np.asarray(v) for s,v in value.items()} for k,value in m['native_stats_train_only'].items()})

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        frame = self.ds[index]
        out = {k:frame[k] for k in ['observation.state','action','action_is_pad']}
        for cam in CAMS:
            value = frame['observation.images.'+cam]
            if value.dtype != torch.uint8:
                value = (value*255).round().clamp(0,255).to(torch.uint8)
            out['observation.images.'+cam] = torch.from_numpy(image224(value.numpy().transpose(1,2,0)).copy())
        out['task'] = 'fold towel'
        return out


@lru_cache(maxsize=3)
def public_arrays(cache, stem):
    with np.load(Path(cache)/(stem+'.npz')) as d:
        state, action = d['state'].copy(), d['action'].copy()
    images = np.load(Path(cache)/(stem+'.images.npy'), mmap_mode='r')
    return state, action, images


class Mixture(Dataset):
    def __init__(self, stage):
        m = manifest()
        self.native = Native(m['train'])
        self.public_cache = m['public_cache']
        self.transforms = {}
        for key in ['observation.state','action']:
            p, n = m['public_stats'][key], m['native_stats_train_only'][key]
            pl, ph, nl, nh = [np.asarray(v,np.float32) for v in [p['q01'],p['q99'],n['q01'],n['q99']]]
            self.transforms[key] = (pl, np.maximum(ph-pl,1e-6), nl, np.maximum(nh-nl,1e-6))
        self.segments = []
        offset = 0
        rows = []
        def append(kind, source, source_start, length):
            nonlocal offset
            self.segments.append((kind,source,source_start,offset,length))
            rows.append({'episode_index':len(rows),'dataset_from_index':offset,'dataset_to_index':offset+length,
                         'length':length,'tasks':['fold towel' if kind=='native' else m['public_task']]})
            offset += length
        relative = 0
        for ep, length in zip(m['train'], self.native.lengths):
            append('native',ep,relative,length)
            relative += length
        if stage == 'mixed':
            for _ in range(m['public_repeats']):
                for ep in m['public_episodes']:
                    append('public',ep['episode'],0,ep['frames'])
        self.ends = [s[3]+s[4] for s in self.segments]
        self.num_frames = offset
        self.num_episodes = len(rows)
        self.meta = copy(self.native.meta)
        self.meta.info = deepcopy(self.meta.info)
        self.meta.info.total_frames = offset
        self.meta.info.total_episodes = len(rows)
        self.meta.episodes = datasets.Dataset.from_list(rows)
        self.episodes = list(range(len(rows)))
        self.absolute_to_relative_idx = None
        self.image_transforms = None

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        kind, source, source_start, offset, length = self.segments[bisect_right(self.ends,index)]
        local = index-offset
        if kind == 'native':
            return self.native[source_start+local]
        state, action, images = public_arrays(self.public_cache,source)
        ids = np.minimum(local+np.arange(50),length-1)
        def canonical(key, value):
            pl, width, nl, native_width = self.transforms[key]
            return torch.from_numpy(((value-pl)/width*native_width+nl).astype(np.float32))
        out = {'observation.state':canonical('observation.state',state[local]),
               'action':canonical('action',action[ids]),
               'action_is_pad':torch.from_numpy(local+np.arange(50)>=length),
               'task':'ALOHA robot: fold towel'}
        for i,cam in enumerate(CAMS):
            out['observation.images.'+cam] = torch.from_numpy(np.array(images[local,i],copy=True))
        return out
