"""Native held-out episode predictions vs demonstration and hold-position baseline."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from data import Native, manifest

ARM = [0,1,2,3,4,5,7,8,9,10,11,12]
GRIP = [6,13]


def metrics(pred, gt, state, mask):
    pred,gt,state,mask = map(np.asarray,[pred,gt,state,mask])
    out = {}
    for name, value in [('model',pred),('hold',np.repeat(state[:,None,:],50,axis=1))]:
        errors = np.abs(value-gt)
        arm = np.degrees(errors[...,ARM]).mean(-1)
        grip = (errors[...,GRIP]*1000).mean(-1)
        out[name] = {'arm_deg_mean':float(arm[mask].mean()),'grip_mm_mean':float(grip[mask].mean()),
            'arm_deg_at_step':{str(k+1):float(arm[mask[:,k],k].mean()) for k in [0,9,24,49]},
            'grip_mm_at_step':{str(k+1):float(grip[mask[:,k],k].mean()) for k in [0,9,24,49]}}
    last = mask.sum(1)-1
    end_gt,end_pr = gt[np.arange(len(gt)),last],pred[np.arange(len(gt)),last]
    cosines,ratios = [],[]
    for side in [ARM[:6],ARM[6:]]:
        dg,dp = end_gt[:,side]-state[:,side],end_pr[:,side]-state[:,side]
        norm = np.linalg.norm(dg,axis=1)
        moving = np.degrees(norm)>5
        cos = (dg*dp).sum(1)/(norm*np.linalg.norm(dp,axis=1)+1e-9)
        cosines.extend(cos[moving]);ratios.extend((np.linalg.norm(dp,axis=1)/(norm+1e-9))[moving])
    out['arm_direction'] = {'moves':len(cosines),'same_direction_ratio':float(np.mean(np.asarray(cosines)>0.7)) if cosines else None,
        'cos_median':float(np.median(cosines)) if cosines else None,'move_size_ratio_median':float(np.median(ratios)) if ratios else None}
    dg,dp = end_gt[:,GRIP]-state[:,GRIP],end_pr[:,GRIP]-state[:,GRIP]
    events = np.abs(dg)>0.003
    out['gripper_direction'] = {'events':int(events.sum()),'same_direction':int((np.sign(dg[events])==np.sign(dp[events])).sum())}
    out['samples'] = len(gt)
    return out


@torch.no_grad()
def evaluate(policy, pre, post, stage, step, output):
    m = manifest()
    was_training = policy.training
    policy.eval()
    base = policy.get_base_model() if hasattr(policy,'get_base_model') else policy
    result = {'stage':stage,'step':step,'checkpoint':str(output),'native_task':'fold towel','every_frames':90}
    started = time.time()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(12345)
            for name,eps in [('val',m['val']),('train',m['train'][:8])]:
                ds = Native(eps)
                pred,gt,state,mask = [],[],[],[]
                offset = 0
                for length in ds.lengths:
                    for i in range(0,length-49,90):
                        frame = ds[offset+i]
                        batch = {k:v.unsqueeze(0).float()/255 if k.startswith('observation.images.') else v.unsqueeze(0)
                                 for k,v in frame.items() if k.startswith('observation.')}
                        batch['task'] = ['fold towel']
                        prediction = post(base.predict_action_chunk(pre(batch)))[0,:50,:14].float().cpu().numpy()
                        assert np.isfinite(prediction).all()
                        pred.append(prediction);gt.append(frame['action'].numpy())
                        state.append(frame['observation.state'].numpy());mask.append(~frame['action_is_pad'].numpy())
                    offset += length
                result[name] = metrics(pred,gt,state,mask)
        result['seconds'] = time.time()-started
        v = result['val']
        result['arm_improvement_vs_hold_percent'] = 100*(1-v['model']['arm_deg_mean']/v['hold']['arm_deg_mean'])
        path = Path('/home/dit/pi0_aloha_towel/finetune/eval')/f'pi05_new31_{stage}_lora_{step}.json'
        path.write_text(json.dumps(result,indent=2)+'\n')
        print('HELD_OUT_VALIDATION',json.dumps(result),flush=True)
        return result
    finally:
        policy.train(was_training)
