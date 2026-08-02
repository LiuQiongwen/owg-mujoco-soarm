#!/usr/bin/env python3
"""Four-object neighborhood evaluation using the isolated extension critic."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import os,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent)); os.environ.setdefault('MUJOCO_GL','egl')
from tango_robot.env_soarm import EnvironmentSoArm
from data.transition_logger import compute_pc_stats
from world_model.train_object_extension_critic import Critic,feature,normalize_object
from world_model.train_robustness_head import Head
from scripts.risk_gated_vla_phase1_eval import _load_scene,_sample_grasp,execute_candidate,geo_score,EVAL_CENTRE_Y,_SPREAD_XY,OBJECTS

OBJECTS4=['cracker','mustard','drill','tomato_soup_can','medium_clamp']; OOD=['pear']; ASSET={'tomato_soup_can':'cylinder','medium_clamp':'medium_clamp','pear':'pear'}
PERT=(('nominal',(0,0,0,0)),('x_plus_2mm',(0.002,0,0,0)),('x_minus_2mm',(-0.002,0,0,0)),('yaw_plus_2deg',(0,0,0,np.deg2rad(2))),('yaw_minus_2deg',(0,0,0,-np.deg2rad(2))))
def bundles(path):
 out=[]
 for p in sorted(Path(path).glob('object_bce_4obj_seed*.pt')):
  d=__import__('torch').load(p,map_location='cpu',weights_only=False); m=Critic(d['dim']);m.load_state_dict(d['state_dict']);m.eval();out.append((m,d['mean'],d['std']))
 return out
def score(rec,cands,bs):
 import torch
 x=torch.tensor([feature(rec,c) for c in cands],dtype=torch.float32)
 with torch.no_grad(): a=[m((x-mean)/std).sigmoid().numpy() for m,mean,std in bs]
 a=np.stack(a); return a.mean(0)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--objects',default=','.join(OBJECTS4));ap.add_argument('--scenes',type=int,default=30);ap.add_argument('--base-seed',type=int,default=1001);ap.add_argument('--k-grasps',type=int,default=10);ap.add_argument('--model-dir',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--all-candidates',action='store_true');ap.add_argument('--robustness-head',default='');a=ap.parse_args()
 out=Path(a.out_dir); 
 if out.exists() and any(out.iterdir()): raise SystemExit('refusing non-empty output')
 out.mkdir(parents=True); bs=bundles(a.model_dir); robust=None
 if a.robustness_head:
  import torch; rd=torch.load(a.robustness_head,map_location='cpu',weights_only=False); robust=(Head(rd['dim']),rd['mean'],rd['std']); robust[0].load_state_dict(rd['state_dict']); robust[0].eval()
 env=EnvironmentSoArm(grasp_mode='physics_weld_after_bilateral',visual=False); rows=[]
 for oi,obj in enumerate([x.strip() for x in a.objects.split(',')]):
  if obj not in OBJECTS4+OOD: raise SystemExit(f'unknown object {obj}')
  asset=ASSET.get(obj,obj)
  for si in range(a.scenes):
   seed=(a.base_seed*10000000+oi*100000+si)%(2**32); rng=np.random.default_rng(seed); cx=float(rng.uniform(-_SPREAD_XY,_SPREAD_XY));cy=(-.20 if obj=='medium_clamp' else EVAL_CENTRE_Y+float(rng.uniform(-.04,.04))); obj_name=('YcbPear' if obj=='pear' else ('YcbMediumClamp' if obj=='medium_clamp' else OBJECTS[asset])); execute_key=('cylinder' if obj=='pear' else ('cylinder' if obj=='medium_clamp' else asset)); oid=_load_scene(env,obj_name,execute_key,cx,cy); pos=env.get_obj_pos(oid).copy(); quat=env.get_obj_pose(oid)['quaternion'].copy(); pc=compute_pc_stats(env.get_obs(pointcloud=True),oid); pool=np.stack([_sample_grasp(pos,rng) for _ in range(a.k_grasps)]); 
   if obj=='medium_clamp': pool[:,2]+=0.04
   rec={'object':obj,'obj_pos_before':pos,'obj_quat_before':quat,'pc_stats_before':pc}; scored=[]
   for pose in pool:
    neigh=[]
    for name,d in PERT:
     q=pose.copy();q[:3]+=d[:3];q[3]+=d[3];neigh.append({'candidate_pose':q.tolist()})
    s=score(rec,neigh,bs); robust_score=0.0
    if robust:
     import torch; xx=(torch.tensor([feature(rec,{'candidate_pose':pose.tolist()})],dtype=torch.float32)-robust[1])/robust[2]
     with torch.no_grad(): robust_score=float(robust[0](xx).sigmoid().item())
    scored.append({'pose':pose.tolist(),'geometry':float(geo_score(pose,pos,pc)),'point':float(s[0]),'mean':float(s.mean()),'std':float(s.std()),'robust_0p5':float(s.mean()-0.5*s.std()),'robust_head':robust_score,'worst_case':float(s.min()),'neigh':neigh})
   sel={'geometry':int(np.argmax([x['geometry'] for x in scored])),'point':int(np.argmax([x['point'] for x in scored])),'mean':int(np.argmax([x['mean'] for x in scored])),'robust_0p5':int(np.argmax([x['robust_0p5'] for x in scored])),'worst_case':int(np.argmax([x['worst_case'] for x in scored]))};
   if robust: sel['robustness_head']=int(np.argmax([x['robust_head'] for x in scored]))
   outcomes=[]
   targets=[(f'candidate_{i}',i) for i in range(len(scored))] if a.all_candidates else list(sel.items())
   for method,idx in targets:
    for name,d in PERT:
     q=np.asarray(scored[idx]['pose']);q[:3]+=d[:3];q[3]+=d[3]; r=execute_candidate(env,execute_key,{'obj_pos':pos,'obj_quat':quat,'pc_stats':pc,'candidates':pool,'seed':int(seed),'cx':cx,'cy':cy,'obj_name':obj_name},q,grasp_debug=False);outcomes.append({'method':method,'perturbation_type':name,'candidate_idx':idx,'delta':list(d),'candidate_pose':q.tolist(),**r})
   rows.append({'object':obj,'scene_idx':si,'seed':int(seed),'obj_pos_before':pos.tolist(),'obj_quat_before':quat.tolist(),'pc_stats_before':[float(v) for v in pc],'candidate_pool':pool.tolist(),'selected':sel,'outcomes':outcomes})
 (out/'scenes.jsonl').write_text('\n'.join(json.dumps(r,default=lambda x:x.tolist() if hasattr(x,'tolist') else x) for r in rows)+'\n');(out/'config.json').write_text(json.dumps(vars(a)|{'objects':OBJECTS4,'asset_alias':ASSET,'perturbations':[x[0] for x in PERT]},indent=2));(out/'trial_manifest.json').write_text(json.dumps({'n_scenes':len(rows)},indent=2)+'\n')
if __name__=='__main__': main()
