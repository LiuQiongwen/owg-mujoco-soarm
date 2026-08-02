#!/usr/bin/env python3
"""Train an isolated local-perturbation survival head."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np,torch
import torch.nn as nn
from world_model.train_object_extension_critic import feature

class Head(nn.Module):
 def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
 def forward(self,x): return self.net(x).squeeze(-1)
def load(paths):
 out=[]
 for path in paths:
  for line in Path(path).read_text().splitlines():
   r=json.loads(line); by={}
   for o in r['outcomes']: by.setdefault(o['candidate_idx'],[]).append(bool(o['success']))
   for idx,pose in enumerate(r['candidate_pool']):
    y=float(sum(by.get(idx,[]))>=4); out.append((feature(r,{'candidate_pose':pose}),y,r['object']))
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=200);ap.add_argument('--seeds',type=int,default=5);a=ap.parse_args(); rows=load(a.data); x=torch.tensor([r[0] for r in rows],dtype=torch.float32);y=torch.tensor([r[1] for r in rows],dtype=torch.float32);mean=x.mean(0);std=x.std(0).clamp_min(1e-6);x=(x-mean)/std;out=Path(a.out);out.mkdir(parents=True,exist_ok=True);summary=[]
 for seed in range(a.seeds):
  torch.manual_seed(seed);m=Head(x.shape[1]);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4);pos=max(float(y.sum()),1);neg=max(float(len(y)-y.sum()),1);w=torch.where(y>0,torch.tensor(len(y)/(2*pos)),torch.tensor(len(y)/(2*neg)))
  for _ in range(a.epochs): opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(m(x),y,weight=w);loss.backward();opt.step()
  with torch.no_grad(): pred=m(x).sigmoid(); acc=float(((pred>=.5)==(y>=.5)).float().mean())
  p=out/f'robustness_head_seed{seed}.pt';torch.save({'state_dict':m.state_dict(),'dim':len(mean),'mean':mean,'std':std,'label':'successes>=4/5'},p);summary.append({'seed':seed,'train_accuracy':acc,'checkpoint':str(p)})
 (out/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__': main()
