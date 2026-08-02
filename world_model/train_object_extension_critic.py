#!/usr/bin/env python3
"""Train an isolated object-relative critic for the four-object extension.

The legacy three-object trainer/checkpoints are intentionally untouched.
Input files may use the historical ``cylinder`` key; it is normalized to
``tomato_soup_can`` in this extension only.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBJECTS = ["cracker", "mustard", "drill", "tomato_soup_can"]

class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,64),nn.SiLU(),nn.Dropout(0.05),
                                 nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self, x): return self.net(x).squeeze(-1)

def normalize_object(name):
    return "tomato_soup_can" if name == "cylinder" else name

def feature(rec, cand):
    pose=np.asarray(cand["candidate_pose"],dtype=np.float32)
    obj=np.asarray(rec["obj_pos_before"],dtype=np.float32)
    xyz=pose[:3]-obj; yaw=float(pose[3])
    base=[*xyz.tolist(),math.sin(yaw),math.cos(yaw),float(pose[4]),float(pose[5])]
    onehot=[float(normalize_object(rec["object"])==x) for x in OBJECTS]
    return base+[float(v) for v in rec["pc_stats_before"]]+onehot

def load(path):
    out=[]
    for line in Path(path).read_text().splitlines():
        r=json.loads(line); obj=normalize_object(r["object"])
        if obj not in OBJECTS: raise ValueError(f"unknown object: {r['object']}")
        out.append({"key":(obj,r["seed"]),"object":obj,
                    "x":[feature(r,c) for c in r["oracle_per_candidate"]],
                    "y":[float(c["success"]) for c in r["oracle_per_candidate"]]})
    return out

def train(scenes, seed, epochs):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    tr=[]; va=[]
    for obj in OBJECTS:
        g=[s for s in scenes if s["object"]==obj]
        order=rng.permutation(len(g)); nv=max(1,round(.2*len(g)))
        va += [g[i] for i in order[:nv]]; tr += [g[i] for i in order[nv:]]
    x=torch.tensor([v for s in tr for v in s["x"]],dtype=torch.float32)
    mean=x.mean(0); std=x.std(0).clamp_min(1e-6); model=Critic(x.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    best=-1.; state=None
    for ep in range(epochs):
        model.train(); opt.zero_grad(); losses=[]
        for s in tr:
            xx=(torch.tensor(s["x"],dtype=torch.float32)-mean)/std
            yy=torch.tensor(s["y"],dtype=torch.float32)
            losses.append(F.binary_cross_entropy_with_logits(model(xx),yy))
        loss=torch.stack(losses).mean(); loss.backward(); opt.step()
        if ep%10==0 or ep==epochs-1:
            model.eval(); ok=0
            with torch.no_grad():
                for s in va:
                    xx=(torch.tensor(s["x"],dtype=torch.float32)-mean)/std
                    yy=np.asarray(s["y"],dtype=int); ok+=int(yy[int(torch.argmax(model(xx)))])
            score=ok/max(len(va),1)
            if score>best: best=score; state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    return model,mean,std,tr,va,best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",nargs="+",required=True)
    ap.add_argument("--out-dir",required=True); ap.add_argument("--epochs",type=int,default=200)
    ap.add_argument("--seeds",type=int,default=5); a=ap.parse_args()
    scenes=sum((load(p) for p in a.data),[]); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for seed in range(a.seeds):
        model,mean,std,tr,va,best=train(scenes,seed,a.epochs)
        path=out/f"object_bce_4obj_seed{seed}.pt"
        torch.save({"variant":"object_bce_4obj","seed":seed,"state_dict":model.state_dict(),
                    "dim":len(mean),"mean":mean,"std":std,"objects":OBJECTS,
                    "train_keys":[s["key"] for s in tr],"val_keys":[s["key"] for s in va]},path)
        summary.append({"seed":seed,"dev_top1":best,"checkpoint":str(path)})
    (out/"validation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")

if __name__ == "__main__": main()
