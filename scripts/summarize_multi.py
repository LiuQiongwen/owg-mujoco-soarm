import json,glob,re,csv,os
import pandas as pd

rows=[]
for f in glob.glob("grasp_6dof/dataset/*_validated.json"):
    bn=os.path.basename(f)
    m=re.search(r"(cube|sphere_small|cylinder)_y(\d+)_v(0\.\d+)_zm(0\.\d+)_s(\d+)_validated\.json", bn)
    if not m: 
        # 兼容 cube.urdf 原名
        m=re.search(r"([A-Za-z0-9]+)_y(\d+)_v(0\.\d+)_zm(0\.\d+)_s(\d+)_validated\.json", bn)
    if not m: 
        continue
    obj,yaw,vox,zm,seed=m.groups()
    js=json.load(open(f))
    succ=sum(1 for g in js if g.get("success"))
    rows.append(dict(obj=obj,yaw=int(yaw),voxel=float(vox),zmargin=float(zm),seed=int(seed),
                     n=len(js),success=succ,success_rate=succ/max(1,len(js)),file=bn))

os.makedirs("grasp_6dof/out",exist_ok=True)
pd.DataFrame(rows).to_csv("grasp_6dof/out/validated_multi.csv",index=False)

df=pd.DataFrame(rows)
g=df.groupby(["obj","yaw","voxel","zmargin"]).success_rate.mean().reset_index()
pivot= (g.groupby(["yaw","voxel","zmargin"])
          .success_rate.agg(["mean","std","count"]).reset_index()
          .sort_values(["mean","std"],ascending=[False,True]))
print(pivot.head(15))
pivot.to_csv("grasp_6dof/out/best_params_across_objects.csv",index=False)

# 给出“跨物体最稳”TOP1（均值高且方差小）
best=pivot.iloc[0]
print("\nBEST across objects:", dict(best))

