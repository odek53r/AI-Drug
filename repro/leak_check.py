#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""洩漏稽核:負對照。若把相似度換成 隨機/單位/全零,label-prop 的 recall 應掉到隨機(~5.6%)。
若仍高 → 訊號來自洩漏(測試標籤沒真的被藏)。同時檢查 mask 是否生效。"""
import numpy as np, csv, json
def load(fn): return np.array([[float(x) for x in r] for r in csv.reader(open(fn))],dtype=np.float32)
Y=load('dataset/KPet/KPet_baseline.csv'); nd,ndis=Y.shape
sp=json.load(open('sup_positives.json')); labels=[(r[0],r[1]) for r in sp]
rng=np.random.RandomState(42); idx=rng.permutation(len(labels)); folds=np.array_split(idx,5)
def norm(S):
    S=(S+S.T)/2; np.fill_diagonal(S,0); return S/np.maximum(S.sum(1,keepdims=True),1e-9)
def lp_recall(S, check_mask=False):
    rk=[]; leaked=0
    for f in range(5):
        Y0=Y.copy()
        for i in folds[f]: d0,c0=labels[i]; Y0[d0,c0]=0
        if check_mask:
            for i in folds[f]:
                d0,c0=labels[i]
                if Y0[d0,c0]!=0: leaked+=1
        F=Y0.copy()
        for _ in range(20): F=0.9*(S@F)+0.1*Y0
        for i in folds[f]:
            d0,c0=labels[i]; col=F[:,c0]; rk.append(int((col>col[d0]).sum())+1)
    rk=np.array(rk)
    return 100*(rk<=50).mean(), int(np.median(rk)), (leaked if check_mask else None)
print("隨機基準(理論): recall@50 ≈ 5.6%")
mech=load('dataset/KPet/drug_mech_sim.csv')
r,m,leak=lp_recall(norm(mech), check_mask=True)
print(f"① 真·機轉相似:   recall@50={r:.0f}% median={m}  | mask 洩漏數={leak}(應為0)")
# 負對照:打散列(破壞相似結構,保留度分布)
Sr=mech.copy(); rng.shuffle(Sr)               # shuffle rows
r,m,_=lp_recall(norm(Sr)); print(f"② 隨機打散相似:   recall@50={r:.0f}% median={m}  (應接近隨機)")
# 負對照:單位矩陣(無相似傳播)
I=np.eye(nd,dtype=np.float32);
r,m,_=lp_recall(norm(I+1e-6)); print(f"③ 單位/無相似:    recall@50={r:.0f}% median={m}  (應接近隨機)")
# 負對照:全1(所有藥同等,等於用該病平均)
r,m,_=lp_recall(norm(np.ones((nd,nd),dtype=np.float32))); print(f"④ 全1(病內平均):  recall@50={r:.0f}% median={m}")
