#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""統一用 tie-aware 中位排名,對 null 相似度做負對照。
若 null(單位/全1/打散)也高 → 訊號來自結構/度數 artifact 或洩漏;
若 null 掉到接近隨機(~5.6%)→ leak-free 的 recall 是真訊號。"""
import numpy as np, csv, json
np.seterr(all="ignore")
def load(fn): return np.array([[float(x) for x in r] for r in csv.reader(open(fn))],dtype=np.float32)
Y=load("dataset/KPet/KPet_baseline.csv"); nd,ndis=Y.shape
sp=json.load(open("sup_positives.json")); labels=[(r[0],r[1]) for r in sp]; n=len(labels)
zc=lambda M:(M-M.mean(0))/(M.std(0)+1e-9)
def norm(S): S=(S+S.T)/2; S=S.copy(); np.fill_diagonal(S,0); return S/np.maximum(S.sum(1,keepdims=True),1e-9)
# 兩種排名指標並列比對
best_rank=lambda col,d:(col>col[d]).sum()+1                       # leak_check 用的(樂觀)
mid_rank =lambda col,d:(col>col[d]).sum()+((col==col[d]).sum()+1)/2.0   # nested_cv 用的(公正)

Nf_real=load("dataset/KPet/drug_sim_fused.csv")
ddb=load("dataset/Kdataset/disease_disease_baseline.csv")
Sdis=np.zeros((ndis,ndis),dtype=np.float32); Sdis[:454,:454]=ddb
for row in csv.DictReader(open("dataset/KPet/KPet_pet_disease_disease.csv")):
    a,b=int(row["Disease1"]),int(row["Disease2"]); h,p=(a,b) if a<b else (b,a)
    if p>=454 and h<454: Sdis[p,:454]=ddb[h,:]; Sdis[:454,p]=ddb[:,h]; Sdis[p,h]=Sdis[h,p]=1

rng=np.random.RandomState(42); idx=rng.permutation(n); folds=np.array_split(idx,5)
def run(Nf, NSdis, ranker):
    Nf=norm(Nf); NSdis=norm(NSdis); hit=0
    for f in range(5):
        test=folds[f]; Y0=Y.copy()
        for i in test: d,c=labels[i]; Y0[d,c]=0
        F=Y0.copy()
        for _ in range(20): F=0.45*(Nf@F)+0.45*(F@NSdis.T)+0.1*Y0
        for i in test:
            d,c=labels[i]
            if ranker(F[:,c],d)<=50: hit+=1
    return 100*hit/n

I_d=np.eye(nd,dtype=np.float32); I_s=np.eye(ndis,dtype=np.float32)
O_d=np.ones((nd,nd),dtype=np.float32); O_s=np.ones((ndis,ndis),dtype=np.float32)
Nf_sh=Nf_real.copy(); rng.shuffle(Nf_sh); Sd_sh=Sdis.copy(); rng.shuffle(Sd_sh)

print("="*74)
print("兩側傳播 recall@50 · 真相似 vs null · 兩種排名指標")
print("="*74)
print(f"{'條件':<26}{'樂觀排名(leak_check式)':<24}{'中位排名(nested_cv式)'}")
print("-"*74)
rows=[("真相似(drug+disease)",Nf_real,Sdis),
      ("單位矩陣(無相似傳播)",I_d,I_s),
      ("全1(病內平均)",O_d,O_s),
      ("打散(破壞結構,保度數)",Nf_sh,Sd_sh)]
for name,a,b in rows:
    rb=run(a,b,best_rank); rm=run(a,b,mid_rank)
    print(f"{name:<24}{rb:>10.1f}%{'':<12}{rm:>10.1f}%")
print("-"*74)
print("隨機基準理論 ≈ 5.6%")
print("判讀:若『中位排名』欄的 null(單位/全1/打散)都掉到接近隨機,")
print("     而真相似明顯高 → leak-free 的 recall 是真訊號、非洩漏或指標假象。")
