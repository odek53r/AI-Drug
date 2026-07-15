#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""① 鎖最終權重(全121標籤一次CV選)② 全資料訓練 ③ 產候選清單。"""
import numpy as np, csv, json
np.seterr(all='ignore')
def load(fn): return np.array([[float(x) for x in r] for r in csv.reader(open(fn))],dtype=np.float32)
Y=load('dataset/KPet/KPet_baseline.csv'); nd,ndis=Y.shape
sp=json.load(open('sup_positives.json')); labels=[(r[0],r[1]) for r in sp]; n=len(labels)
deg=Y.sum(1); md=np.median([deg[d] for d,c in labels]); nonpop=np.array([deg[d]<=md for d,c in labels])
rank_of=lambda col,d:(col>col[d]).sum()+((col==col[d]).sum()+1)/2.0
def norm(S): S=(S+S.T)/2; S=S.copy(); np.fill_diagonal(S,0); return S/np.maximum(S.sum(1,keepdims=True),1e-9)
Sfused=load('dataset/KPet/drug_sim_fused.csv'); Nf=norm(Sfused)
ddb=load('dataset/Kdataset/disease_disease_baseline.csv')
Sdis=np.zeros((ndis,ndis),dtype=np.float32); Sdis[:454,:454]=ddb
for row in csv.DictReader(open('dataset/KPet/KPet_pet_disease_disease.csv')):
    a,b=int(row['Disease1']),int(row['Disease2']); h,p=(a,b) if a<b else (b,a)
    if p>=454 and h<454: Sdis[p,:454]=ddb[h,:]; Sdis[:454,p]=ddb[:,h]; Sdis[p,h]=Sdis[h,p]=1
NSdis=norm(Sdis)
GNN=load('resultKPetSup2_par_42/result.csv'); zc=lambda M:(M-M.mean(0))/(M.std(0)+1e-9); Az=zc(GNN)
def prop(Y0,it=20):
    F=Y0.copy()
    for _ in range(it): F=0.45*(Nf@F)+0.45*(F@NSdis.T)+0.1*Y0
    return F
def nmf(Y0,seed=0,r=50,it=80):
    rs=np.random.RandomState(seed); W=np.abs(rs.rand(nd,r)).astype(np.float32); H=np.abs(rs.rand(r,ndis)).astype(np.float32)
    for _ in range(it): H*=(W.T@Y0)/np.maximum(W.T@W@H,1e-6); W*=(Y0@H.T)/np.maximum(W@H@H.T,1e-6)
    return W@H
def stack(Y0,b,g,seed=0): return Az+b*zc(prop(Y0))+g*zc(nmf(Y0,seed))
# ① 鎖權重:全121一次 5-fold CV,挑非人氣 recall 最佳
rng=np.random.RandomState(0); idx=rng.permutation(n); folds=np.array_split(idx,5)
best=None; bsc=-1
for b in [0.4,0.7,1.0]:
    for g in [0.3,0.5,0.7,1.0]:
        rk=np.zeros(n)
        for f in range(5):
            Y0=Y.copy()
            for i in folds[f]: d,c=labels[i]; Y0[d,c]=0
            M=stack(Y0,b,g)
            for i in folds[f]: d,c=labels[i]; rk[i]=rank_of(M[:,c],d)
        sc=100*np.mean(rk[nonpop]<=50)
        if sc>bsc: bsc=sc; best=(b,g)
print(f"① 鎖定最終權重 β={best[0]} γ={best[1]}(非人氣 recall@50={bsc:.0f}%)")
# ② 全資料訓練(不遮任何標籤)
comb=stack(Y.copy(),best[0],best[1])
print("② 全資料訓練完成")
# ③ 產候選:名稱解析 + 每寵物病 top 候選(排除已知)
node2db={int(r['ID']):r['Drug'] for r in csv.DictReader(open('dataset/Kdataset/omics/drug.csv'))}
node2name={r[0]:r[2] for r in sp}
DBN={'DB00619':'imatinib','DB01254':'dasatinib','DB00398':'sorafenib','DB01268':'sunitinib','DB08896':'regorafenib','DB01229':'paclitaxel','DB00444':'teniposide','DB00773':'etoposide','DB00631':'clofarabine','DB00262':'carmustine','DB11581':'venetoclax','DB06176':'romidepsin','DB00997':'doxorubicin','DB00541':'vincristine','DB00570':'vinblastine','DB01590':'everolimus','DB06287':'temsirolimus','DB00877':'sirolimus'}
nm=lambda d: node2name.get(d) or DBN.get(node2db.get(d,''),node2db.get(d,str(d)))
petname={int(r['kpet_index']):r['name'].replace('寵物-','') for r in csv.DictReader(open('dataset/KPet/KPet_pet_diseases.csv'))}
known={}
for r in sp: known.setdefault(r[1],set()).add(r[0])
rows=[['disease','rank','drug','drugbank','score','類型']]
pet_cols=sorted(petname.keys())
for c in pet_cols:
    kn=known.get(c,set())
    order=np.argsort(-comb[:,c])
    top=[d for d in order if d not in kn][:8]
    for r,d in enumerate(top,1):
        fam='家族' if (len(kn) and max((Sfused[d][k] for k in kn),default=0)>=0.3) else '非家族'
        rows.append([petname.get(c,c),r,nm(d),node2db.get(d,''),f"{comb[d][c]:.2f}",fam])
with open('stack_candidates.csv','w',newline='') as f: csv.writer(f).writerows(rows)
print(f"③ 產出 stack_candidates.csv:{len(pet_cols)} 病 × top8 = {len(rows)-1} 候選")
# 示範幾個癌症
print("\n=== 示範(3 個癌)===")
for tgt in ['Osteosarcoma','Urinary Bladder Neoplasms','Precursor Cell Lymphoblastic Leukemia-Lymphoma']:
    c=[k for k,v in petname.items() if v==tgt]
    if not c: continue
    c=c[0]; kn=known.get(c,set()); order=[d for d in np.argsort(-comb[:,c]) if d not in kn][:5]
    print(f"\n{tgt}(已知藥 {len(kn)} 個):")
    for r,d in enumerate(order,1):
        fam='家族' if (len(kn) and max((Sfused[d][k] for k in kn),default=0)>=0.3) else '非家族'
        print(f"  {r}. {nm(d):<16} [{fam}] score={comb[d][c]:.2f}")
