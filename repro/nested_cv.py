#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""巢狀 CV(無偏)最佳 Stack。外層報分、內層調(β,γ),outer-test 全程隔離。
硬檢查:預測 outer-test 時它必須被遮(leak_flags 必須=0)。tie-aware。"""
import numpy as np, csv, json, time
np.seterr(all='ignore')
def load(fn): return np.array([[float(x) for x in r] for r in csv.reader(open(fn))],dtype=np.float32)
Y=load('dataset/KPet/KPet_baseline.csv'); nd,ndis=Y.shape; n=0
sp=json.load(open('sup_positives.json')); labels=[(r[0],r[1]) for r in sp]; n=len(labels)
is_repo=np.array([('重定位' in r[4]) for r in sp]); deg=Y.sum(1)
md=np.median([deg[d] for d,c in labels]); nonpop=np.array([deg[d]<=md for d,c in labels])
rank_of=lambda col,d:(col>col[d]).sum()+((col==col[d]).sum()+1)/2.0
def norm(S): S=(S+S.T)/2; S=S.copy(); np.fill_diagonal(S,0); return S/np.maximum(S.sum(1,keepdims=True),1e-9)
Nf=norm(load('dataset/KPet/drug_sim_fused.csv'))
ddb=load('dataset/Kdataset/disease_disease_baseline.csv')
Sdis=np.zeros((ndis,ndis),dtype=np.float32); Sdis[:454,:454]=ddb
for row in csv.DictReader(open('dataset/KPet/KPet_pet_disease_disease.csv')):
    a,b=int(row['Disease1']),int(row['Disease2']); h,p=(a,b) if a<b else (b,a)
    if p>=454 and h<454: Sdis[p,:454]=ddb[h,:]; Sdis[:454,p]=ddb[:,h]; Sdis[p,h]=Sdis[h,p]=1
NSdis=norm(Sdis)
GNN=load('resultKPetSup2_par_42/result.csv'); zc=lambda M:(M-M.mean(0))/(M.std(0)+1e-9); Az=zc(GNN)
def maskset(S):
    Y0=Y.copy()
    for i in S: d,c=labels[i]; Y0[d,c]=0
    return Y0
def prop(Y0,ad=0.45,adis=0.45,it=20):
    F=Y0.copy()
    for _ in range(it): F=ad*(Nf@F)+adis*(F@NSdis.T)+(1-ad-adis)*Y0
    return F
def nmf(Y0,seed,r=50,it=60):
    rs=np.random.RandomState(seed); W=np.abs(rs.rand(nd,r)).astype(np.float32); H=np.abs(rs.rand(r,ndis)).astype(np.float32)
    for _ in range(it): H*=(W.T@Y0)/np.maximum(W.T@W@H,1e-6); W*=(Y0@H.T)/np.maximum(W@H@H.T,1e-6)
    return W@H
def stack(Y0,b,g,seed): return Az+b*zc(prop(Y0))+g*zc(nmf(Y0,seed))
grid=[(b,g) for b in [0.4,0.7,1.0] for g in [0.3,0.6,1.0]]
def nested(seed):
    rng=np.random.RandomState(seed); idx=rng.permutation(n); outer=np.array_split(idx,5)
    outer_ranks=np.zeros(n); leak=0; chosen=[]
    for o in range(5):
        otest=list(outer[o]); otrain=[i for f in range(5) if f!=o for i in outer[f]]
        rng2=np.random.RandomState(100+seed*10+o); it_idx=rng2.permutation(otrain); inner=np.array_split(it_idx,3)
        best=None; bsc=-1
        for (b,g) in grid:
            scs=[]
            for ii in range(3):
                itest=list(inner[ii])
                Y0=maskset(set(otest)|set(itest))          # 遮 outer-test + inner-test
                M=stack(Y0,b,g,seed=1000+seed*100+o*10+ii)
                r=[rank_of(M[:,labels[i][1]],labels[i][0]) for i in itest]
                scs.append(np.mean([x<=50 for x in r]))
            s=np.mean(scs)
            if s>bsc: bsc=s; best=(b,g)
        chosen.append(best)
        Y0=maskset(set(otest))                              # 只用 best 權重,遮 outer-test
        for i in otest:                                     # 硬檢查:outer-test 必須被遮
            d,c=labels[i]
            if Y0[d,c]!=0: leak+=1
        M=stack(Y0,best[0],best[1],seed=2000+seed*10+o)
        for i in otest: d,c=labels[i]; outer_ranks[i]=rank_of(M[:,c],d)
    r50=lambda m:100*np.mean(outer_ranks[m]<=50)
    return leak, chosen, (r50(np.ones(n,bool)), r50(~is_repo), r50(is_repo), r50(nonpop))
t0=time.time(); res=[]; totleak=0
for s in [0,1,2]:
    leak,chosen,m=nested(s); totleak+=leak; res.append(m)
    print(f"seed{s}: leak={leak} 選中權重(β,γ)={chosen} → 全{m[0]:.0f} 標{m[1]:.0f} 重定位{m[2]:.0f} 非人氣{m[3]:.0f}")
A=np.array(res)
print(f"\n=== 巢狀CV 無偏估計(3 seed mean±std)===  [總 leak={totleak}(必須0)]")
print(f"  全部 {A[:,0].mean():.0f}±{A[:,0].std():.0f}  重定位 {A[:,2].mean():.0f}±{A[:,2].std():.0f}  非人氣 {A[:,3].mean():.0f}±{A[:,3].std():.0f}")
print(f"  對比(單層挑最佳,偏樂觀): 77/22/63")
print(f"[{time.time()-t0:.0f}s]")
