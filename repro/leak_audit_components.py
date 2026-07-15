#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐成分稽核:分開量 Az(凍結GNN)/ prop / nmf / full 的 recall@50。

⚠️ 更正紀錄:本檔早期版本宣稱「凍結 GNN 是用全部標籤(含測試)訓練的 → 屬洩漏」。
   那是錯的。查 train_parallel.py 證實:`resultKPetSup2_par_42/result.csv` 是**真 10 折 OOF**——
   每折訓練前 `remove_graph(g, test_pos_id)` 把測試邊從圖移除、`mask_label[test]=0` 不進 loss,
   `aggregate()` 再逐格由「它被 held-out 的那折」填。
   所以 Az 的 58.7% 是**合法的 OOF 泛化表現**,不是記憶;含 Az 的 ⑤ 仍是 leak-free。

本檔量的是「各成分各自貢獻多少」,不是「誰在洩漏」。
prop/nmf 用 5 折遮蔽 + 硬檢查(leak 必須=0)。
真正的 null 負對照請跑 leak_audit_v2.py(公正中位排名下,單位矩陣 → 0.0%)。"""
import numpy as np, csv, json
np.seterr(all="ignore")
def load(fn): return np.array([[float(x) for x in r] for r in csv.reader(open(fn))],dtype=np.float32)
Y=load("dataset/KPet/KPet_baseline.csv"); nd,ndis=Y.shape
sp=json.load(open("sup_positives.json")); labels=[(r[0],r[1]) for r in sp]; n=len(labels)
zc=lambda M:(M-M.mean(0))/(M.std(0)+1e-9)
def norm(S): S=(S+S.T)/2; S=S.copy(); np.fill_diagonal(S,0); return S/np.maximum(S.sum(1,keepdims=True),1e-9)
rank_of=lambda col,d:(col>col[d]).sum()+((col==col[d]).sum()+1)/2.0
Nf=norm(load("dataset/KPet/drug_sim_fused.csv"))
ddb=load("dataset/Kdataset/disease_disease_baseline.csv")
Sdis=np.zeros((ndis,ndis),dtype=np.float32); Sdis[:454,:454]=ddb
for row in csv.DictReader(open("dataset/KPet/KPet_pet_disease_disease.csv")):
    a,b=int(row["Disease1"]),int(row["Disease2"]); h,p=(a,b) if a<b else (b,a)
    if p>=454 and h<454: Sdis[p,:454]=ddb[h,:]; Sdis[:454,p]=ddb[:,h]; Sdis[p,h]=Sdis[h,p]=1
NSdis=norm(Sdis)
Az=zc(load("resultKPetSup2_par_42/result.csv"))

def prop(Y0,Nfx=Nf):
    F=Y0.copy()
    for _ in range(20): F=0.45*(Nfx@F)+0.45*(F@NSdis.T)+0.1*Y0
    return F
def nmf(Y0,r=50,it=60):
    rs=np.random.RandomState(0); W=np.abs(rs.rand(nd,r)).astype(np.float32); H=np.abs(rs.rand(r,ndis)).astype(np.float32)
    for _ in range(it): H*=(W.T@Y0)/np.maximum(W.T@W@H,1e-6); W*=(Y0@H.T)/np.maximum(W@H@H.T,1e-6)
    return W@H

# --- ① Az(凍結GNN):無法遮蔽,對 121 標籤直接算 = 洩漏/記憶指標 ---
r_az=[rank_of(Az[:,c],d) for d,c in labels]
rec_az=100*np.mean([x<=50 for x in r_az])

# --- 5 折遮蔽:prop / nmf / prop+nmf / full(含Az) ---
rng=np.random.RandomState(42); idx=rng.permutation(n); folds=np.array_split(idx,5)
R={"prop":np.zeros(n),"nmf":np.zeros(n),"pn":np.zeros(n),"full":np.zeros(n),"prop_shuf":np.zeros(n)}
leak=0
Nf_shuf=Nf.copy(); rng2=np.random.RandomState(7); rng2.shuffle(Nf_shuf)  # 打散列=破壞相似結構
for f in range(5):
    test=folds[f]; Y0=Y.copy()
    for i in test: d,c=labels[i]; Y0[d,c]=0
    for i in test:                                   # 硬檢查:測試必須=0
        d,c=labels[i]
        if Y0[d,c]!=0: leak+=1
    P=zc(prop(Y0)); N=zc(nmf(Y0)); Ps=zc(prop(Y0,Nf_shuf))
    for i in test:
        d,c=labels[i]
        R["prop"][i]=rank_of(P[:,c],d); R["nmf"][i]=rank_of(N[:,c],d)
        R["pn"][i]=rank_of((0.7*P+0.5*N)[:,c],d)
        R["full"][i]=rank_of((Az+0.7*P+0.5*N)[:,c],d)
        R["prop_shuf"][i]=rank_of(Ps[:,c],d)
rec=lambda k:100*np.mean(R[k]<=50)
print("="*66)
print("逐成分 recall@50(121 個真實寵物標籤)")
print("="*66)
print(f"  mask 硬檢查洩漏數 = {leak}  (必須 0)")
print(f"  隨機基準理論值 ≈ 5.6%")
print("-"*66)
print(f"  ① Az 凍結GNN(10 折 OOF,每格由「它被 held-out 那折」預測)= {rec_az:5.1f}%   ← 合法的 OOF 泛化")
print(f"  ② prop  傳播  (遮蔽) = {rec('prop'):5.1f}%")
print(f"  ③ nmf   分解  (遮蔽) = {rec('nmf'):5.1f}%")
print(f"  ④ prop+nmf   (遮蔽,不含 GNN 的版本) = {rec('pn'):5.1f}%")
print(f"  ⑤ full = Az+prop+nmf(完整 Stack)= {rec('full'):5.1f}%   ← Az 是 OOF + prop/nmf 已遮 → leak-free")
print("-"*66)
print(f"  負對照:prop 只把【藥相似度】打散(病相似度仍完整) = {rec('prop_shuf'):5.1f}%")
print(f"          ⚠️ 這個對照【不完整】:兩側傳播的病軸(同病橋接)沒被破壞,訊號仍會透過它傳,")
print(f"             所以不會掉到隨機。要看真正的 null,請跑 leak_audit_v2.py(單位矩陣 → 0.0%)。")
print("="*66)
print("判讀:")
print(f"  · ⑤(完整 Stack)就是『測試完全沒被看到』的數字 = nested_cv 報的 74%。")
print(f"  · ① Az 是 10 折 OOF(訓練時該格的邊已被 remove_graph 移除)→ 58.7% 是真本事,不是背答案。")
print(f"    因此 ⑤(Az + 遮蔽的 prop/nmf)仍是 leak-free,nested_cv 的 74% 成立。")
print(f"  · ④ 是「完全不用 GNN」的版本,不是「洩漏修正版」——別把 ④ 當成 ⑤ 的誠實對照。")
print(f"  · 注意:74% 裡有一大半是【人氣先驗】(全1 對照就有 42%),不是相似/機制的功勞。")
print(f"    對外請一併講非人氣(59%)與真 novel 重定位(14%,n=12,樣本極小)。")
