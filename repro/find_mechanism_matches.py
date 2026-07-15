#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系統性找「像 teniposide 那樣」的候選:
  對每個寵物病 → 定出已知有效藥的「共同機制靶點」(被 ≥2 個已知藥打的靶)
  → 看 Stack 前段候選(非已知)有誰也打中該靶 → 就是機制對得上的候選
輸出:每筆 (病, 候選, Stack排名, 共同靶點, 幾個已知藥共享, 選擇性)
這是「篩選」(KG),挑出值得 docking 確認的;teniposide 已 docking 確認(-11)。
"""
import csv, json, urllib.request, time
import numpy as np
np.seterr(all="ignore")
from rdkit import Chem

# 5 個出現的機制靶點 → 人類可讀名(UniProt 標準,硬編可核實)
TGT_NAME={"P00915":"碳酸酐酶 CA1","P04150":"糖皮質激素受體 NR3C1",
          "P12821":"血管收縮素轉化酶 ACE","P14867":"GABA-A 受體 GABRA1",
          "P11388":"拓撲異構酶 TOP2A"}
_nc={}
def pubchem_name(node, db, smi):
    if db in _nc: return _nc[db]
    nm=db
    try:
        ik=Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
        url=f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{ik}/property/Title/JSON"
        with urllib.request.urlopen(url,timeout=12) as r:
            nm=json.load(r)["PropertyTable"]["Properties"][0]["Title"]
    except Exception: nm=db
    _nc[db]=nm; time.sleep(0.2); return nm

def load(fn): return np.array([[float(x) for x in r] for r in csv.reader(open(fn))],dtype=np.float32)
Y=load("dataset/KPet/KPet_baseline.csv"); nd,ndis=Y.shape
PET=range(454,504)
TOPK=30   # 只看 Stack 前 30 的候選(要夠前段才算模型也看好)

# --- KG: 藥->靶點 ---
dp={}
for r in csv.DictReader(open("dataset/Kdataset/associations/drug_protein.csv")):
    dp.setdefault(int(r["Drug"]),[]).append(int(r["Protein"]))
prot_name={int(r["ID"]):r["Protein"] for r in csv.DictReader(open("dataset/Kdataset/omics/protein.csv"))}
node2db,smiles={},{}
for r in csv.DictReader(open("dataset/Kdataset/omics/drug.csv")):
    node2db[int(r["ID"])]=r["Drug"]; smiles[int(r["ID"])]=r["SMILES"]
pet_name={int(r["kpet_index"]):r["name"].replace("寵物-","") for r in csv.DictReader(open("dataset/KPet/KPet_pet_diseases.csv"))}

# --- Stack 分數(同 produce_candidates)---
zc=lambda M:(M-M.mean(0))/(M.std(0)+1e-9)
def norm(S): S=(S+S.T)/2; S=S.copy(); np.fill_diagonal(S,0); return S/np.maximum(S.sum(1,keepdims=True),1e-9)
Nf=norm(load("dataset/KPet/drug_sim_fused.csv"))
ddb=load("dataset/Kdataset/disease_disease_baseline.csv")
Sdis=np.zeros((ndis,ndis),dtype=np.float32); Sdis[:454,:454]=ddb
for row in csv.DictReader(open("dataset/KPet/KPet_pet_disease_disease.csv")):
    a,b=int(row["Disease1"]),int(row["Disease2"]); h,p=(a,b) if a<b else (b,a)
    if p>=454 and h<454: Sdis[p,:454]=ddb[h,:]; Sdis[:454,p]=ddb[:,h]; Sdis[p,h]=Sdis[h,p]=1
NSdis=norm(Sdis); Az=zc(load("resultKPetSup2_par_42/result.csv"))
def prop(Y0,it=20):
    F=Y0.copy()
    for _ in range(it): F=0.45*(Nf@F)+0.45*(F@NSdis.T)+0.1*Y0
    return F
def nmf(Y0,r=50,it=80):
    rs=np.random.RandomState(0); W=np.abs(rs.rand(nd,r)).astype(np.float32); H=np.abs(rs.rand(r,ndis)).astype(np.float32)
    for _ in range(it): H*=(W.T@Y0)/np.maximum(W.T@W@H,1e-6); W*=(Y0@H.T)/np.maximum(W@H@H.T,1e-6)
    return W@H
comb=Az+0.7*zc(prop(Y.copy()))+0.5*zc(nmf(Y.copy()))

matches=[]
for d in PET:
    known=set(np.where(Y[:,d]==1)[0])
    if len(known)<3: continue                    # 已知藥太少,無機制訊號
    # 共同機制靶點:被 >=2 個已知藥打的靶
    tgt_count={}
    for kd in known:
        for t in set(dp.get(kd,[])): tgt_count[t]=tgt_count.get(t,0)+1
    mech_tgts={t:c for t,c in tgt_count.items() if c>=2}
    if not mech_tgts: continue
    # 候選:Stack 前 TOPK(非已知)且打中任一機制靶
    order=[i for i in np.argsort(-comb[:,d]) if i not in known][:TOPK]
    for rank,cand in enumerate(order,1):
        hit=[t for t in dp.get(cand,[]) if t in mech_tgts]
        if hit:
            best=max(hit,key=lambda t:mech_tgts[t])
            up=prot_name.get(best,str(best))
            matches.append(dict(dis=int(d), dis_name=pet_name.get(d,str(d)),
                                cand=int(cand), db=node2db.get(cand,str(cand)), rank=int(rank),
                                tgt_uniprot=up, tgt_name=TGT_NAME.get(up,up),
                                shared=int(mech_tgts[best]), n_known=int(len(known)),
                                selectivity=int(len(dp.get(cand,[])))))

matches.sort(key=lambda m:(m["rank"], -m["shared"]))
print(f"找到 {len(matches)} 筆;查 PubChem 藥名中(不猜)...")
for m in matches:
    m["name"]=pubchem_name(m["cand"], m["db"], smiles.get(m["cand"],""))
print(f"\n{'病名':<32}{'候選藥':<22}{'排名':<5}{'共同靶':<22}{'共享'}")
print("-"*90)
for m in matches:
    print(f"{m['dis_name'][:30]:<32}{m['name'][:20]:<22}#{m['rank']:<4}{m['tgt_name'][:20]:<22}{m['shared']}/{m['n_known']}")
json.dump(matches, open("mechanism_matches.json","w"), ensure_ascii=False, indent=1)
print(f"\n→ 存 mechanism_matches.json({len(matches)} 筆)")
