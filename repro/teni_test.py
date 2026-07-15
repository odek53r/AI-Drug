#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把四關流水線跑在 teniposide → 犬淋巴瘤 上,重點看機制關(打不打 TOP2A)。
TOP2A = 淋巴瘤已知有效藥(etoposide/doxorubicin/mitoxantrone)的共同靶點。
與肥大細胞瘤對比:那邊 top 候選沒一個打 KIT;這邊看 teniposide 打不打 TOP2A。
誠實:消融已證模型排 teniposide 高是「化學像 etoposide」(表親型),
      TOP2A 是合理的事後機制佐證(KG 事實 + 生物),不是模型的排名理由。
"""
import csv, os, sys
import numpy as np
np.seterr(all="ignore")
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, FilterCatalog, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

def load(fn): return np.array([[float(x) for x in r] for r in csv.reader(open(fn))],dtype=np.float32)
Y=load("dataset/KPet/KPet_baseline.csv"); nd,ndis=Y.shape
DIS=462; TENI=110; TOP2A=11237
known=set(np.where(Y[:,DIS]==1)[0])

pet={int(r["kpet_index"]):r["name"] for r in csv.DictReader(open("dataset/KPet/KPet_pet_diseases.csv"))}
prot={r["Protein"]:int(r["ID"]) for r in csv.DictReader(open("dataset/Kdataset/omics/protein.csv"))}
smiles={int(r["ID"]):r["SMILES"] for r in csv.DictReader(open("dataset/Kdataset/omics/drug.csv"))}
node2db={int(r["ID"]):r["Drug"] for r in csv.DictReader(open("dataset/Kdataset/omics/drug.csv"))}
dp={}
for r in csv.DictReader(open("dataset/Kdataset/associations/drug_protein.csv")):
    dp.setdefault(int(r["Drug"]),[]).append(int(r["Protein"]))
DBN={"DB00444":"teniposide","DB00773":"etoposide","DB00997":"doxorubicin","DB01204":"mitoxantrone"}
nm=lambda i: DBN.get(node2db.get(i,""), node2db.get(i,str(i)))

print("="*70); print("0. ID 核對"); print("="*70)
print(f"  疾病 462 = {pet.get(DIS)}")
print(f"  TOP2A = protein {TOP2A}(UniProt P11388→{prot.get('P11388')})")
print(f"  teniposide = node {TENI}(DB={node2db.get(TENI)})")

print("\n"+"="*70); print("① 機制關:TOP2A 是不是淋巴瘤已知有效藥的共同靶點?"); print("="*70)
# 462 的已知藥裡,誰打 TOP2A
known_hit_top2a=[d for d in known if TOP2A in dp.get(d,[])]
print(f"  462 已知藥共 {len(known)} 個;其中打 TOP2A 的:{[nm(d) for d in known_hit_top2a]}")
teni_hits_top2a = TOP2A in dp.get(TENI,[])
teni_targets = dp.get(TENI,[])
print(f"  teniposide 靶點 = {teni_targets} → 打 TOP2A? {'✅ 是' if teni_hits_top2a else '❌ 否'}(選擇性:只打 {len(teni_targets)} 個靶=高專一)")
g1 = teni_hits_top2a and len(known_hit_top2a)>0
print(f"  → 機制關判定:{'✅ 過(打中已知有效藥的共同靶點)' if g1 else '❌ 未過'}")

print("\n"+"="*70); print("②③ 藥性關(RDKit)"); print("="*70)
mol=Chem.MolFromSmiles(smiles[TENI])
mw=Descriptors.MolWt(mol); logp=Crippen.MolLogP(mol)
hbd=Descriptors.NumHDonors(mol); hba=Descriptors.NumHAcceptors(mol)
ro5=sum([mw>500,logp>5,hbd>5,hba>10]); qed=QED.qed(mol)
sa=sascorer.calculateScore(mol)
params=FilterCatalog.FilterCatalogParams(); params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
npains=len(FilterCatalog.FilterCatalog(params).GetMatches(mol))
g2 = sa<4.5; g3 = (npains==0) and (ro5<=1)
print(f"  ② SA score = {sa:.2f}  → {'✅ 好合成' if g2 else '⚠️ 偏難(大分子天然物衍生)'}")
print(f"  ③ PAINS 警訊 = {npains}、Ro5 違反 = {ro5}(MW {mw:.0f}, LogP {logp:.2f}, HBD {hbd}, HBA {hba})、QED = {qed:.2f}")
print(f"     → {'✅ 無明顯警訊' if g3 else '⚠️ 注意(大分子,Ro5 違反)'}")

print("\n"+"="*70); print("Stack 模型排名(內插分數,供對照)"); print("="*70)
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
cand=sorted([i for i in range(nd) if i not in known],key=lambda i:-comb[i,DIS])
rank=cand.index(TENI)+1
print(f"  teniposide → 犬淋巴瘤 Stack 排名 = #{rank}(共 {len(cand)} 候選)")

print("\n"+"="*70); print("Go/No-Go 總判 · teniposide"); print("="*70)
print(f"  ① 機制(打 TOP2A)      : {'✅ 過' if g1 else '❌'}")
print(f"  ② 可合成(SA)          : {'✅ 過' if g2 else '⚠️ 注意'}")
print(f"  ③ 安全(PAINS/Ro5)     : {'✅ 過' if g3 else '⚠️ 注意'}")
print(f"  ④ IP                    : N/A(需律師)")
verdict = "✅ 進場・且打中機制靶點" if g1 else "⚠️"
if not (g2 and g3): verdict += "(藥性有注意項:大分子)"
print(f"  → {verdict}")
print("\n  對比肥大細胞瘤:那邊 top15 沒一個打 KIT;這邊 teniposide ✅ 打中 TOP2A。")
print("  誠實:模型排 teniposide 高的『理由』是化學像 etoposide(消融證,表親型);")
print("        TOP2A 是站得住的事後機制佐證(它+已知有效藥都打同一靶),但非模型的排名依據。")
