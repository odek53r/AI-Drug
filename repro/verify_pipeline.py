#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
獨立驗證 hit_pipeline 這條流水線「本身是正確的」。逐條 PASS/FAIL:
 A. 疾病/靶點 ID 對不對(503=肥大細胞瘤、17695=KIT/P10721)
 B. KIT 靶點在 KG 裡「生物上」對不對(打它的藥該是已知 KIT 抑制劑)
 C. RDKit 描述子對不對(用已知文獻分子量交叉驗,抓 SMILES 對錯)
 D. 「top15 沒一個打 KIT」重算屬實
 E. 9 個 KIT 藥的排名重算(#39–582、imatinib #109)
 F. RDKit 決定性:PAINS/SA 重跑數字一致

註:D/E 的排名是用 10 折 OOF 的 result.csv 算的(確定性,可重現)。
    若你自己重訓 GNN,排名會不同 —— 那是 m2v 非確定性,不是錯誤。
"""
import csv, os, sys
import numpy as np
np.seterr(all="ignore")
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, FilterCatalog, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

PASS=[]; FAIL=[]
def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

def load(fn): return np.array([[float(x) for x in r] for r in csv.reader(open(fn))],dtype=np.float32)
Y=load("dataset/KPet/KPet_baseline.csv"); nd,ndis=Y.shape
DIS=503; KIT=17695
known=set(np.where(Y[:,DIS]==1)[0])

print("="*70); print("A. 疾病 / 靶點 ID"); print("="*70)
pet={int(r["kpet_index"]):r["name"] for r in csv.DictReader(open("dataset/KPet/KPet_pet_diseases.csv"))}
check("503 = 肥大細胞瘤(Mast-Cell)", "Mast" in pet.get(DIS,""), pet.get(DIS,"?"))
prot={r["Protein"]:int(r["ID"]) for r in csv.DictReader(open("dataset/Kdataset/omics/protein.csv"))}
check("17695 = KIT(UniProt P10721)", prot.get("P10721")==KIT, f"P10721→{prot.get('P10721')}")

print("="*70); print("B. KIT 靶點在 KG 裡生物上正確?(打它的藥該是 KIT 抑制劑)"); print("="*70)
dp={}
for r in csv.DictReader(open("dataset/Kdataset/associations/drug_protein.csv")):
    dp.setdefault(int(r["Drug"]),[]).append(int(r["Protein"]))
node2db={int(r["ID"]):r["Drug"] for r in csv.DictReader(open("dataset/Kdataset/omics/drug.csv"))}
KIT_INHIB_DB={"DB00619":"imatinib","DB01254":"dasatinib","DB01268":"sunitinib","DB08896":"regorafenib",
              "DB04868":"nilotinib","DB08901":"ponatinib","DB00398":"sorafenib"}
kit_drugs=[d for d in range(nd) if KIT in dp.get(d,[])]
kit_db=set(node2db.get(d,"") for d in kit_drugs)
hit_known=[nm for db,nm in KIT_INHIB_DB.items() if db in kit_db]
# 至少多數已知 KIT 抑制劑要出現在「打 17695」的清單裡
check("打 17695 的藥=已知 KIT 抑制劑", len(hit_known)>=5,
      f"{len(kit_drugs)} 個藥打 17695,其中已知 KIT-i:{hit_known}")

print("="*70); print("C. RDKit 描述子 vs 已知文獻分子量(抓 SMILES 對錯)"); print("="*70)
smiles={int(r["ID"]):r["SMILES"] for r in csv.DictReader(open("dataset/Kdataset/omics/drug.csv"))}
db2node={v:k for k,v in node2db.items()}
REF_MW={"DB00563":454.4,"DB00997":543.5,"DB00570":810.9,"DB00091":1202.6,"DB00987":243.2}  # 文獻值
# 注意:DB00570=vinblastine;vincristine=DB00541。逐一核。
REF_MW={"DB00563":(454.4,"methotrexate"),"DB00997":(543.5,"doxorubicin"),
        "DB00541":(824.9,"vincristine"),"DB00091":(1202.6,"cyclosporine"),
        "DB00987":(243.2,"cytarabine")}
for db,(mw_ref,nm) in REF_MW.items():
    node=db2node.get(db)
    mol=Chem.MolFromSmiles(smiles.get(node,"")) if node is not None else None
    mw=Descriptors.MolWt(mol) if mol else -1
    check(f"{nm}({db}) MW≈{mw_ref}", mol is not None and abs(mw-mw_ref)<1.5, f"算得 {mw:.1f}")

print("="*70); print("D+E. 重算 Stack 排名 → top15 打 KIT? + KIT 藥排名"); print("="*70)
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
col=comb[:,DIS]
cand=sorted([i for i in range(nd) if i not in known],key=lambda i:-col[i])
top15=cand[:15]
top15_hit=[d for d in top15 if KIT in dp.get(d,[])]
check("D: top15 沒一個打 KIT", len(top15_hit)==0, f"打 KIT 的 top15:{top15_hit}")
kit_ranks={node2db.get(d,d):(cand.index(d)+1) for d in kit_drugs if d in cand}
rmin=min(kit_ranks.values()); rmax=max(kit_ranks.values())
ima=kit_ranks.get("DB00619")
check("E: KIT 藥排名落在 #39–582 區間", 30<=rmin<=50 and 550<=rmax<=650, f"實際 #{rmin}–{rmax}")
check("E: imatinib(DB00619) ≈ #109", ima is not None and 95<=ima<=125, f"實際 #{ima}")

# 註:原本這裡有一項「F. 與 produce_candidates 的 top8 比對」,已移除。
#     produce_candidates.py 已被 run_all.py 取代(截斷也從 top8 改為 top50);
#     更根本的是 —— GNN 訓練非確定性,拿排名當一致性檢查只會產生假警報。

print("="*70); print("F. RDKit 決定性:重跑 PAINS/SA 與文件數字一致"); print("="*70)
params=FilterCatalog.FilterCatalogParams(); params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains=FilterCatalog.FilterCatalog(params)
DOC={"DB00997":(4.49,1),"DB00563":(3.09,0),"DB00541":(7.26,0)}  # 文件記的 (SA, PAINS)
for db,(sa_doc,pn_doc) in DOC.items():
    node=db2node.get(db); mol=Chem.MolFromSmiles(smiles.get(node,""))
    sa=sascorer.calculateScore(mol); pn=len(pains.GetMatches(mol))
    check(f"{db} SA≈{sa_doc}、PAINS={pn_doc} 可重現", abs(sa-sa_doc)<0.05 and pn==pn_doc, f"重算 SA {sa:.2f} / PAINS {pn}")

print("\n"+"="*70)
print(f"總計:PASS {len(PASS)} / FAIL {len(FAIL)}")
if FAIL: print("失敗項:", FAIL)
else: print("✅ 流水線所有關鍵宣稱獨立重算屬實")
print("="*70)
