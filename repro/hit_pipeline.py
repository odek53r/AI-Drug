#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
進場檢查流水線(hit-to-lead 事前粗篩)· 肥大細胞瘤(KIT 驅動)候選
串起關卡:
  ① 有效性(機制):候選有沒有打到驅動靶點 KIT + 選擇性(打幾個靶)  [用 KG]
  ② 可合成性:RDKit SA score(現有藥多半 OK)
  ③ 安全粗篩:PAINS 假陽性警訊 + Lipinski 類藥性 + QED + 描述子       [RDKit]
  ④ IP:軟體做不了,標 N/A(需律師 FTO)
輸出一張 Go/No-Go 總表。
誠實提醒:老藥的 ②③ 多半已由臨床數據滿足,這裡是「重新確認 + 選擇性/靶點」;預測值只當粗篩。
"""
import csv, sys, os
import numpy as np
np.seterr(all="ignore")
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, FilterCatalog, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer; HAS_SA=True
except Exception:
    HAS_SA=False

def load(fn): return np.array([[float(x) for x in r] for r in csv.reader(open(fn))],dtype=np.float32)
Y=load("dataset/KPet/KPet_baseline.csv"); nd,ndis=Y.shape
DIS=503; KIT=17695
known=set(np.where(Y[:,DIS]==1)[0])

# --- Stack 排序 (同 run_all.py 的 Stack) ---
zc=lambda M:(M-M.mean(0))/(M.std(0)+1e-9)
def norm(S): S=(S+S.T)/2; S=S.copy(); np.fill_diagonal(S,0); return S/np.maximum(S.sum(1,keepdims=True),1e-9)
Nf=norm(load("dataset/KPet/drug_sim_fused.csv"))
ddb=load("dataset/Kdataset/disease_disease_baseline.csv")
Sdis=np.zeros((ndis,ndis),dtype=np.float32); Sdis[:454,:454]=ddb
for row in csv.DictReader(open("dataset/KPet/KPet_pet_disease_disease.csv")):
    a,b=int(row["Disease1"]),int(row["Disease2"]); h,p=(a,b) if a<b else (b,a)
    if p>=454 and h<454: Sdis[p,:454]=ddb[h,:]; Sdis[:454,p]=ddb[:,h]; Sdis[p,h]=Sdis[h,p]=1
NSdis=norm(Sdis)
Az=zc(load("resultKPetSup2_par_42/result.csv"))
def prop(Y0,it=20):
    F=Y0.copy()
    for _ in range(it): F=0.45*(Nf@F)+0.45*(F@NSdis.T)+0.1*Y0
    return F
def nmf(Y0,r=50,it=80):
    rs=np.random.RandomState(0); W=np.abs(rs.rand(nd,r)).astype(np.float32); H=np.abs(rs.rand(r,ndis)).astype(np.float32)
    for _ in range(it): H*=(W.T@Y0)/np.maximum(W.T@W@H,1e-6); W*=(Y0@H.T)/np.maximum(W@H@H.T,1e-6)
    return W@H
comb=Az+0.7*zc(prop(Y.copy()))+0.5*zc(nmf(Y.copy()))
order=[d for d in np.argsort(-comb[:,DIS]) if d not in known][:15]   # top15 候選

# --- 名稱 / SMILES / KG 靶點 ---
smiles={int(r["ID"]):r["SMILES"] for r in csv.DictReader(open("dataset/Kdataset/omics/drug.csv"))}
node2db={int(r["ID"]):r["Drug"] for r in csv.DictReader(open("dataset/Kdataset/omics/drug.csv"))}
DBN={"DB00619":"imatinib","DB01254":"dasatinib","DB00398":"sorafenib","DB01268":"sunitinib","DB08896":"regorafenib","DB00997":"doxorubicin","DB00444":"teniposide","DB00773":"etoposide","DB01229":"paclitaxel","DB00570":"vinblastine","DB00541":"vincristine","DB01590":"everolimus","DB00877":"sirolimus","DB09053":"ibrutinib","DB12010":"fostamatinib","DB08901":"ponatinib","DB04868":"nilotinib","DB00171":"ATP"}
nm=lambda i: DBN.get(node2db.get(i,""), node2db.get(i,str(i)))
dp={}
for r in csv.DictReader(open("dataset/Kdataset/associations/drug_protein.csv")):
    dp.setdefault(int(r["Drug"]),[]).append(int(r["Protein"]))

# --- PAINS 過濾器 ---
params=FilterCatalog.FilterCatalogParams(); params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains=FilterCatalog.FilterCatalog(params)

print("="*118)
print("進場檢查總表 · 肥大細胞瘤(KIT 驅動)· Stack top15 候選 · 現有藥重定位")
print("="*118)
hdr=f"{'候選藥':<15}{'①打KIT':<8}{'選擇性':<8}{'②SA':<7}{'③PAINS':<8}{'類藥Ro5':<8}{'QED':<6}{'MW':<7}{'LogP':<7}{'判定'}"
print(hdr); print("-"*118)
rows_out=[["候選藥","①打KIT","靶點數","②SA_score","③PAINS警訊","Ro5違反","QED","MW","LogP","綜合判定"]]
for d in order:
    smi=smiles.get(d,""); mol=Chem.MolFromSmiles(smi) if smi else None
    tgts=dp.get(d,[]); hitkit="✅" if KIT in tgts else "—"; nt=len(tgts)
    if mol is None:
        print(f"{nm(d):<15}{hitkit:<8}{nt:<8}{'?':<7}{'?':<8}{'?':<8}{'?':<6}{'?':<7}{'?':<7}(無SMILES)")
        continue
    mw=Descriptors.MolWt(mol); logp=Crippen.MolLogP(mol); hbd=Descriptors.NumHDonors(mol); hba=Descriptors.NumHAcceptors(mol)
    ro5=sum([mw>500,logp>5,hbd>5,hba>10]); qed=QED.qed(mol)
    sa=sascorer.calculateScore(mol) if HAS_SA else float("nan")
    npains=len(pains.GetMatches(mol))
    # 綜合判定:打KIT=強;SA<4.5 且 PAINS=0 且 Ro5<=1 = 過事前篩
    g1 = KIT in tgts
    g2 = (not np.isnan(sa)) and sa<4.5
    g3 = (npains==0) and (ro5<=1)
    verdict = "✅ 進場" if (g2 and g3) else "⚠️ 注意"
    if g1: verdict += "・打KIT"
    print(f"{nm(d):<15}{hitkit:<8}{nt:<8}{sa:<7.2f}{npains:<8}{ro5:<8}{qed:<6.2f}{mw:<7.0f}{logp:<7.2f}{verdict}")
    rows_out.append([nm(d),"是" if g1 else "否",nt,f"{sa:.2f}",npains,ro5,f"{qed:.2f}",f"{mw:.0f}",f"{logp:.2f}",verdict])
print("-"*118)
print("關卡④ IP:軟體做不了,需律師 FTO 檢索 → 全部標 N/A")
print("判定說明:②SA<4.5=好合成 ③PAINS=0且Ro5違反<=1=無明顯警訊 → 事前篩過='進場';打KIT=機制對得上(加分)")
print("誠實:老藥的②③多為已知(這裡重新確認);預測只當粗篩,真有效性/安全仍需外包實驗。")
with open("hit_gono_mct.csv","w",newline="") as f: csv.writer(f).writerows(rows_out)
print("\n→ 已存 hit_gono_mct.csv")
