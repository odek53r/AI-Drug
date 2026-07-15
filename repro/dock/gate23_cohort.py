#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 關卡②(SA)+③(ADMET)· 淋巴瘤 TOP2A 世代
import csv, json, os, sys
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, FilterCatalog, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer
from admet_ai import ADMETModel

COHORT = [("teniposide","DB00444"),("etoposide","DB00773"),
          ("daunorubicin","DB00694"),("idarubicin","DB01177"),("dactinomycin","DB00970")]
db2s = {r["Drug"]: r["SMILES"] for r in csv.DictReader(open("/workspace/MRDDA/dataset/Kdataset/omics/drug.csv"))}
params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains = FilterCatalog.FilterCatalog(params)

rows = {}
for nm, db in COHORT:
    mol = Chem.MolFromSmiles(db2s[db])
    mw = Descriptors.MolWt(mol); logp = Crippen.MolLogP(mol)
    ro5 = sum([mw > 500, logp > 5, Descriptors.NumHDonors(mol) > 5, Descriptors.NumHAcceptors(mol) > 10])
    rows[nm] = dict(db=db, MW=round(mw, 0), logP=round(logp, 2),
                    SA=round(sascorer.calculateScore(mol), 2),
                    PAINS=len(pains.GetMatches(mol)), Ro5=int(ro5), QED=round(QED.qed(mol), 2))

model = ADMETModel()
df = model.predict(smiles=[db2s[db] for _, db in COHORT])
df.index = [nm for nm, _ in COHORT]

def pc(nm, k):
    col = [c for c in df.columns if c.startswith(k) and "percentile" in c]
    return round(float(df[col[0]][nm]), 0) if col else None

for nm, _ in COHORT:
    for k in ["hERG", "DILI", "AMES", "ClinTox"]:
        rows[nm][k] = round(float(df[k][nm]), 2)
        rows[nm][k + "_pc"] = pc(nm, k)

json.dump(rows, open("/workspace/MRDDA/dock/gate23_cohort.json", "w"), ensure_ascii=False, indent=1)
print("=== 關卡②③ 世代結果 ===")
for nm, _ in COHORT:
    r = rows[nm]
    print("%-14s SA %-5s PAINS %s Ro5 %s | hERG %s(%s%%) DILI %s(%s%%) AMES %s(%s%%)" %
          (nm, r["SA"], r["PAINS"], r["Ro5"], r["hERG"], r["hERG_pc"],
           r["DILI"], r["DILI_pc"], r["AMES"], r["AMES_pc"]))
print("→ 存 gate23_cohort.json")
