import csv, sys
from admet_ai import ADMETModel
sm={int(r["ID"]):r["SMILES"] for r in csv.DictReader(open("/workspace/MRDDA/dataset/Kdataset/omics/drug.csv"))}
smis={"teniposide":sm[110],"etoposide":sm[9]}
model=ADMETModel()
df=model.predict(smiles=list(smis.values()))
df.index=list(smis.keys())
df.to_csv("/workspace/MRDDA/dock/admet_out.csv")
# 挑關鍵終點(名稱依 admet-ai 欄位;找得到才印)
KEY=["molecular_weight","logP","hydrogen_bond_donors","hydrogen_bond_acceptors",
     "hERG","DILI","AMES","ClinTox","Carcinogens_Lagunin","LD50_Zhu",
     "CYP3A4_Veith","CYP2D6_Veith","Caco2_Wang","BBB_Martins",
     "Bioavailability_Ma","Solubility_AqSolDB","HIA_Hou"]
cols=[c for c in KEY if c in df.columns]
print("=== ADMET 關鍵終點(teniposide vs etoposide)===")
print(df[cols].T.to_string())
print("\n欄位總數:",len(df.columns))
