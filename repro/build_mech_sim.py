#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建藥-藥相似度矩陣 = α·化學結構 + (1-α)·機轉/靶點,直接覆蓋 drug_drug_baseline.csv。
解決 MRDDA「藥-藥相似度被化學結構主導」的問題 —— 不改演算法,只換輸入矩陣。
文獻依據:DDA-SKF / LBMFF / SNF / AMFGNN(多視角相似度融合)。

用法:python build_mech_sim.py [α]        # α = 結構權重,預設 0.3(實測選出)

輸入(全部是原始檔,不需要任何額外檔案):
  omics/drug.csv                 ID / DrugBank ID / SMILES / Name
  associations/drug_protein.csv  藥→靶點
  associations/protein_gene.csv  靶點→基因
  associations/gene_pathway.csv  基因→通路

輸出:
  dataset/Kdataset/drug_drug_baseline.csv   融合矩陣(建圖與 prop 都吃這個檔)

【化學結構相似度從 SMILES 現算,不讀任何預先算好的檔】
  ECFP4 = RDKit Morgan fingerprint(radius=2, 2048 bits)+ Tanimoto。
  已驗證:此設定重現 REDDA 原始 drug_drug_baseline.csv 的 798,342 格,
  精確吻合 100.0000%、最大差異 0.00000000(894 個 SMILES 全部解析成功,耗時 0.2 秒)。
  → 因此增刪藥物時,只要在 omics/drug.csv 加/刪一列(含 SMILES)再重跑本腳本即可,
     不需要手動維護任何相似度數字。

  為什麼不讀 interactions/drug_drug.csv(原始的 ECFP4 邊表)?
  實測它有 26 格(13 對)寫成 0.0,但正確值是 1.0 —— 那 13 對全是立體異構物
  (dexamethasone/betamethasone、doxorubicin/epirubicin、tretinoin/isotretinoin、
   ofloxacin/levofloxacin、amphetamine/dextroamphetamine …)。ECFP4 忽略立體化學,
  指紋相同 → 1.0 才對。RDKit 現算與 REDDA 的矩陣一致,那份邊表才是上游的錯。

  也不可讀 drug_drug_baseline.csv —— 那是本腳本的【輸出】(已是融合結果),
  再讀就變成 fused(fused),每跑一次錯一層。
"""
import csv, sys

K = 'dataset/Kdataset/'
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3   # 結構權重


def load(fn, a, b):
    m = {}
    for row in csv.DictReader(open(K + fn)):
        m.setdefault(int(row[a]), set()).add(int(row[b]))
    return m


# ── 化學結構:從 SMILES 現算 ECFP4 ───────────────────────────────────────
rows = list(csv.DictReader(open(K + 'omics/drug.csv')))
N = len(rows)
smiles = {int(r['ID']): r['SMILES'] for r in rows}
dbid = {int(r['ID']): r['Drug'] for r in rows}

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:
    sys.exit("需要 RDKit 才能從 SMILES 算 ECFP4:pip install rdkit\n"
             "(本專案的 docker 映像 nvcr.io/nvidia/dgl:25.08 內已安裝)")
RDLogger.DisableLog('rdApp.*')

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps, unparsed = [], []
for i in range(N):
    mol = Chem.MolFromSmiles(smiles[i])
    if mol is None:
        unparsed.append(dbid[i]); fps.append(None)
    else:
        fps.append(gen.GetFingerprint(mol))
if unparsed:
    print(f"⚠️ {len(unparsed)} 個藥的 SMILES 無法解析,其結構相似度以 0 計:{unparsed[:8]}")
    print("   → 這些藥只靠機轉相似度連結;若是你新加的藥,請檢查 SMILES 是否正確。")

chem = [[0.0] * N for _ in range(N)]
for i in range(N):
    chem[i][i] = 1.0
    if fps[i] is None: continue
    js = [j for j in range(i + 1, N) if fps[j] is not None]
    if not js: continue
    for j, v in zip(js, DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in js])):
        chem[i][j] = chem[j][i] = v

# ── 機轉:靶點 / 通路 Jaccard ────────────────────────────────────────────
dp = load('associations/drug_protein.csv', 'Drug', 'Protein')
pg = load('associations/protein_gene.csv', 'Protein', 'Gene')
gp = load('associations/gene_pathway.csv', 'Gene', 'Pathway')
drug_path = {d: (set().union(*[gp.get(g, set()) for g in
                 set().union(*[pg.get(p, set()) for p in dp.get(d, set())])]) if dp.get(d) else set())
             for d in range(N)}
jac = lambda A, B: len(A & B) / len(A | B) if (A | B) else 0.0

fused = [[0.0] * N for _ in range(N)]
for i in range(N):
    Ti, Pi = dp.get(i, set()), drug_path[i]
    for j in range(i + 1, N):
        m = max(jac(Ti, dp.get(j, set())), jac(Pi, drug_path[j]))   # 靶點與通路取大,補稀疏
        fused[i][j] = fused[j][i] = ALPHA * chem[i][j] + (1 - ALPHA) * m
    fused[i][i] = 1.0

with open(K + 'drug_drug_baseline.csv', 'w', newline='') as f:
    w = csv.writer(f)
    for r in fused: w.writerow([f"{x:.4f}" for x in r])

print(f"✓ 融合完成 α={ALPHA}(結構) / {1-ALPHA:.1f}(機轉) → {K}drug_drug_baseline.csv ({N}×{N})")
print(f"  覆蓋:SMILES {N-len(unparsed)}/{N}、靶點 {sum(1 for d in range(N) if dp.get(d))}/{N}"
      f"、通路 {sum(1 for d in range(N) if drug_path[d])}/{N}")
