#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建「機轉/靶點相似度」矩陣,並與化學結構相似度融合(Similarity Kernel Fusion 精神)。
解決 MRDDA「藥-藥相似度被化學結構主導」的問題 —— 不改演算法,只換輸入矩陣。
文獻依據:DDA-SKF / LBMFF / SNF / AMFGNN(多視角相似度融合)。

輸出(可經 KPET_DRUGSIM 注入訓練):
  dataset/KPet/drug_mech_sim.csv   純機轉(靶點∪通路 Jaccard)
  dataset/KPet/drug_sim_fused.csv  融合 = α·結構 + (1-α)·機轉
"""
import csv, sys
K='dataset/Kdataset/'
ALPHA=float(sys.argv[1]) if len(sys.argv)>1 else 0.3   # 結構權重

def load(fn,a,b):
    m={}
    for row in csv.DictReader(open(K+fn)): m.setdefault(int(row[a]),set()).add(int(row[b]))
    return m

chem=[[float(x) for x in row] for row in csv.reader(open(K+'drug_drug_baseline.csv'))]
N=len(chem)
dp=load('associations/drug_protein.csv','Drug','Protein')
pg=load('associations/protein_gene.csv','Protein','Gene')
gp=load('associations/gene_pathway.csv','Gene','Pathway')
drug_path={d:(set().union(*[gp.get(g,set()) for g in
             set().union(*[pg.get(p,set()) for p in dp.get(d,set())])]) if dp.get(d) else set())
           for d in range(N)}
jac=lambda A,B: len(A&B)/len(A|B) if (A|B) else 0.0

mech=[[0.0]*N for _ in range(N)]; fused=[[0.0]*N for _ in range(N)]
for i in range(N):
    Ti,Pi=dp.get(i,set()),drug_path[i]
    for j in range(i+1,N):
        m=max(jac(Ti,dp.get(j,set())), jac(Pi,drug_path[j]))   # 靶點與通路取大,補稀疏
        mech[i][j]=mech[j][i]=m
        fused[i][j]=fused[j][i]=ALPHA*chem[i][j]+(1-ALPHA)*m
    mech[i][i]=fused[i][i]=1.0

def save(mat,fn):
    with open(fn,'w',newline='') as f:
        w=csv.writer(f)
        for r in mat: w.writerow([f"{x:.4f}" for x in r])
save(mech,'dataset/KPet/drug_mech_sim.csv')
save(fused,'dataset/KPet/drug_sim_fused.csv')
print(f"✓ 融合完成 α={ALPHA}(結構) / {1-ALPHA:.1f}(機轉)。覆蓋:靶點 {sum(1 for d in range(N) if dp.get(d))}/{N}、通路 {sum(1 for d in range(N) if drug_path[d])}/{N}")
