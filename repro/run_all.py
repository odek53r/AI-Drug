#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_all.py — 一鍵完整重現候選藥物(全資料訓練版)

跑完整條流水線並自我驗證:
  ① 資料完整性檢查(維度、標籤數)
  ② 載入凍結的全資料 GNN(result_full.csv)
  ③ prop 雙向標籤傳播(秒級現算,零參數)
  ④ NMF 非負矩陣分解(秒級現算,零參數)
  ⑤ Stack 合成(權重鎖死 β=0.7 / γ=0.5)
  ⑥ 產候選:對「還沒有藥-病關聯」的格子排名
  ⑦ 自我驗證:比對已知指紋,確認完全重現

── 兩個設計決定(重要)────────────────────────────────────────
為什麼不在這裡重訓 GNN?
    GNN 訓練用 GPU atomicAdd,有非確定性 → 重訓不會逐位元相同 → 候選會抖動。
    本腳本讀「凍結的 result_full.csv」→ 後段全部確定性 → 候選 100% 可重現。
    要自己重訓:python train_parallel.py -da KPet --mode full -sp resultKPetFull -se 42

為什麼權重 β/γ 鎖死而不搜?
    β=0.7/γ=0.5 是當初用「誠實的 10 折 OOF 版」選出來的。
    全資料版 GNN 看過所有標籤,拿它重搜權重 = 用洩漏的分數挑參數 → 不可。

⚠️ 誠實界線:本腳本產的是「候選假設」,不是已證實的療法。
    效能數字請看 nested_cv.py(用 OOF 版,leak-free):全部 74±1 / 非人氣 59±1 / 真novel 14±4
"""
import csv, json, os, sys
import numpy as np
np.seterr(all="ignore")

GNN_FULL = "resultKPetFull_42/result_full.csv"   # 全資料訓練(產候選用)
BETA, GAMMA = 0.7, 0.5                            # 鎖死:由 OOF 版誠實選出
TOPK = 8
FAIL = []

def ok(cond, msg, detail=""):
    print(f"  [{'✅' if cond else '❌'}] {msg}" + (f" — {detail}" if detail else ""))
    if not cond: FAIL.append(msg)

def load(fn): return np.array([[float(x) for x in r] for r in csv.reader(open(fn))], dtype=np.float32)

print("=" * 72)
print("① 資料完整性檢查")
print("=" * 72)
for f in [GNN_FULL, "dataset/KPet/KPet_baseline.csv", "dataset/KPet/drug_sim_fused.csv",
          "dataset/Kdataset/disease_disease_baseline.csv", "dataset/KPet/KPet_pet_disease_disease.csv",
          "dataset/KPet/KPet_pet_diseases.csv", "dataset/Kdataset/omics/drug.csv", "sup_positives.json"]:
    ok(os.path.exists(f) and os.path.getsize(f) > 0, f"{f} 存在且非空")
if FAIL:
    print("\n❌ 缺檔,無法continue"); sys.exit(1)

Y = load("dataset/KPet/KPet_baseline.csv"); nd, ndis = Y.shape
sp = json.load(open("sup_positives.json"))
ok(Y.shape == (894, 504), "標籤矩陣 = 894 藥 × 504 病", f"{nd}×{ndis}")
ok(int(Y[:, :454].sum()) == 2704, "人類標籤 = 2,704", f"{int(Y[:,:454].sum())}")
ok(int(Y[:, 454:].sum()) == 121, "寵物標籤 = 121", f"{int(Y[:,454:].sum())}")
ok(len(sp) == 121, "sup_positives.json = 121 筆")

print("=" * 72)
print("② 載入凍結的全資料 GNN")
print("=" * 72)
GNN = load(GNN_FULL)
ok(GNN.shape == (894, 504), f"{GNN_FULL} = 894 × 504", f"{GNN.shape}")
print(f"  分數範圍 {GNN.min():.4f}~{GNN.max():.4f};已知有效藥平均 {GNN[Y==1].mean():.3f} / 未知平均 {GNN[Y==0].mean():.3f}")

zc = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-9)
def norm(S):
    S = (S + S.T) / 2; S = S.copy(); np.fill_diagonal(S, 0)
    return S / np.maximum(S.sum(1, keepdims=True), 1e-9)

Nf = norm(load("dataset/KPet/drug_sim_fused.csv"))
ddb = load("dataset/Kdataset/disease_disease_baseline.csv")
Sdis = np.zeros((ndis, ndis), dtype=np.float32); Sdis[:454, :454] = ddb
nb = 0
for row in csv.DictReader(open("dataset/KPet/KPet_pet_disease_disease.csv")):
    a, b = int(row["Disease1"]), int(row["Disease2"]); h, p = (a, b) if a < b else (b, a)
    if p >= 454 and h < 454:
        Sdis[p, :454] = ddb[h, :]; Sdis[:454, p] = ddb[:, h]; Sdis[p, h] = Sdis[h, p] = 1; nb += 1
NSdis = norm(Sdis)
ok(nb == 100, "寵物病↔人類病橋接 = 100 列(50 條雙向)", f"{nb}")

print("=" * 72)
print("③④ prop 雙向傳播 + NMF(秒級現算,零訓練參數)")
print("=" * 72)
def prop(Y0, it=20):
    F = Y0.copy()
    for _ in range(it): F = 0.45 * (Nf @ F) + 0.45 * (F @ NSdis.T) + 0.1 * Y0
    return F
def nmf(Y0, r=50, it=80):
    rs = np.random.RandomState(0)
    W = np.abs(rs.rand(nd, r)).astype(np.float32); H = np.abs(rs.rand(r, ndis)).astype(np.float32)
    for _ in range(it):
        H *= (W.T @ Y0) / np.maximum(W.T @ W @ H, 1e-6); W *= (Y0 @ H.T) / np.maximum(W @ H @ H.T, 1e-6)
    return W @ H
P = prop(Y.copy()); N = nmf(Y.copy())
print(f"  prop 完成(20 次迭代)· NMF 完成(rank=50, 80 次迭代)· 皆用完整 Y(不遮任何標籤)")

print("=" * 72)
print(f"⑤ Stack 合成:z(GNN) + {BETA}·z(prop) + {GAMMA}·z(NMF)")
print("=" * 72)
comb = zc(GNN) + BETA * zc(P) + GAMMA * zc(N)
print(f"  權重鎖死 β={BETA} γ={GAMMA}(由 OOF 版誠實選出,未用全資料版重搜)")

print("=" * 72)
print("⑥ 產候選:對「還沒有藥-病關聯」的格子排名")
print("=" * 72)
node2db = {int(r["ID"]): r["Drug"] for r in csv.DictReader(open("dataset/Kdataset/omics/drug.csv"))}
node2name = {r[0]: r[2] for r in sp}
DBN = {"DB00619":"imatinib","DB01254":"dasatinib","DB00398":"sorafenib","DB01268":"sunitinib",
       "DB08896":"regorafenib","DB01229":"paclitaxel","DB00444":"teniposide","DB00773":"etoposide",
       "DB00694":"daunorubicin","DB01177":"idarubicin","DB00970":"dactinomycin","DB00997":"doxorubicin",
       "DB00541":"vincristine","DB00570":"vinblastine","DB01590":"everolimus","DB00877":"sirolimus",
       "DB06176":"romidepsin","DB11581":"venetoclax","DB00563":"methotrexate","DB01204":"mitoxantrone"}
nm = lambda d: node2name.get(d) or DBN.get(node2db.get(d, ""), node2db.get(d, str(d)))
pet = {int(r["kpet_index"]): r["name"].replace("寵物-", "") for r in csv.DictReader(open("dataset/KPet/KPet_pet_diseases.csv"))}

rows = [["disease", "rank", "drug", "drugbank", "score"]]
n_unknown = 0
for c in range(454, 504):
    known = set(np.where(Y[:, c] == 1)[0])
    cand = [d for d in np.argsort(-comb[:, c]) if d not in known]
    n_unknown += len(cand)
    for r, d in enumerate(cand[:TOPK], 1):
        rows.append([pet.get(c, c), r, nm(d), node2db.get(d, ""), f"{comb[d][c]:.2f}"])
with open("stack_candidates.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"  待預測格子:894 藥 × 50 寵物病 − 121 已知 = {n_unknown:,} 格")
print(f"  → stack_candidates.csv:50 病 × top{TOPK} = {len(rows)-1} 個候選")

print("=" * 72)
print("⑦ 自我驗證:比對已知指紋(確認完全重現)")
print("=" * 72)
db2node = {v: k for k, v in node2db.items()}
DIS = 462
known462 = set(np.where(Y[:, DIS] == 1)[0])
c462 = [d for d in np.argsort(-comb[:, DIS]) if d not in known462]
EXPECT = [("daunorubicin", "DB00694", 5), ("dactinomycin", "DB00970", 8),
          ("idarubicin", "DB01177", 9), ("teniposide", "DB00444", 10)]
for name, db, exp in EXPECT:
    r = c462.index(db2node[db]) + 1
    ok(r == exp, f"犬淋巴瘤 {name} 排名 = #{exp}", f"實際 #{r}")
for name, db, _ in EXPECT:
    d = db2node[db]
    ok(Y[d, DIS] == 0, f"{name} 對犬淋巴瘤是「未知」(無洩漏)", f"Y={Y[d,DIS]:.0f}")

print("=" * 72)
if FAIL:
    print(f"❌ {len(FAIL)} 項未通過:{FAIL}"); sys.exit(1)
print("✅ 全部通過 — 候選藥物完全重現")
print("=" * 72)
print("誠實提醒:")
print("  · 這些是「候選假設」,不是已證實療法;真效力需濕實驗驗證。")
print("  · 本檔用全資料 GNN(看過所有標籤)→ 只能產候選,不可報 recall/AUC。")
print("  · 誠實效能請跑 nested_cv.py(OOF 版):全部 74±1 / 非人氣 59±1 / 真novel 14±4(隨機 5.6%)")
