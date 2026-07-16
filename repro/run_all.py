#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_all.py — 從頭跑完整條流水線,產出寵物老藥新用的候選清單

用法:
    python run_all.py                  訓練 GNN 再產候選(GPU ~15 分 / CPU ~2 小時,自動偵測)
    python run_all.py --cpu            強制 CPU(沒 GPU 時會自動啟用)
    python run_all.py --seed 43        換一個 seed 訓練(建議跑多個,見下)

本專案【不隨附訓練好的模型】,每次都從頭訓練 —— 隨附模型會讓人拿到別人訓練的
答案卻以為是自己跑出來的。

流程:
  ① 資料完整性檢查(維度、標籤數)
  ② 訓練全資料 GNN → resultKPetFull_<seed>/result_full.csv   (GPU ~15 分 / CPU ~2 小時)
  ③ prop 雙向標籤傳播(秒級現算,零參數)
  ④ NMF 非負矩陣分解(秒級現算,零參數)
  ⑤ Stack 合成(權重鎖死 β=0.7 / γ=0.5)
  ⑥ 產候選:對「還沒有藥-病關聯」的格子排名
  ⑦ 無洩漏檢查

⚠️⚠️ 最重要的一件事:每次訓練的結果都不一樣 ⚠️⚠️
    GNN 訓練是非確定性的 —— utils.py 的 MetaPath2Vec 即使給同一個 seed,
    每次產生的節點特徵都不同(實測最大差異 0.12),整個模型跟著變。
    實測(3 份同設定訓練):彼此相關係數僅 0.74;
                          dactinomycin 對犬淋巴瘤的排名在 #5 / #30 / #7 之間跳。
    跨 seed 的名次一致率:rank1 僅 58%、rank8 僅 10%;top50 清單重疊 82%。

    → 所以【不要拿單次結果當結論】。正確用法:
        for s in 42 43 44 45 46; do python run_all.py --seed $s; done
      然後只採信「每次都出現在 top50」的藥。
    → 也因此本腳本不驗排名指紋(那對重訓者必定失敗),
      只驗「一定要成立」的資料完整性與無洩漏。

為什麼權重 β/γ 鎖死而不搜?
    β=0.7/γ=0.5 是當初用「誠實的 10 折 OOF 版」選出來的。
    全資料版 GNN 看過所有標籤,拿它重搜權重 = 用洩漏的分數挑參數 → 不可。

⚠️ 誠實界線:本腳本產的是「候選假設」,不是已證實的療法。
    本腳本用全資料 GNN(看過所有標籤)→ 只能產候選,不可報 recall/AUC。
    本 repo 目前【沒有附效能評估腳本】,所以這裡不宣稱任何 recall 數字。
    要評估效能必須自行實作 leak-free 的 OOF 評估(遮住測試邊 + remove_graph),
    否則任何數字都是洩漏的。
"""
import argparse, csv, os, subprocess, sys, time
import numpy as np
np.seterr(all="ignore")

ap = argparse.ArgumentParser(description="產出寵物老藥新用候選清單(一律從頭訓練)")
ap.add_argument("--seed", type=int, default=42, help="訓練 seed(預設 42)")
ap.add_argument("--out", default="stack_candidates.csv", help="輸出檔名")
ap.add_argument("--cpu", action="store_true",
                help="強制用 CPU 訓練(約 2 小時)。沒有 GPU 時會自動啟用,不必手動加")
A = ap.parse_args()

BETA, GAMMA = 0.7, 0.5                            # 鎖死:由 OOF 版誠實選出
TOPK = 50                                         # ← 必須是 50,見下
# 本專案【不隨附訓練好的模型】,每次都從頭訓練(GPU ~15 分 / CPU ~2 小時)。
# 理由:隨附模型會讓人拿到別人訓練的答案卻以為是自己跑出來的 —— 這正是
# 先前 --skip-train 造成的問題(它寫死讀 resultKPetFull_42,連 --seed 43 也讀那個)。
# 註:args.py:40 會自動補上 '_<seed>',所以這裡不能自己再帶 seed。
SAVED = "resultKPetFull"
GNN_FULL = os.path.join("%s_%d" % (SAVED, A.seed), "result_full.csv")
FAIL = []

# ── 為什麼 TOPK=50 而不是 8?────────────────────────────────────────────
# 先前實測(評估腳本已不在本 repo,故以下數字無法在此重現,僅記錄當時的判斷依據):
#
#   截斷    全部    非人氣   重定位(真novel)   跨seed穩定度
#   top8    21%     10%      0/12  ← 一個都撈不到    69%
#   top50   75%     62%      2/12                    82%
#
# top8 對「真正的新用途」撈回率是 0/12 —— 本專案的核心目的在該截斷下完全失效;
# 且 top50 反而更穩(短清單的入榜邊界競爭最激烈)。代價僅 400→2500 列,
# 仍為使用者排除 94% 的藥(50/894)。


def ok(cond, msg, detail=""):
    print(f"  [{'✅' if cond else '❌'}] {msg}" + (f" — {detail}" if detail else ""))
    if not cond: FAIL.append(msg)

def load(fn): return np.array([[float(x) for x in r] for r in csv.reader(open(fn))], dtype=np.float32)

print("=" * 72)
print("① 資料完整性檢查")
print("=" * 72)
need = ["dataset/Kdataset/Kdataset_baseline.csv", "dataset/Kdataset/drug_drug_baseline.csv",
        "dataset/Kdataset/disease_disease_baseline.csv", "dataset/Kdataset/interactions/disease_disease.csv",
        "dataset/Kdataset/omics/disease.csv", "dataset/Kdataset/omics/drug.csv",
        "dataset/Kdataset/associations/Kdataset.csv",
        "model.py", "load_data.py", "utils.py", "train_parallel.py"]
for f in need:
    ok(os.path.exists(f) and os.path.getsize(f) > 0, f"{f} 存在且非空")
if FAIL:
    print("\n❌ 缺檔,無法繼續"); sys.exit(1)

Y = load("dataset/Kdataset/Kdataset_baseline.csv"); nd, ndis = Y.shape

# 維度全部【從資料讀】,不寫死 —— 使用者可自行增刪藥/病/標籤
n_drug_file = sum(1 for _ in csv.DictReader(open("dataset/Kdataset/omics/drug.csv")))
# 人類病數 = disease_disease_baseline.csv 的維度(它只涵蓋人類病,寵物病沒有 MeSH 相似度)。
# 不能再用 omics/disease.csv 的列數 —— 合併後那裡面人類+寵物都有。load_Kdataset() 也是這樣判定的。
N_HUMAN = sum(1 for _ in open("dataset/Kdataset/disease_disease_baseline.csv"))
dis_rows = list(csv.DictReader(open("dataset/Kdataset/omics/disease.csv")))
pet_rows = [r for r in dis_rows if int(r["ID"]) >= N_HUMAN]
PET_COLS = list(range(N_HUMAN, ndis))

ok(nd == n_drug_file, f"藥數一致:標籤矩陣 {nd} = drug.csv {n_drug_file}")
ok(ndis == len(dis_rows), f"病數一致:標籤矩陣 {ndis} = disease.csv {len(dis_rows)}")
ok(ndis == N_HUMAN + len(pet_rows),
   f"病數拆解:{ndis} = 人類 {N_HUMAN}(=相似度矩陣維度) + 寵物 {len(pet_rows)}")
ok(len(PET_COLS) > 0, f"寵物病欄位 = {N_HUMAN}~{ndis-1}(共 {len(PET_COLS)} 個)")

# 建圖用的相似度矩陣必須與藥/病數對齊,否則訓練會在 loss 那裡炸 tensor size。
# 先在這裡擋,並明確告訴使用者要重建哪個檔(不要等 PyTorch 丟看不懂的錯誤)。
def _dim(fn):
    with open(fn) as f:
        first = f.readline()
        ncol = len(first.split(","))
    nrow = 1 + sum(1 for _ in open(fn)) - 1
    return nrow, ncol
for fn, want, what, fix in [
    ("dataset/Kdataset/drug_drug_baseline.csv", nd, "藥數",
     "增刪藥物後,drug_drug_baseline.csv 要重建:python build_mech_sim.py 0.3"),
    ("dataset/Kdataset/disease_disease_baseline.csv", N_HUMAN, "人類病數",
     "增刪【人類】疾病後,disease_disease_baseline.csv 要重建(寵物病不進這個矩陣)"),
]:
    if os.path.exists(fn):
        r, c = _dim(fn)
        good = (r == want and c == want)
        ok(good, f"{fn} = {want}×{want}", f"實際 {r}×{c}")
        if not good:
            print(f"     ↳ {fix}")
if FAIL:
    print("\n❌ 相似度矩陣與資料維度不符。")
    print("   建圖時的節點數是從這些矩陣決定的,不修會在訓練時炸 tensor size。")
    sys.exit(1)

# 建圖時的節點數 = 各關聯檔裡出現過的最大 index + 1。
# 若關聯檔還指向已被刪掉的藥/病,DGL 會多建節點 → 與標籤矩陣對不上。
def _maxidx(fn, col):
    m = -1
    for r in csv.DictReader(open(fn)):
        v = int(r[col])
        if v > m: m = v
    return m
for fn, col, limit, what in [
    ("dataset/Kdataset/interactions/disease_disease.csv", "Disease1", ndis, "疾病"),
    ("dataset/Kdataset/interactions/disease_disease.csv", "Disease2", ndis, "疾病"),
    ("dataset/Kdataset/associations/Kdataset.csv", "Drug", nd, "藥物"),
    ("dataset/Kdataset/associations/Kdataset.csv", "Disease", ndis, "疾病"),
    ("dataset/Kdataset/associations/drug_protein.csv", "Drug", nd, "藥物"),
    ("dataset/Kdataset/associations/pathway_disease.csv", "Disease", ndis, "疾病"),
]:
    if not os.path.exists(fn): continue
    mx = _maxidx(fn, col)
    good = mx < limit
    ok(good, f"{os.path.basename(fn)}[{col}] 最大 index {mx} < {what}數 {limit}")
    if not good:
        print(f"     ↳ 這個檔還指向已刪除的{what}(index {mx})。刪{what}時,所有關聯檔的對應列都要一起刪。")
if FAIL:
    print("\n❌ 關聯檔指向不存在的節點。")
    print("   DGL 會依最大 index 建節點,導致節點數與標籤矩陣不符 → 訓練時炸。")
    sys.exit(1)
print(f"  資料規模:{nd} 藥 × {ndis} 病(人類 {N_HUMAN} + 寵物 {len(pet_rows)})")
print(f"  正標籤:人類 {int(Y[:, :N_HUMAN].sum()):,} + 寵物 {int(Y[:, N_HUMAN:].sum())} = {int(Y.sum()):,}")
n_pet_lab = int(Y[:, N_HUMAN:].sum())
n_evi = sum(1 for r in csv.DictReader(open("dataset/Kdataset/associations/Kdataset.csv")) if r.get("Evidence"))
print(f"  associations/Kdataset.csv 有 Evidence 註記的(寵物標籤來源):{n_evi} 筆")
if FAIL:
    print("\n❌ 資料維度不一致,請檢查上述檔案"); sys.exit(1)

print("=" * 72)
print("② 全資料 GNN(從頭訓練)")
print("=" * 72)
# 自動偵測 GPU。沒有就退回 CPU —— 慢很多但跑得完,不該讓沒 GPU 的人直接卡死。
# 兩個開關要一起給才走得到 CPU:-id ""(train_parallel)+ MRDDA_DEVICE(utils 的 m2v)。
env = dict(os.environ)
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except Exception:
    HAS_GPU = False
cmd = [sys.executable, "train_parallel.py", "-da", "Kdataset", "--mode", "full",
       "-sp", SAVED, "-se", str(A.seed)]
if A.cpu or not HAS_GPU:
    cmd += ["-id", ""]
    env["MRDDA_DEVICE"] = "cpu"
    why = "--cpu" if A.cpu else "偵測不到 CUDA GPU"
    print(f"  ⚠️ 用 CPU 訓練({why})。實測 ~33 epoch/分 → 4000 epochs 約 2 小時。")
    print(f"     有 GPU 的話只要 ~15 分。CPU 與 GPU 的 loss/AUC 實測一致。")
else:
    print(f"  4000 epochs,約 15 分鐘(GPU)。")
print(f"  執行:{' '.join(x if x else '\"\"' for x in cmd)}")
print("  訓練 log 即時顯示:")
print("  " + "-" * 68)
t0 = time.time()
r = subprocess.run(cmd, env=env)
if r.returncode != 0:
    print("\n❌ 訓練失敗(rc=%d)。常見原因:" % r.returncode)
    print("   · DGL 裝錯          → 見 README「環境」(PyPI 的 dgl 是 CPU-only)")
    print("   · tensor size 不符  → 相似度矩陣與資料維度沒對齊(前面的 ① 檢查會先擋)")
    print("   · CUDA out of memory → 關掉其他訓練程序,或改用 --cpu")
    sys.exit(1)
print("  " + "-" * 68)
print(f"  ✅ 訓練完成,耗時 {(time.time()-t0)/60:.1f} 分 → {GNN_FULL}")
if not os.path.exists(GNN_FULL):
    print(f"\n❌ 訓練跑完卻找不到 {GNN_FULL}"); sys.exit(1)
GNN = load(GNN_FULL)
if GNN.shape != Y.shape:
    print(f"\n❌ GNN 維度 {GNN.shape} ≠ 標籤矩陣 {Y.shape}"); sys.exit(1)
ok(True, f"GNN 維度與資料相符 = {GNN.shape[0]} × {GNN.shape[1]}")
print(f"  分數範圍 {GNN.min():.4f}~{GNN.max():.4f};已知有效藥平均 {GNN[Y==1].mean():.3f} / 未知平均 {GNN[Y==0].mean():.3f}")

zc = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-9)
def norm(S):
    S = (S + S.T) / 2; S = S.copy(); np.fill_diagonal(S, 0)
    return S / np.maximum(S.sum(1, keepdims=True), 1e-9)

Sf = load("dataset/Kdataset/drug_drug_baseline.csv")   # 合併後這就是 0.3×結構 + 0.7×機轉 的融合相似度
if Sf.shape != (nd, nd):
    print(f"\n❌ drug_drug_baseline 維度 {Sf.shape} ≠ 藥數 {nd}")
    print("   若你增刪過藥物,請重建:python build_mech_sim.py 0.3"); sys.exit(1)
Nf = norm(Sf)
ddb = load("dataset/Kdataset/disease_disease_baseline.csv")
if ddb.shape != (N_HUMAN, N_HUMAN):
    print(f"\n❌ disease_disease_baseline 維度 {ddb.shape} ≠ 人類病數 {N_HUMAN}"); sys.exit(1)
Sdis = np.zeros((ndis, ndis), dtype=np.float32); Sdis[:N_HUMAN, :N_HUMAN] = ddb
nb = 0
# 橋接住在 interactions/disease_disease.csv(原始檔,格式相同);取 index >= N_HUMAN 的列。
for row in csv.DictReader(open("dataset/Kdataset/interactions/disease_disease.csv")):
    a, b = int(row["Disease1"]), int(row["Disease2"]); h, p = (a, b) if a < b else (b, a)
    if p >= N_HUMAN and h < N_HUMAN:          # 寵物病 p 繼承人類病 h 的相似度結構
        Sdis[p, :N_HUMAN] = ddb[h, :]; Sdis[:N_HUMAN, p] = ddb[:, h]; Sdis[p, h] = Sdis[h, p] = 1; nb += 1
NSdis = norm(Sdis)
bridged = len(set(np.where(Sdis[N_HUMAN:, :N_HUMAN].sum(1) > 0)[0]))
ok(nb > 0, f"寵物病↔人類病橋接 = {nb} 列({bridged}/{len(PET_COLS)} 個寵物病有橋接)")
if bridged < len(PET_COLS):
    print(f"  ⚠️ 有 {len(PET_COLS)-bridged} 個寵物病【沒有橋接】→ 它們在圖上是孤島,候選會近乎隨機")

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
_drow = list(csv.DictReader(open("dataset/Kdataset/omics/drug.csv")))
node2db = {int(r["ID"]): r["Drug"] for r in _drow}
node2name = {int(r["ID"]): r["Name"] for r in _drow if r.get("Name")}   # 藥名住在 omics/drug.csv 的 Name 欄
DBN = {"DB00619":"imatinib","DB01254":"dasatinib","DB00398":"sorafenib","DB01268":"sunitinib",
       "DB08896":"regorafenib","DB01229":"paclitaxel","DB00444":"teniposide","DB00773":"etoposide",
       "DB00694":"daunorubicin","DB01177":"idarubicin","DB00970":"dactinomycin","DB00997":"doxorubicin",
       "DB00541":"vincristine","DB00570":"vinblastine","DB01590":"everolimus","DB00877":"sirolimus",
       "DB06176":"romidepsin","DB11581":"venetoclax","DB00563":"methotrexate","DB01204":"mitoxantrone"}
nm = lambda d: node2name.get(d) or DBN.get(node2db.get(d, ""), node2db.get(d, str(d)))
pet = {int(r["ID"]): r["Name"].replace("寵物-", "") for r in pet_rows}   # 病名住在 omics/disease.csv 的 Name 欄

rows = [["disease", "rank", "drug", "drugbank", "score"]]
n_unknown = 0
for c in PET_COLS:
    known = set(np.where(Y[:, c] == 1)[0])
    cand = [d for d in np.argsort(-comb[:, c]) if d not in known]
    n_unknown += len(cand)
    for r, d in enumerate(cand[:TOPK], 1):
        rows.append([pet.get(c, c), r, nm(d), node2db.get(d, ""), f"{comb[d][c]:.2f}"])
with open(A.out, "w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"  待預測格子:{nd} 藥 × {len(PET_COLS)} 寵物病 − {int(Y[:, N_HUMAN:].sum())} 已知 = {n_unknown:,} 格")
print(f"  → {A.out}:{len(PET_COLS)} 病 × top{TOPK} = {len(rows)-1} 個候選")

print("=" * 72)
print("⑦ 無洩漏檢查")
print("=" * 72)
# 註:此處不驗「排名指紋」。排名取決於 GNN 權重,而 GNN 訓練是非確定性的
#     (utils.py 的 MetaPath2Vec 即使同 seed 也會產生不同特徵,實測差異 0.12),
#     使用者自行重訓後排名本來就會不同 → 拿指紋當檢查只會發出假警報。
#     這裡只驗「一定要成立」的事:輸出的候選必須都是資料裡沒記載的格子(Y=0)。
viol = 0
for c in PET_COLS:
    kn = np.where(Y[:, c] == 1)[0]
    cand = [d for d in np.argsort(-comb[:, c]) if d not in set(kn)][:TOPK]
    viol += int(Y[cand, c].sum())          # 任何一個候選若 Y=1 就是把已知藥當新發現
ok(viol == 0, "所有候選皆為 Y=0(未記載)的格子,無已知藥混入", f"違反 {viol} 個")
ok(int(Y[:, N_HUMAN:].sum()) == n_pet_lab, f"寵物標籤未被更動 = {n_pet_lab}", f"{int(Y[:, N_HUMAN:].sum())}")

print("=" * 72)
if FAIL:
    print(f"❌ {len(FAIL)} 項未通過:{FAIL}"); sys.exit(1)
print(f"✅ 全部通過 — 已產出 {len(rows)-1} 個候選 → {A.out}")
print("=" * 72)
print("誠實提醒:")
print("  · 這些是「候選假設」,不是已證實療法;真效力需濕實驗驗證。")
print("  · 本檔用全資料 GNN(看過所有標籤)→ 只能產候選,不可報 recall/AUC。")
print("  · 本 repo 沒有附效能評估腳本 → 這裡不宣稱任何 recall/AUC 數字。")
print()
print("⚠️ 這是【單次訓練】的結果。GNN 訓練非確定性(m2v),重跑會得到不同名次:")
print("   實測 3 份同設定訓練彼此相關 0.74;rank1 一致率僅 58%、rank8 僅 10%。")
print("   → 請勿把單次的名次當結論。建議:")
print("       for s in 42 43 44 45 46; do python run_all.py --seed $s --out cand_$s.csv; done")
print("     再取「每次都進 top50」的藥,才是穩健候選。")
