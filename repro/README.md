# MRDDA 寵物老藥新用 — 可重現包

把**人類現有藥**重新配對到**寵物疾病**(老藥新用)的完整可重現流程:資料 + 程式。

> **先講結論(誠實版)**:這個模型是**內插引擎** —— 它擅長「補齊已知有效藥的同類藥」,
> 不擅長發現全新機制。寵物疾病沒有自己的藥/靶點/通路資料,**100% 靠「橋接到人類同名病」
> 繼承圖結構**,所以撈回的多半是該病的標準藥。產出是**候選假設,不是已證實療法**。
>
> ⚠️ **本包不含效能評估腳本,因此不宣稱任何 recall/AUC 數字**(見「四、誠實界線」)。

---

## 一、署名與授權(請先讀)

| 項目 | 出處 |
|---|---|
| **底層演算法 + base pipeline** | [github.com/Ethereal1z/MRDDA](https://github.com/Ethereal1z/MRDDA)(作者 Ethereal1z)· *J Transl Med* 2025, [10.1186/s12967-025-06783-x](https://doi.org/10.1186/s12967-025-06783-x) |
| **Kdataset(人類資料)** | [github.com/gu-yaowen/REDDA](https://github.com/gu-yaowen/REDDA)(Gu et al., *Comput Biol Med* 2022;150:106127)· MRDDA 原封不動複製(md5 相同) |
| **本包新增** | 寵物資料(48 病 / 158 標籤 / 48 條橋接)、人類 TNBC(23 標籤)、寵物疾病相似度(Wang's method)、機轉融合相似度、Stack 融合層。每筆的來源見 `data_provenance.csv` / `sources.csv`,版本沿革見 `DATA_VERSIONS.md` |

⚠️ **授權狀態**:上游 `Ethereal1z/MRDDA` **沒有 LICENSE 檔**,因此其程式碼的再散布條件未明確。
本包為研究重現用途整理;**若你要商用或再散布,請先向上游作者確認授權**。
`model.py`(原始演算法)在本專案中**逐行未改**。

---

## 二、環境

**GPU 不是必要的** —— `run_all.py` 會自動偵測,沒有 GPU 就退回 CPU:

| | 4000 epochs | |
|---|---|---|
| GPU | **~15 分** | ~300 epoch/分 |
| CPU | **~2 小時** | ~33 epoch/分。實測 loss/AUC 與 GPU 一致 |

### ✅ 已驗證的環境(唯一實測過的)

```
nvcr.io/nvidia/dgl:25.08-py3
  python 3.12.3 · torch 2.13.0+cu130 · dgl 2.5 · rdkit 2026.03.3 · aarch64
```

```bash
docker run -d --name mrdda-gpu --gpus all \
  -v /path/to/repro:/workspace/MRDDA \
  nvcr.io/nvidia/dgl:25.08-py3 sleep infinity

docker exec mrdda-gpu bash -lc 'cd /workspace/MRDDA && python3 run_all.py'
```

> ⚠️ **為什麼要 docker**:PyPI 上的 `dgl` 是 **CPU-only**,ARM64 更是沒有 CUDA wheel。
> 想吃 GPU 就得用 NVIDIA 的映像。x86_64 或許能自行 pip 裝 CUDA 版 dgl,
> **但我們沒測過** —— 上面那組是唯一實測過的組合。
>
> ⚠️ `utils.py` 為新版 numpy 打過補丁(`np.mat` → `np.asmatrix`),所以
> **上游 README 寫的 python 3.10 / torch 2.5.1 我們沒測過**,不保證可跑。

`rdkit` 只有 `build_mech_sim.py`(增刪藥物時重建相似度)才需要。

> 📌 **本包不隨附訓練好的模型**,每次都從頭訓練。
> 理由:隨附模型會讓人拿到別人訓練的答案,卻以為是自己跑出來的。

---

## 三、重現

```bash
python run_all.py                  # 訓練 + 產候選(GPU ~15 分 / CPU ~2 小時)
python run_all.py --seed 43        # 換 seed
python run_all.py --cpu            # 強制 CPU(沒 GPU 時會自動啟用,不必手動加)
```

輸出 `stack_candidates.csv`:48 個寵物病 × top50 候選藥。

```csv
disease,rank,drug,drugbank,score
Precursor Cell Lymphoblastic Leukemia-Lymphoma,7,teniposide,DB00444,5.77
```

### ⚠️⚠️ 跑之前必讀:每次訓練的結果都不一樣

GNN 訓練是**非確定性**的 —— `utils.py` 的 MetaPath2Vec 即使給同一個 seed,
每次產生的節點特徵都不同(實測最大差異 0.12),整個模型跟著變。

實測(3 份同設定訓練):彼此相關係數僅 **0.74**;
dactinomycin 對犬淋巴瘤的排名在 **#5 / #30 / #7** 之間跳。
跨 seed 名次一致率:rank1 僅 **58%**、rank8 僅 **10%**;top50 清單重疊 **77~82%**。

→ **不要拿單次結果當結論**:
```bash
for s in 42 43 44 45 46; do python run_all.py --seed $s --out cand_$s.csv; done
# 再取「每次都進 top50」的藥 = 穩健候選
```

### 流程

```
① 資料完整性檢查
② 全資料 GNN 訓練 → resultKPetFull_<seed>/result_full.csv   ← 唯一有訓練參數的部分
③ prop 雙向標籤傳播(秒級現算,零參數)
④ NMF 非負矩陣分解(秒級現算,零參數)
⑤ Stack = z(GNN) + 0.7·z(prop) + 0.5·z(NMF)                 ← 融合層不訓練,只算術
⑥ 對「還沒有藥-病關聯」的格子排名,取 top50
⑦ 無洩漏檢查(候選必須全是未記載的格子)
```

`β=0.7 / γ=0.5` 鎖死 —— 它們是用誠實的 OOF 版選出來的;
拿全資料版(看過所有標籤)重搜等於用洩漏的分數挑參數。

---

## 四、⚠️ 誠實界線

**① 產出的是「候選假設」,不是已證實療法。** 真效力需濕實驗驗證。

**② 本包沒有附效能評估腳本,所以不宣稱任何 recall/AUC 數字。**
`run_all.py` 用的是**全資料 GNN(看過所有標籤)** —— 它只能產候選;拿它報效能一定是洩漏的。
要評估必須自行實作 leak-free 的 OOF(遮住測試邊 + `remove_graph()` + `Y0[測試]=0`)。

> 📌 本包**先前**版本附過 `nested_cv.py` / `leak_audit_v2.py` / 四關卡腳本,報過
> recall@50 = 全部 74±1 / 非人氣 59±1 / 真 novel 重定位 14±4(人氣先驗對照 42.1%、隨機 5.6%)。
> 那些腳本已隨資料合併一併移除(它們讀的 `KPet_baseline.csv` / `sup_positives.json` /
> `drug_sim_fused.csv` 都已併入原始檔案),**因此那些數字目前無法用本包重現**。
> 留著別人驗證不了的數字不誠實,故一併撤下。
> 歷史版本仍可於 git 記錄中取得。

**③ 這個方法本質是內插(重組已知關聯),不是機制發現。**
寵物病只透過**同病橋接**繼承人類病結構,所以「治人類同一種病的藥」自然對寵物版噴高分
—— 合法,但接近**跨物種搬運**,不是新機制的發現。

**④ 每次訓練結果不同**(見上)。

---

## 五、資料

`dataset/Kdataset/` —— **只有這一份**,18 個檔,檔名清單與上游 REDDA/MRDDA 一字不差。
寵物資料已完全併入,沒有任何附加檔案。

| 維度 | 內容 |
|---|---|
| **894 藥** | **全部是人類藥**(DrugBank)—— 零新增寵物藥,這是「老藥新用」的前提 |
| **503 病** | 455 人類病(MeSH,含 v2 新增的 TNBC)+ **48 寵物病**(欄 455–502) |
| **2,885 正標籤** | 2,727 人類 + **158 寵物**(`associations/Kdataset.csv` 的 `Evidence` 欄記來源) |
| **寵物橋接** | 96 列 = **48 條唯一雙向**「寵物病↔人類病」(`interactions/disease_disease.csv` 裡 `Sim == 1.0` 的列) |
| **疾病相似度** | 同檔的 `Sim < 1` 是 **Wang's method** 算出的寵物↔人類相似度,給 `run_all.py` 的 prop 當**權重**,**不是圖上的邊** |
| **預測空間** | 894 藥 × 48 寵物病 − 158 已知 = **42,754 格** |

**要換成自己的資料集?** 見 **[EDIT_DATASET.md](EDIT_DATASET.md)** —— 18 個檔逐一說明,含實測過的完整步驟。

最常見的兩件事:
```bash
# 加一顆藥:填 SMILES 就好,相似度自動算(1.2 秒)
echo '894,DB99999,CC(=O)Oc1ccccc1C(=O)O,aspirin' >> dataset/Kdataset/omics/drug.csv
# ...(Kdataset_baseline.csv 加一列,見 EDIT_DATASET.md)
python build_mech_sim.py 0.3
python run_all.py

# 加一個寵物病:關鍵是橋接到對應的人類病
echo '503,D054198,寵物-新病名' >> dataset/Kdataset/omics/disease.csv
# Sim=1.0 那兩列是【橋接邊】(必要);其餘人類病的 Wang 相似度是 prop 的權重(可選)
printf '503,16,1.0\n16,503,1.0\n' >> dataset/Kdataset/interactions/disease_disease.csv
# ...(Kdataset_baseline.csv 加一欄,見 EDIT_DATASET.md)
python run_all.py
```

> ✅ 只要動過資料,`run_all.py` 會在**開跑前**檢查並明確告訴你要修哪個檔,
> 不會讓你對著 PyTorch 的 `Target size must be the same as input size` 或
> DGL 的 `Expect number of features to match number of nodes` 猜半天。

---

## 六、演算法

```
MRDDA(model.py,逐行未改)
  HeteroGraphConv + GATConv + SemanticAttention + InnerProductDecoder
  BCEWithLogitsLoss(pos_weight = 負/正 ≈ 158.5)· Adam + CyclicLR
  seed 42 · epoch 4000 · patience 300 · K=128 · lr=0.01 · dropout=0.4
```

**Stack(本包新增,融合層無任何訓練參數)**

| 成分 | 是什麼 | 訊號來源 |
|---|---|---|
| **z(GNN)** | MRDDA 異質圖 GNN | 圖結構路徑 |
| **z(prop)** | **雙向標籤傳播**(Bi-Random-Walk 式)`F = 0.45·(Nf@F) + 0.45·(F@NSdisᵀ) + 0.1·Y0`,20 次 | 相似鄰居 |
| **z(NMF)** | 非負矩陣分解 rank=50, 80 次迭代, RandomState(0) | 共現規律 |

`z()` = 逐病(欄)標準化。**融合層不訓練**(純算術)→「不改演算法」約束成立。

**藥-藥相似度(本包新增,`build_mech_sim.py`)**
```
drug_drug_baseline.csv = 0.3·化學(ECFP4 Tanimoto)+ 0.7·機轉(靶點∪通路 Jaccard)
```
從 `omics/drug.csv` 的 SMILES 現算(RDKit Morgan r=2, 2048 bits)。
**已驗證**:ECFP4 部分重現 REDDA 原始 `drug_drug_baseline.csv` 的 798,342 格,
精確吻合 **100.0000%**、最大差異 **0.00000000**。

---

## 七、已知的上游問題(我們維持原狀不修,以保持與上游一致)

| 問題 | 實測 |
|---|---|
| `disease_disease_baseline.csv` 的列序與 `omics/disease.csv` 的 ID 對不上 | 454 列裡只有 **2** 列沒位移 |
| `load_data.py` 的 `argpartition(row, 15)` 沒取到 top-15 | 平均只命中真正 top-15 中的 **2.00** 個 |
| `disease_sim = disease_disease` 是參考不是複製 → 節點特徵被塞入假的 1.0 | 疾病 **6,745** 格、藥物 **13,371** 格 |
| `interactions/drug_drug.csv` 有 13 對立體異構物寫成 0.0(正確是 1.0) | 因此 `build_mech_sim.py` 改從 SMILES 現算 |

> 前三項是上游 REDDA → MRDDA 一路複製下來的。MRDDA 論文 Table 1 的邊數
> (Disease-disease **7,199** / Drug-drug **14,291**)**只有用那段有 bug 的程式碼才能精確重現**
> —— 正確的 top-15 寫法給的是 6,810 / 13,410,對不上。可自行驗證。
>
> 這代表圖上的「疾病相似度」訊號基本上是噪音。**我們的寵物橋接不吃這個矩陣**
> (寵物病特徵 = 0,只有 1 條橋接邊),所以核心路徑不受影響。
>
> 📌 **prop 的部分已經避開它**:先前 prop 是把橋接人類病的相似度**整列複製**給寵物病,
> 等於把上面第一項的位移一起繼承了。現在寵物病的相似度改用 **Wang's method 現算**
> (`interactions/disease_disease.csv` 的 `Sim` 欄)。實測差別:
> `寵物-Hypertension` 的最近鄰從「IgA 血管炎 0.26」變成「原發性高血壓 0.72 / 肺高壓 0.60」。
> ⚠️ 但**候選排名幾乎沒變**(top50 重疊 76.2%,而同設定重跑的基準就是 77~82%)——
> 落在雜訊範圍內,不宣稱有改善。詳見 `DATA_VERSIONS.md` 的 data-v4。

---

## 八、檔案地圖

**只想產候選?你只需要 `python run_all.py`。**

| 檔案 | 角色 |
|---|---|
| `run_all.py` | **入口** —— 訓練 → prop → NMF → Stack → 產候選 |
| `train_parallel.py` | GNN 訓練(`--mode full` 全資料 / `--mode fold` 10 折 OOF) |
| `model.py` | **原始 MRDDA 演算法,零修改** |
| `load_data.py` `utils.py` `args.py` | 建圖 / MetaPath2Vec / 參數 |
| `main.py` | 上游原始入口(10 折 CV) |
| `build_mech_sim.py` | 重建 `drug_drug_baseline.csv`(增刪藥物後要跑) |
| `dataset/Kdataset/` | 全部資料(18 個檔) |
| `stack_candidates.csv` | 範例輸出(2,400 候選) |
| `EDIT_DATASET.md` | 增刪藥物/疾病的逐檔說明 |
| `MECH_PROGRESS.md` | 機轉融合的開發紀錄(含當初高報後自行修正的過程) |

**資料來源與沿革**(要查「這筆標籤哪來的」就看這幾份)

| 檔案 | 內容 |
|---|---|
| `DATA_VERSIONS.md` | 資料版本沿革 v1→v4,每版改了什麼、為什麼、驗證結果 |
| `sources.csv` | 25 個資料來源 × 14 欄:endpoint / 版本 / 授權 / 已知問題 / 驗證方式與結果 |
| `data_provenance.csv` | 282 列 × 24 欄:逐列掃訓練資料,含 pmid · doi · nct_id · chembl_id · mesh_id · drugbank_id · source_url · query_used |
| `pet_labels_provenance.csv` | 原始 121 筆寵物標籤的文獻查證(含判定「移除」的理由) |
| `pet_bridges_provenance.csv` | 病-病橋接的跨物種查證(哪些是 ANALOGOUS 而非等同) |
| `pet_drugs_fda_status.csv` | FDA Green Book 的動物用藥核准狀態 |
| `pet_labels_expansion.csv` `tnbc_drug_disease.csv` | v3 / v2 新增標籤的逐筆來源 |
| `REMOVED_PET_DATA.md` | 稽核後**刪掉**哪些資料、依據什麼 |

```
run_all.py ──(subprocess)──> train_parallel.py ──(import)──> args, load_data, model, utils
```

---

線上 Demo:[證據圖](https://odek53r.github.io/AI-Drug/cohort_graph.html) · [RAG Demo v2](https://odek53r.github.io/AI-Drug/rag_demo_v2.html)
