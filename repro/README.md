# MRDDA 寵物老藥新用 — 可重現包

把**人類現有藥**重新配對到**寵物疾病**(老藥新用)的完整可重現流程:資料、程式、已訓練輸出、驗證腳本。

> **先講結論(誠實版)**:這個模型是**內插引擎**——它擅長「補齊已知有效藥的同類藥」,不擅長發現全新機制。
> 遮住已知寵物藥後能撈回 **74%**,但其中**人氣先驗就佔 42%**;真正 novel 的重定位只撈回 **14%**(隨機 5.6%)。
> 所有數字皆可用本包重跑核對。

---

## 一、署名與授權(請先讀)

| 項目 | 出處 |
|---|---|
| **底層演算法 + base pipeline** | https://github.com/Ethereal1z/MRDDA(commit `Base: MRDDA + KPet semi-supervised pipeline`,作者 Ethereal1z) |
| **本包新增** | 寵物資料(KPet)、Stack 融合層、驗證/洩漏稽核、四關卡(docking/ADMET)腳本、demo 網頁 |

⚠️ **授權狀態**:上游 `Ethereal1z/MRDDA` **沒有 LICENSE 檔**,因此其程式碼的再散布條件未明確。本包為研究重現用途整理;**若你要商用或再散布,請先向上游作者確認授權**。`model.py`(原始演算法)在本專案中**逐行未改**。

---

## 二、環境

```bash
# 核心(訓練 / 評估)
python>=3.10, numpy, pandas, scikit-learn, torch, dgl

# 四關卡(選用)
pip install rdkit                      # ②可合成 ③結構警訊
pip install admet-ai                   # ③ADMET(注意:會拉自己的 torch)
apt-get install -y autodock-vina openbabel   # ①分子對接
```
> GPU 訓練建議用 `nvcr.io/nvidia/dgl:25.08`(ARM64 上 PyPI 的 dgl 是 CPU-only)。

---

## 三、重現(由快到慢)

### 1️⃣ 產出候選清單(數秒,免重訓)
```bash
python produce_candidates.py          # → stack_candidates.csv(50 病 × top8)
```
用已附的 `resultKPetSup2_par_42/result.csv`(10 折 OOF 輸出)+ Stack 融合。

### 2️⃣ 無偏評估(~15 分)
```bash
python nested_cv.py
```
**預期輸出**:`全部 74±1  重定位 14±4  非人氣 59±1  [總 leak=0]`

### 3️⃣ 洩漏稽核(~2 分)
```bash
python leak_audit_components.py       # 逐成分 recall + mask 硬檢查
python leak_audit_v2.py               # null 負對照(單位/全1/打散)× 兩種排名指標
```
**預期**:mask leak=0;公正中位排名下「單位矩陣 → 0.0%」(證明高分非平手假象)。

### 4️⃣ 四大關卡(需 rdkit / vina / admet-ai)
```bash
python find_mechanism_matches.py      # 全 50 病掃「機制對得上」的候選 → mechanism_matches.json
python hit_pipeline.py                # 肥大細胞瘤 Go/No-Go 表
python teni_test.py                   # teniposide 四關卡
bash dock/dock_run.sh                 # ①Vina 對接(需先抓 PDB 3QX3)
python dock/gate23_cohort.py          # ②SA + ③ADMET(GPU)
```

### 5️⃣ 從頭重訓 GNN(每折約 40 分 × 10 折)
```bash
for k in $(seq 0 9); do
  python train_parallel.py -da KPet --mode fold --fold $k -sp resultKPet_par -se 42
done
python train_parallel.py -da KPet --mode aggregate -sp resultKPet_par   # → result.csv
```

---

## 四、演算法

```
Stack = z(GNN) + 0.7·z(prop) + 0.5·z(NMF)
```
| 成分 | 是什麼 | 訊號來源 |
|---|---|---|
| **z(GNN)** | 凍結的 MRDDA 異質圖 GNN(HeteroGraphConv + GAT + 語義注意力 + 內積解碼) | 圖結構路徑 |
| **z(prop)** | **雙向標籤傳播**(Bi-Random-Walk 式)`F = 0.45·(Nf@F) + 0.45·(F@NSdisᵀ) + 0.1·Y0`,20 次 | 相似鄰居 |
| **z(NMF)** | 非負矩陣分解 rank=50 | 共現規律 |

- `z()` = 逐病(欄)標準化。**融合層不訓練**(純算術)→「不改演算法」約束成立。
- **`drug_sim_fused.csv` 不是純化學相似**:實測反推 **α=0.3** → `0.3·化學結構 + 0.7·機轉(靶點∪通路 Jaccard)`。
- GNN loss:`BCEWithLogitsLoss(pos_weight = 負/正 ≈ 158)`,只對訓練格計算。

---

## 五、資料

| 維度 | 內容 |
|---|---|
| **894 藥** | **全部是人類藥**(DrugBank)——零新增寵物藥,這是「老藥新用」的前提 |
| **504 病** | 454 人類病(MeSH)+ **50 寵物病**(欄 454–503) |
| **2,825 正標籤** | 2,704 人類 + **121 寵物**(`sup_positives.json`:109 標準療法 + 12 重定位) |
| **寵物橋接** | 100 列 = **50 條唯一雙向**「寵物病↔人類病」(寵物病靠此繼承人類病的圖結構) |
| **預測空間** | 894 藥 × 50 寵物病 = **44,700 格**,已知僅 121 格 |

---

## 六、結果(本包可重跑核對)

### recall@50(巢狀 CV,3 seeds,leak=0)
| 指標 | 數字 | 說明 |
|---|---|---|
| 全部 | **74 ± 1 %** | ⚠️ 被人氣先驗與同病橋接墊高 |
| 非人氣藥 | **59 ± 1 %** | 低頻藥 |
| **重定位(真 novel)** | **14 ± 4 %** | **最誠實的硬指標** |
| 人氣先驗對照(全1矩陣) | 42.1 % | ← 74% 有一大半只是這個 |
| 隨機基準 | 5.6 % | 50/894 |

### 逐成分(5 折遮蔽,tie-aware 中位排名)
| 成分 | recall@50 |
|---|---|
| 只有 GNN(OOF) | 58.7 % |
| 只有 prop | 55.4 % |
| 只有 NMF | 39.7 % |
| prop + NMF | 64.5 % |
| **完整 Stack** | **75.2 %** |

### 人類側 GNN(10 折 OOF)
`AUC 0.905` / **`AUPR 0.362`** ← 正例僅 0.6%,**AUPR 才是誠實指標**(Accuracy 0.991 無意義)

---

## 七、洩漏驗證(全部可重跑)

| 成分 | 怎麼驗 | 結果 |
|---|---|---|
| GNN | `train_parallel.py` 每折 `remove_graph(g, test_pos_id)` 把測試邊從圖移除;每折只存自己的測試格;`aggregate()` 逐格由「它被 held-out 那折」填 | ✅ **真 out-of-fold** |
| prop / NMF | 評估時 `Y0[測試]=0` + 硬檢查 | ✅ **leak = 0**(3 seeds) |
| 指標本身 | 負對照:單位矩陣 | ✅ 公正中位排名 → **0%** |

> ⚠️ **踩過的坑**:`leak_check.py` 用**樂觀 tie 排名**`(col>col[d]).sum()+1`,會讓單位矩陣得 95% — 那是**指標假象不是洩漏**。請用 `nested_cv.py` 的中位排名。

---

## 八、誠實界線(務必一起讀)

1. **模型是內插,不是機制發現**——它重組「已知的相似藥 / 相似病 / 共現」。
2. **74% 會誤導**:人氣先驗佔 42%;寵物病只透過**同病橋接**繼承人類病結構,所以「治人類同一種病的藥」自然對寵物版噴高分——合法,但接近**跨物種搬運**。
3. **真 novel 重定位僅 14%**——內插的結構性天花板。
4. **候選是假設,不是已證實**。例:犬淋巴瘤的 teniposide/daunorubicin/idarubicin 對接 TOP2 都 < −10 kcal/mol(對照已知有效的 etoposide −9.78),但**只有 idarubicin 有直接犬試驗**(31 犬 Phase I)。
5. **關卡①只驗一條機制會誤殺**:dactinomycin 對接 **+13.84**(塞不進 TOP2)被打槍,但它**是臨床在用的犬淋巴瘤搶救藥**(49 例/DMAC 72%)——它走 DNA 嵌入這條別的路。

---

## 九、檔案地圖

```
repro/
├── model.py load_data.py utils.py args.py      # 原始 MRDDA(model.py 零修改)
├── train_parallel.py                           # 10 折 OOF 訓練
├── produce_candidates.py                       # Stack → 候選清單
├── nested_cv.py                                # 無偏評估
├── build_mech_sim.py                           # 融合相似度(α=0.3)
├── leak_check.py leak_audit_components.py leak_audit_v2.py   # 洩漏稽核
├── verify_pipeline.py hit_pipeline.py teni_test.py find_mechanism_matches.py  # 四關卡
├── dock/                                       # ①Vina 對接 + ③ADMET
├── sup_positives.json                          # 121 寵物標籤
├── resultKPetSup2_par_42/result.csv            # 已訓練 GNN 輸出(10折 OOF,免重訓)
└── dataset/{Kdataset,KPet,Bdataset}/           # 全部訓練資料
```

線上 Demo:[證據圖](https://odek53r.github.io/AI-Drug/cohort_graph.html) · [RAG Demo v2](https://odek53r.github.io/AI-Drug/rag_demo_v2.html)
