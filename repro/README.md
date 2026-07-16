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

### A. 用附的現成 GNN 產候選 — 不需要 GPU / DGL
```bash
pip install numpy pandas scikit-learn      # 這樣就夠了
python run_all.py --skip-train             # 4 秒
```
> `--skip-train` 只讀現成的 GNN 輸出 + 純 numpy,**不需要 torch/dgl/GPU**。
> 但那是重跑我們既有模型的答案,**不是你自己訓練的**。

### B. 自己訓練(`run_all.py` 預設)— 需要 GPU + DGL
**建議直接用容器**(ARM64 上 PyPI 的 dgl 是 CPU-only,自己裝很痛苦):
```bash
docker run --gpus all -it -v $PWD:/workspace nvcr.io/nvidia/dgl:25.08 bash
```

**本包實測通過的版本組合**(DGL 對 torch 版本很挑,這組確認可跑):
| 套件 | 版本 |
|---|---|
| python | 3.12.3 |
| torch | 2.13.0+cu130 |
| **dgl** | **2.5** |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| CUDA | 13.0(實測於 NVIDIA GB10) |

### 四關卡(選用)
```bash
pip install rdkit                            # ②可合成 ③結構警訊
pip install admet-ai                         # ③ADMET ⚠️ 會拉自己的 torch,可能覆蓋既有版本
apt-get install -y autodock-vina openbabel   # ①分子對接
```
> ⚠️ `admet-ai` 會安裝自己的 torch,**可能覆蓋掉容器裡調好的 torch**。建議另開容器/venv 跑 ADMET。
> 若 `import admet_ai` 報 `operator torchvision::nms does not exist`,執行 `pip uninstall -y torchvision` 即可(ADMET 用不到 torchvision,是 torchmetrics 的影像模組硬拉的)。

---

## 三、重現(由快到慢)

### 0️⃣ 🚀 一鍵跑完整條流水線

```bash
python run_all.py                  # 訓練 GNN(實測 14.7 分,需 GPU)→ 產候選
python run_all.py --skip-train     # 跳過訓練,用附的現成 GNN(4 秒,純 CPU)
python run_all.py --seed 43 --out cand_43.csv    # 換 seed
```
流程:資料完整性檢查 → **訓練/載入 GNN** → prop → NMF → Stack → 產候選 → 無洩漏檢查。
產出 `stack_candidates.csv`(50 病 × **top50** = 2500 候選)。

### ⚠️⚠️ 跑之前必讀:每次訓練的結果都不一樣

**GNN 訓練是非確定性的**,而且源頭在原始碼:`utils.py` 的 **MetaPath2Vec 即使給同一個 seed,
每次產生的節點特徵都不同**(實測最大差異 0.12),整個模型跟著變。

實測(3 份**同設定**訓練):

| 量測 | 結果 |
|---|---|
| 三份模型彼此相關係數 | **僅 0.74** |
| dactinomycin 對犬淋巴瘤的排名 | **#5 / #30 / #7** 之間跳 |
| 跨 seed 名次一致率 | rank1 **58%**、rank5 14%、rank8 **10%** |
| top50 清單重疊 | 82% |

**→ 不要把單次訓練的名次當結論。** 正確用法:
```bash
for s in 42 43 44 45 46; do python run_all.py --seed $s --out cand_$s.csv; done
# 再取「每次都進 top50」的藥 = 穩健候選
```
> 💡 單跑一份只吃 34% GPU;**平行跑 3 份可吃到 93%**,5 個 seed 從 77 分縮到約 40 分:
> ```bash
> for s in 42 43 44; do python run_all.py --seed $s --out cand_$s.csv & done; wait
> ```

> 📌 `--skip-train` 用的是本包附的 `resultKPetFull_42/result_full.csv`。
> 那**只是重跑我們既有模型的答案,不是你自己訓練出來的結果**——腳本也會這樣提醒你。

### ⚠️ 兩個模型檔,兩種用途(**不可混用**)

| 檔案 | 訓練方式 | 只能用來 | 已知有效藥的平均分 |
|---|---|---|---|
| `resultKPetSup2_par_42/result.csv` | 10 折 **OOF** | **報效能**(leak-free) | 0.621(真預測) |
| `resultKPetFull_42/result_full.csv` | **全資料**(2,825 標籤全用) | **產候選** | **0.999**(≈背下來了) |

**證據**:全資料版把已知標籤背到 **0.999** → 它是在默寫答案,**絕不可拿來報 recall/AUC**;
但正因為它用光所有已知標籤,**產候選時更準**。權重 β=0.7/γ=0.5 **鎖死**(由 OOF 版誠實選出,不用全資料版重搜,否則是拿洩漏分數挑參數)。


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

### 5️⃣ 從頭重訓 GNN

**A. 全資料版(產候選用)— 約 15 分**
```bash
python train_parallel.py -da KPet --mode full -sp resultKPetFull -se 42   # → result_full.csv
```

**B. 10 折 OOF 版(報效能用)— 每折約 15 分 × 10**
```bash
for k in $(seq 0 9); do
  python train_parallel.py -da KPet --mode fold --fold $k -sp resultKPet_par -se 42
done
python train_parallel.py -da KPet --mode aggregate -sp resultKPet_par     # → result.csv
```

> ⚠️ 重訓後**候選會與本包附的不同**(見前面「每次訓練的結果都不一樣」:三份同設定訓練彼此相關僅 0.74)。
> 這不是錯誤,是 MetaPath2Vec 的非確定性。想要穩健結論請跑多個 seed 取共識。
> 📌 `--mode full` 的效能備註:原始 per-fold 迴圈有兩處純浪費——用 45 萬元素的 Python tuple 做索引
> (313ms × 4/epoch)、以及算完就丟的 AUPR(64ms)。全資料訓練時 mask=全部格子,
> `score[mask].flatten()` 與 `score.flatten()` **逐位元相同**(已驗證),故直接 flatten。
> 結果不變,速度約 8×,GPU 使用率 3% → 66%。`model.py` 零修改。

---

## 三之二、想換成自己的資料集?

腳本的維度**全部從資料讀**(不寫死 894/504/454),所以可以增刪藥/病/標籤。實測結果:

| 你想做的 | 可以嗎 | 要注意 |
|---|---|---|
| **新增寵物疾病** | ✅ 最單純 | 4 個檔要一起改(見下),然後**必須重訓** |
| **新增/刪除標籤** | ✅ | 改 `KPet_baseline.csv` + `sup_positives.json`,**必須重訓** |
| **新增/刪除藥物** | ⚠️ 可以但麻煩 | 相似度矩陣要重建,見下 |
| **刪除疾病** | ⚠️ 可以但麻煩 | 所有關聯檔指向它的列都要一起刪 |

> ⚠️ **只要動過資料,`--skip-train` 就不能用**——附的 GNN 是舊維度,腳本會擋下並要你重訓。

### 新增一個寵物疾病(最常見)
```
1. dataset/KPet/KPet_baseline.csv        加一欄(0/1),1 = 已知有效
2. dataset/KPet/KPet_pet_diseases.csv    加一列(kpet_index 接續、name、species)
3. dataset/KPet/KPet_pet_disease_disease.csv  加【雙向兩列】橋接到對應的人類病
4. sup_positives.json                    加該病的已知用藥標籤
5. python run_all.py                     重訓(~15 分)
```
> 💡 **第 3 步別漏**:沒橋接的寵物病在圖上是**孤島**,候選會近乎隨機。腳本會警告你。

### 新增/刪除藥物
建圖的藥物節點數是由**相似度矩陣的維度**決定的,不是標籤矩陣。所以要同步重建:
```
dataset/Kdataset/drug_drug_baseline.csv   ← 必須是 藥數×藥數
dataset/KPet/drug_sim_fused.csv           ← python build_mech_sim.py 0.3
```

### 刪除疾病/藥物
DGL 依**關聯檔裡出現過的最大 index** 建節點。若還有檔案指向已刪的 index,
節點數就會跟標籤矩陣對不上。要一併清理:
`Kdataset.csv` / `drug_protein.csv` / `pathway_disease.csv` / `KPet_pet_disease_disease.csv`

> ✅ 上述每種情況腳本都會在**開跑前**檢查並明確告訴你要修哪個檔,
> 不會讓你對著 PyTorch 的 `Target size must be the same as input size` 或
> DGL 的 `Expect number of features to match number of nodes` 猜半天。

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

> ⚠️ **踩過的坑(留作警惕)**:早期的稽核腳本用**樂觀 tie 排名** `(col>col[d]).sum()+1`,
> 會把測試藥擺在平手塊最前面 → 單位矩陣(等於完全不傳播)竟得到 **95%**,看起來像嚴重洩漏。
> 那是**指標假象,不是洩漏**。換成公正的中位排名 `(col>col[d]).sum()+((col==col[d]).sum()+1)/2` 後,單位矩陣 → **0%**。
> 該腳本已移除;`nested_cv.py` 與 `leak_audit_v2.py` 都用中位排名。

---

## 八、誠實界線(務必一起讀)

1. **模型是內插,不是機制發現**——它重組「已知的相似藥 / 相似病 / 共現」。
2. **74% 會誤導**:人氣先驗佔 42%;寵物病只透過**同病橋接**繼承人類病結構,所以「治人類同一種病的藥」自然對寵物版噴高分——合法,但接近**跨物種搬運**。
3. **真 novel 重定位僅 14%**——內插的結構性天花板。
4. **候選是假設,不是已證實**。例:犬淋巴瘤的 teniposide/daunorubicin/idarubicin 對接 TOP2 都 < −10 kcal/mol(對照已知有效的 etoposide −9.78),但**只有 idarubicin 有直接犬試驗**(31 犬 Phase I)。
5. **關卡①只驗一條機制會誤殺**:dactinomycin 對接 **+13.84**(塞不進 TOP2)被打槍,但它**是臨床在用的犬淋巴瘤搶救藥**(49 例/DMAC 72%)——它走 DNA 嵌入這條別的路。

---

## 九、檔案地圖 — 哪些該看,哪些不用

**只想產候選?你只需要 `run_all.py`。** 其他都是選用的。

### 🟢 核心(跑 `run_all.py` 必要,共 6 支)
| 檔案 | 角色 |
|---|---|
| `run_all.py` | **入口**——訓練 → prop → NMF → Stack → 產候選 |
| `train_parallel.py` | GNN 訓練(`--mode full` 全資料 / `--mode fold` 10 折 OOF) |
| `model.py` | **原始 MRDDA 演算法,零修改** |
| `load_data.py` `utils.py` `args.py` | 建圖 / MetaPath2Vec / 參數 |

```
run_all.py ──(subprocess)──> train_parallel.py ──(import)──> args, load_data, model, utils
```

### 🔵 誠實評估(產出我們對外宣稱的數字)
| 檔案 | 產出 |
|---|---|
| `nested_cv.py` | **74±1 / 59±1 / 14±4**(巢狀 CV,leak=0)← README 引用的數字來源 |
| `leak_audit_v2.py` | null 負對照(單位/全1/打散)× 兩種排名指標 |
| `leak_audit_components.py` | 各成分貢獻(GNN/prop/NMF 各撈回多少) |

### 🟡 四關卡分析(需 RDKit / AutoDock Vina / ADMET-AI)
| 檔案 | 用途 |
|---|---|
| `hit_pipeline.py` | 四關卡 Go/No-Go 表(以肥大細胞瘤為例) |
| `teni_test.py` | teniposide 走完四關 |
| `find_mechanism_matches.py` | 全 50 病掃「打中共同機制靶點」的候選 |
| `verify_pipeline.py` | 獨立驗證上面幾支的宣稱(14 項 PASS/FAIL) |
| `dock/` | ①Vina 對接 + ③ADMET 的腳本 |

### ⚪ 資料生成(已附成品,通常不用跑)
| 檔案 | 產出 |
|---|---|
| `build_mech_sim.py` | `drug_sim_fused.csv`(α=0.3:30% 化學 + 70% 機轉) |

### 資料與模型
```
sup_positives.json                    121 個寵物標籤
resultKPetSup2_par_42/result.csv      10 折 OOF GNN  → 只給評估用
resultKPetFull_42/result_full.csv     全資料 GNN     → 只給產候選用(--skip-train 時)
dataset/{Kdataset,KPet,Bdataset}/     全部訓練資料
stack_candidates.csv                  範例輸出(2500 候選)
```

> 🗑️ **已移除**:`produce_candidates.py`(被 `run_all.py` 取代,且會覆蓋輸出)、
> `leak_check.py`(用樂觀 tie 排名,會讓單位矩陣得到 95% 的假象;`leak_audit_v2.py` 已涵蓋且用公正的中位排名)。

線上 Demo:[證據圖](https://odek53r.github.io/AI-Drug/cohort_graph.html) · [RAG Demo v2](https://odek53r.github.io/AI-Drug/rag_demo_v2.html)
