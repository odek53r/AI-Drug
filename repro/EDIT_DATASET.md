# 資料集逐檔說明 — 每一份要怎麼調整

> 本文每個格式與數字都由實際檔案查證。改完務必重訓:`python run_all.py`

寵物資料**已完全併入原始的 18 個檔**,沒有 `dataset/KPet/`、沒有 `sup_positives.json`、
沒有任何附加檔案。整個專案只有一份資料:`dataset/Kdataset/`。

---

## 開始前:三個必懂的規則

**① 全部用整數 index 互相參照**
藥 `0~893`、病 `0~503`(人類 `0~453` + 寵物 `454~503`)。
`protein`(18,877)/ `gene`(20,561)/ `pathway`(314)**各自獨立編號**。

**② 人類/寵物的分界線 = `disease_disease_baseline.csv` 的維度**
```
人類病數 N_HUMAN = disease_disease_baseline.csv 的列數(454)
疾病總數        = omics/disease.csv 的列數(504)
寵物病          = index >= N_HUMAN 的那些
```
不是看 `omics/disease.csv` —— 那裡面人類和寵物都有。

**③ 刪除會讓 index 全部位移** → 所有含 index 的檔都要重編。
> 💡 **能不刪就不刪**。想棄用某藥/病,**把標籤清成 0** 就好(見 F 節),**零 index 重編**,實測可行。

---

# A. `omics/` — 節點定義

## `drug.csv` — 藥物節點(894 列)
```csv
ID,Drug,SMILES,Name
0,DB00860,C[C@]12C[C@H](O)[C@H]3[C@@H](CCC4=CC(=O)...,prednisolone
```
| 欄 | 意義 | 增刪時 |
|---|---|---|
| `ID` | **藥物 index(0 起連續,不可有洞)** | 加:接續最後一個(`894`);刪:後面全部 −1 |
| `Drug` | DrugBank ID | 自由填 |
| `SMILES` | 分子結構式 | **必填** —— `build_mech_sim.py` 用它算 ECFP4 化學相似度 |
| `Name` | 藥名(顯示用,可空) | 輸出的 CSV 會用它;空的話顯示 DrugBank ID |

> `Name` 欄不是我們發明的格式 —— 上游 REDDA 的 `Bdataset/omics/drug.csv` 本來就是
> `Name,Drug,SMILES`(該資料集與本專案無關,已移除)。
> 目前 894 個藥裡有 79 個有名字(候選 CSV 的藥名覆蓋率約 27%)。

## `disease.csv` — 疾病節點(504 列 = 人類 454 + 寵物 50)
```csv
ID,Disease,Name
1,D006973,Hypertension              ← 人類病
454,D006973,寵物-Hypertension       ← 寵物病:同一個 MeSH 概念,不同物種
```
| 欄 | 意義 |
|---|---|
| `ID` | 疾病 index。`0~453` = 人類,`454~503` = 寵物 |
| `Disease` | MeSH ID。寵物列填**它橋接到的人類病的 MeSH**(同一個醫學概念)|
| `Name` | 病名。寵物慣例前綴 `寵物-`,`run_all.py` 輸出時會去掉前綴 |

> ⚠️ **人類列(0~453)一般不要動** —— 動了所有寵物病 index 都要跟著位移,
> 而且 `disease_disease_baseline.csv`(454×454)也要重建。
> **加寵物病**:接在最後(`504`),見下面 C 節的橋接。

## `protein.csv`(18,877)/ `gene.csv`(20,561)/ `pathway.csv`(314)
```csv
ID,Protein,Sequence          ID,Gene         ID,Pathway
0,Q16851,MSRFVQDLSKAMSQ...   0,Entrez1       0,hsa05134
```
> 增刪藥/病時**完全不用動這三個**。它們有自己獨立的編號體系。

---

# B. `associations/` — 關聯邊(機制路徑)

這五個檔串起 **藥 → 蛋白 → 基因 → 通路 → 病** 的機制路徑。

## `Kdataset.csv` — 藥治病(2,825 列)⭐ 訓練正標籤
```csv
Drug,Disease,Evidence
830,33,                    ← 人類關聯(2,704 筆),Evidence 空白
390,488,標準               ← 寵物關聯(121 筆),Evidence 有值
```
| 欄 | 意義 |
|---|---|
| `Drug` / `Disease` | **index**(不是名字)|
| `Evidence` | 只有寵物標籤才填:`標準` / `標準CHOP` / `重定位` …。**訓練不讀這欄**,是給人看的來源註記 |

| 增刪時 |
|---|
| **加藥**:選填。加了 = 告訴模型這個新藥治哪些**人類病** |
| **加寵物標籤**:加一列 `藥index,寵物病index,證據`,**同時** `Kdataset_baseline.csv` 對應格要改成 `1.0` |
| **刪藥/病**:必須刪掉指向它的列 + **重編**兩欄的 index |

## `drug_protein.csv` — 藥打哪些靶點(4,397 列)
```csv
Drug,Protein
0,7635          ← 藥#0 打 蛋白#7635
```
> ❗ **protein 不會自動關聯,要自己加。**
> **不加不會壞**(894 個藥裡本來就有 66 個沒靶點),但新藥會**只剩化學相似度**這一條路。

**怎麼加**:查藥的 UniProt accession → 在 `omics/protein.csv` 找它的 `ID` → 填進來。
```bash
grep -n "P11388" dataset/Kdataset/omics/protein.csv    # 找 TOP2A 的 index
# 11237,P11388,MEVSSPLIS...   → 在 drug_protein.csv 加一列 894,11237
```

### ⚠️ 實測:光加藥物但什麼關聯都不給,它進不了任何候選

實際加了一顆 aspirin(`DB99999`)、跑完整訓練(895 藥 × 504 病 = 44,629 格待預測):

```
新藥 DB99999 進 top50 的次數:0   ← 一次都沒有

它在圖上的連結:
  靶點數        : 0        ← 沒加 drug_protein.csv
  治哪些人類病  : 0        ← 沒加 Kdataset.csv
  正標籤數      : 0        ← 標籤矩陣整列都是 0
  融合相似度    : 平均 0.120  ← 唯一的連結來源
```

**對照 teniposide**(能排進犬淋巴瘤 top10):靶點 1 個、治 5 種人類病、5 個正標籤。

> 💡 **「加一顆藥」不等於「模型會推薦它」。**
> **想讓新藥有機會被推薦,至少要給它:**
> | 加什麼 | 效果 |
> |---|---|
> | `drug_protein.csv` 的靶點 | 打通機制路徑(藥→蛋白→基因→通路→病)|
> | `Kdataset.csv` 的人類適應症 | 讓它能沿「同病橋接」轉移到寵物病 ← **影響最大** |
>
> (teniposide 之所以排上犬淋巴瘤,關鍵正是它被標記治**人類的同一種淋巴瘤**,再沿橋接轉移過來。)

## `protein_gene.csv`(18,545)/ `gene_pathway.csv`(25,995)
```csv
Protein,Gene        Gene,Pathway
0,5628              1,186
```
> 增刪藥/病時**不用動**。這是蛋白/基因/通路之間的固定關係。

## `pathway_disease.csv` — 通路關聯疾病(19,530 列)
```csv
Pathway,Disease
213,0           ← 通路#213 關聯 病#0
```
| 增刪時 |
|---|
| **加寵物病**:不用動(寵物病靠橋接,不靠通路 —— 50 個寵物病目前都沒有通路關聯)|
| **刪病**:必須刪相關列 + **重編 `Disease` 欄** |

---

# C. `interactions/` — 相似度邊表

## `disease_disease.csv`(42,146 列)⭐ 寵物橋接住在這裡
```csv
Disease1,Disease2,Sim
0,4,0.13636363599999998      ← 人類↔人類的 MeSH 相似度(42,046 列,原始資料)
1,454,1.0                    ← 人類#1 ↔ 寵物#454 的橋接(100 列,Sim=1.0)
503,453,1.0                  ← 反向,必須成對!
```
> ❗ **寵物病沒有自己的藥/蛋白/通路資料,100% 靠這裡的橋接繼承人類病的圖結構。**
> **沒橋接 = 圖上的孤島 → 候選近乎隨機**(`run_all.py` 會警告)。
>
> `load_data.py` 只取 **index >= N_HUMAN** 的列當橋接;
> 人類↔人類那 42,046 列它不讀(那份資料是從 `disease_disease_baseline.csv` 讀的)。
>
> **加一個寵物病要加雙向兩列**:`504,16,1.0` 和 `16,504,1.0`
>
> **橋接對象怎麼選?慣例:同名的人類病**
> ```
> 寵物-Hypertension     → 人類病 #1
> 寵物-Breast Neoplasms → 人類病 #4
> ```
> 查法:`grep -n "Lymphoma" dataset/Kdataset/omics/disease.csv`

## `drug_drug.csv` — 藥-藥化學相似(798,316 列)
```csv
Drug1,Drug2,Sim
0,1,0.4189189189189189
```
> 這是上游隨附的 ECFP4 相似度邊表。**本專案的流程不讀它**
> (`build_mech_sim.py` 直接從 SMILES 現算,見 D 節)。增刪藥物時不用動。
>
> ⚠️ 已實測:這份邊表有 **26 格(13 對)寫成 0.0,但正確值是 1.0** —— 全是立體異構物
> (dexamethasone/betamethasone、doxorubicin/epirubicin、ofloxacin/levofloxacin …)。
> ECFP4 忽略立體化學 → 指紋相同 → 1.0 才對。這是上游資料的錯,所以我們不用它。

## `protein_protein.csv`(2,013,782)/ `gene_gene.csv`(712,546)/ `pathway_pathway.csv`(1,669)
> 增刪藥/病時**完全不用動**。

---

# D. 根目錄的三個矩陣(無表頭純數字)

## `Kdataset_baseline.csv` — 藥×病標籤矩陣(894×504)⭐
```
0,0,1,1,1,1,0,0,0,0,...     ← 第 0 列 = 藥#0 對 504 個病
```
| 位置 | 意義 |
|---|---|
| 第 `i` **列** | 藥 `#i` |
| 第 `j` **欄** | 病 `#j`(0~453 人類,454~503 寵物) |
| 值 | `1` = 已知有效、`0` = **未知**(不是「已知無效」!)|

| 增刪時 |
|---|
| **加藥**:加一**列**,504 個 `0` |
| **加寵物病**:每列尾端加一個 `0`(加一**欄**)|
| **刪**:刪對應的列/欄 |

> ⚠️ **必須與 `associations/Kdataset.csv` 一致**:那裡有一列,這裡對應格就要是 `1`。

## `drug_drug_baseline.csv`(894×894)⭐ 決定圖的藥物節點數
```
1.0000,0.2657,0.9115,...      ← 對角線 = 1.0
```
> ❗ **圖的藥物節點數 = 這個檔的維度**,不是標籤矩陣!
> 內容 = **融合相似度 = 0.3·化學(ECFP4) + 0.7·機轉(靶點∪通路 Jaccard)**。
>
> ### 🟢 加藥後這樣重建(不用手算任何數字)
> ```bash
> python build_mech_sim.py 0.3      # 約 1.2 秒
> ```
> 它從 `omics/drug.csv` 的 **SMILES 現算 ECFP4**(RDKit Morgan r=2, 2048 bits),
> 再融合機轉相似度,直接覆蓋本檔。**你只需要填 SMILES。**
>
> **已驗證**:這個設定重現 REDDA 原始 `drug_drug_baseline.csv` 的 798,342 格,
> 精確吻合 **100.0000%**、最大差異 **0.00000000** —— 所以重建不會改變既有數值。

## `disease_disease_baseline.csv`(454×454)
```
1.0,0.0,0.0,0.13636363636363635,...
```
> **只含人類病**。寵物病不在這裡 —— 它們特徵為 0,靠 C 節的橋接傳訊息。
> 這個檔的**列數就是 `N_HUMAN`**,是人類/寵物的分界線。
>
> ⚠️ 增刪【人類】疾病才需要動它。**一般不要動。**
>
> ⚠️ **已知上游問題**(實測,非推測):這份 MeSH 相似度矩陣的**列序與 `omics/disease.csv` 的 ID 對不上**
> (454 列裡只有 2 列沒位移)。它來自 REDDA(Gu et al. 2022, Comput Biol Med),
> MRDDA 原封不動複製(md5 相同)。我們**維持原狀不修**,以保持與上游一致。

---

# E. 速查:每種操作要動哪些檔

| 檔案 | 加藥 | 刪藥 | 加寵物病 | 刪寵物病 |
|---|---|---|---|---|
| `omics/drug.csv` | ✅ 加列(含 SMILES) | 🔴 刪+重編 | — | — |
| `omics/disease.csv` | — | — | ✅ 加列 | 🔴 刪+重編 |
| `omics/protein,gene,pathway.csv` | — | — | — | — |
| `associations/Kdataset.csv` | ⬜ 選填 | 🔴 刪+重編 | ⬜ 加寵物標籤 | 🔴 刪+重編 |
| `associations/drug_protein.csv` | ⬜ **選填**(靶點) | 🔴 刪+重編 | — | — |
| `associations/protein_gene, gene_pathway` | — | — | — | — |
| `associations/pathway_disease.csv` | — | — | — | 🔴 重編 |
| `interactions/disease_disease.csv` | — | — | ✅ **加雙向兩列** | 🔴 刪+重編 |
| `interactions/drug_drug.csv` | — | — | — | — |
| `interactions/protein_protein, gene_gene, pathway_pathway` | — | — | — | — |
| `Kdataset_baseline.csv` | ✅ 加列 | 🔴 刪列 | ✅ 加欄 | 🔴 刪欄 |
| `drug_drug_baseline.csv` | ✅ **跑 `build_mech_sim.py 0.3`** | 🔴 同左 | — | — |
| `disease_disease_baseline.csv` | — | — | — | — |

✅ 必改 ⬜ 選填 🔴 要重編 index — 不用動

## 加一顆藥的完整步驟
```bash
# 1. omics/drug.csv 加一列(ID 接續、必填 SMILES)
echo '894,DB99999,CC(=O)Oc1ccccc1C(=O)O,aspirin' >> dataset/Kdataset/omics/drug.csv

# 2. Kdataset_baseline.csv 加一列 504 個 0
python -c "open('dataset/Kdataset/Kdataset_baseline.csv','a').write(','.join(['0.0']*504)+'\n')"

# 3. 重建融合相似度(從 SMILES 現算,1.2 秒)
python build_mech_sim.py 0.3

# 4.(強烈建議)給它靶點和人類適應症,否則它是圖上的孤島
#    dataset/Kdataset/associations/drug_protein.csv  加 894,<protein_id>
#    dataset/Kdataset/associations/Kdataset.csv      加 894,<human_disease_id>,

# 5. 重訓
python run_all.py
```

## 加一個寵物病的完整步驟
```bash
# 1. omics/disease.csv 加一列(ID=504、Disease 填橋接目標的 MeSH、Name 前綴 寵物-)
echo '504,D054198,寵物-新病名' >> dataset/Kdataset/omics/disease.csv

# 2. interactions/disease_disease.csv 加雙向橋接(16 = 對應的人類病 index)
printf '504,16,1.0\n16,504,1.0\n' >> dataset/Kdataset/interactions/disease_disease.csv

# 3. Kdataset_baseline.csv 每列尾端加一個 0(加一欄)
python - <<'EOF'
p='dataset/Kdataset/Kdataset_baseline.csv'
L=[l.rstrip('\n')+',0.0' for l in open(p)]
open(p,'w').write('\n'.join(L)+'\n')
EOF

# 4.(選填)給它已知有效的藥 → Kdataset.csv 加 <drug>,504,<證據>
#    同時 Kdataset_baseline.csv 對應格改成 1.0

# 5. 重訓
python run_all.py
```

---

# F. 🟢 最推薦:不刪,只清標籤(實測可行)

想棄用某個藥/病,**不要刪** —— 只要:
1. `Kdataset_baseline.csv` 對應格改成 `0.0`
2. `associations/Kdataset.csv` 移除相關列

**零 index 重編、零矩陣調整。** 它會留在圖上但不再是正標籤。

**實測**:清掉 prednisolone 的 10 個寵物標籤(121→111),`run_all.py` 直接跑通 ✅
```
正標籤:人類 2,704 + 寵物 111 = 2,815
✅ 全部通過 — 已產出 2500 個候選
```

---

# G. 改完必做

```bash
python run_all.py        # 一律從頭重訓(GPU ~15 分 / CPU ~2 小時,自動偵測)
```

`run_all.py` 會在**開跑前**檢查並告訴你哪裡不一致:
```
[❌] dataset/Kdataset/drug_drug_baseline.csv = 895×895 — 實際 894×894
   ↳ 增刪藥物後,drug_drug_baseline.csv 要重建:python build_mech_sim.py 0.3

[❌] disease_disease.csv[Disease1] 最大 index 504 < 疾病數 504
   ↳ 這個檔還指向已刪除的疾病(index 504)
```

## 錯誤對照表
| 錯誤訊息 | 真正原因 |
|---|---|
| `ValueError: Target size (451080) must be the same as input size (450576)` | `drug_drug_baseline.csv` 維度沒跟著改 → 跑 `build_mech_sim.py 0.3` |
| `DGLError: Expect number of features to match number of nodes. Got 503 and 504` | 關聯檔還指向已刪除的 index |
| `ValueError: all the input array dimensions except for the concatenation axis must match` | `disease_disease_baseline.csv` 的維度 ≠ 人類病數 |
| 候選近乎隨機 | 新寵物病忘了加橋接 → 圖上孤島 |

---

# H. 誠實提醒:每次訓練結果都不一樣

GNN 訓練是**非確定性**的 —— `utils.py` 的 MetaPath2Vec 即使給同一個 seed,
每次產生的節點特徵都不同(實測最大差異 0.12)。

實測(3 份同設定訓練):彼此相關係數僅 **0.74**;rank1 一致率 **58%**、top50 清單重疊 **82%**。

→ **不要拿單次結果當結論**:
```bash
for s in 42 43 44 45 46; do python run_all.py --seed $s --out cand_$s.csv; done
```
只採信「每次都出現在 top50」的藥。
