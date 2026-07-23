# 資料版本

寵物資料集的版本紀錄。與舊 tag `v1.0-frozen` / `v2.0-full`(**模型/流程**版本)是不同維度 ——
這裡記的是**資料集本身**。每一版都經過完整重訓驗證才打 tag。

| | data-v1 | data-v2 | data-v3 | data-v4 |
|---|---|---|---|---|
| **主題** | 合併重構 | **三陰性乳癌(TNBC)** | **寵物標籤證據擴充** | **相似度改 Wang's method** |
| 疾病總數 | 504 | **503** | 503 | 503 |
| 人類病 | 454 | **455**(+TNBC) | 455 | 455 |
| 寵物病 | 50 | **48** | 48 | 48 |
| 寵物標籤 | 121 | **89** | **158** | 158 |
| TNBC 標籤 | — | **23** | 23 | 23 |
| 正標籤總計 | 2,825 | **2,816** | **2,885** | 2,885 |
| 寵物疾病相似度 | 複製橋接對象整列 | 同左 | 同左 | ✅ **Wang's method 現算** |
| 每筆可回溯來源 | ❌ | ✅ | ✅ | ✅ + `data_provenance.csv` |

---

## 唯一的來源紀錄:`data_provenance.csv`

**282 列 × 29 欄。** 逐列對應訓練資料裡**本專案新增的每一筆**,每列都能查回原始出處。

| | 列數 |
|---|---|
| `drug_disease_label` 藥-病關聯 | 181 |
| `disease_node` 疾病節點 | 49 |
| `disease_bridge` 寵物↔人類橋接 | 48 |
| `column` / `matrix` 矩陣層級 | 4 |

**282 / 282 列帶識別碼或可點 URL**,零缺口。

**識別碼欄**:`pmid` · `doi` · `nct_id` · `chembl_id` · `mesh_id` · `mondo_id` · `venom_id` ·
`drugbank_id` · `source_url` · `ref_url` · `query_used`(當初用的查詢字串)

**來源分布**

| 來源 | 列數 | 授權 |
|---|---|---|
| PubMed / PMC (NCBI) | 177 | 公眾領域(摘要);全文依期刊 |
| MeSH (NLM) | 40 | 公眾領域(U.S. Government work) |
| 獸醫指引 / 專業手冊 | 40 | 期刊/機構版權 |
| ClinicalTrials.gov | 23 | 公眾領域(U.S. Government work) |
| DrugBank / REDDA | 2 | ⚠️ 學術/商用分流,商用需授權 |

**證據等級**:case series 45 · 指引 45 · 前瞻試驗 31 · review 20 · RCT 17 ·
臨床 phase 2 (9) / phase 1 (8) / phase 3 (6)

---

## data-v1 — 合併重構

寵物資料併入原始 Kdataset 的 18 個檔,零新增檔案。演算法 `model.py` 逐行未改。

---

## data-v2 — 新增三陰性乳癌(TNBC)

新增人類病 `#454 = D064726 Triple Negative Breast Neoplasms`(資料集原本只有籠統的
`D001943 Breast Neoplasms`)+ **23 筆藥物關聯**。

**Phase 3 六個**:paclitaxel(348 個試驗)· doxorubicin(146)· capecitabine(110)·
gemcitabine(82)· epirubicin(81)· eribulin(56)

**資料來源(全部結構化資料庫)**

| 層 | 來源 | 結果 |
|---|---|---|
| 主來源 | **ChEMBL** `drug_indication?mesh_id=D064726` | 125 筆適應症,含臨床階段 |
| 交叉驗證 | **ClinicalTrials.gov API v2** | **23/23 皆有真實 TNBC 試驗**(附 NCT 編號) |
| ID 驗證 | **PubChem PUG REST** `xref/RegistryID→synonyms` | 逐筆確認 DrugBank ID |
| 疾病 ID | **NLM MeSH SPARQL** | D064726 確認 |
| 相似度 | **REDDA 的 Wang's method**(δ=0.5)現算 | TNBC↔Breast Neoplasms = **0.7975**(最高,符合 MeSH 父子關係) |

`disease_disease_baseline.csv` 454×454 → **455×455**;六個檔同步改,寵物病 index 全部 +1。

---

## data-v3 — 寵物標籤證據擴充

替現有寵物病補**有文獻證據**的標準療法(寵物標籤 89 → 158)。

**+69 筆**,來源:PubMed 獸醫期刊(JVIM/JAVMA/Vet Comp Oncol/J Feline Med Surg)+
ACVIM/AAHA/ICADA/LeishVet/ABCD 指引 + Merck-MSD 手冊

**證據等級**:RCT 8 · 前瞻試驗 13 · 指引 14 · case series 17 · review 17

**替 5 個癌症補上該病真正的化療藥**:
```
急性髓白血病  cytarabine + doxorubicin   Vet Comp Oncol 2023(11 犬,7 緩解)
大腸癌        piroxicam                  直腸栓劑,8 犬 4 有效
非小細胞肺癌  vinorelbine                犬肺癌一線,10 犬 8 部分緩解
肝細胞癌      sorafenib                  不可切除犬 HCC,TTP 363 天
甲狀腺癌      doxorubicin
```

**實測成效**:這 5 個癌的 top1 從 `gabapentin`(止痛藥,score 21~28)換成該病真正的
化療藥(肝癌 → gemcitabine/eribulin/doxorubicin;AML → vincristine/melphalan)。

---

## data-v4 — 寵物疾病相似度改用 Wang's method

**標籤資料一筆未動**(仍是 v3 的 2,885)。改的是 `run_all.py` 裡 prop 用的疾病相似度矩陣。

**問題**:舊做法 `Sdis[p,:] = ddb[h,:]` 是把橋接人類病的相似度**整列複製**給寵物病。
但 `disease_disease_baseline.csv` 的列序與 `omics/disease.csv` 的 ID 對不上(上游已知問題,
454 列裡只有 2 列沒位移),複製過來的其實是**別的病**的相似度輪廓:

```
寵物-Hypertension 的最近鄰
  舊  IgA Vasculitis 0.2551 · Hypereosinophilic Syndrome 0.2441      ← 錯位造成
  新  Essential Hypertension 0.7241 · 肺高壓 0.6000 · 動脈阻塞 0.4286  ← 生物學上正確
```

**改法**:用 **Wang's method**(MeSH DAG,δ=0.5)直接算 48 寵物病 × 455 人類病。
公式與 REDDA `disease_similarity.py` 相同(已驗證能重現原矩陣 95.62%)。
結果寫進 `interactions/disease_disease.csv`(42,142 → **50,850** 列)。
逐格只有 **72.3%** 與舊值相同。

**48 條橋接的 `Sim` 是 1.0** —— 那是 Wang 的**計算輸出**(48/48 驗證相符),不是寫死的常數。
寵物病與橋接人類病共用同一個 MeSH ID,`Sim(D007946, D007946)` 必然 = 1.0。

**圖結構不變**:`load_data.py` 只把 `Sim == 1.0` 的當**邊**(96 條);`Sim < 1` 是給 prop 的
**權重**,不是邊。若一併當邊,寵物邊會從 96 暴增到 4,498。實測 894 藥 / 503 病 /
寵物邊 96 / 寵物特徵 0,與 v3 完全相同。

**重訓驗證**(GPU 14.7 分,seed 42):Loss 0.103 · Train AUC 0.997 · 三項檢查全過。

> ⚠️ **誠實結論:候選排名幾乎沒變。** 48 個病 top50 重疊 **76.2%**,而同設定重跑的基準
> 本來就是 **77~82%**(MetaPath2Vec 非確定性)。改動落在雜訊範圍內,**不能宣稱
> 「Wang 版比較好」** —— 只能說相似度輪廓在生物學上正確了。要證明對結果有益,
> 得跑多 seed + OOF 評估,本包沒做。

**產出**:`stack_candidates_v4.csv` —— 48 病 × top50 = **2,400 個候選**

```
待預測格子   894 藥 × 48 寵物病 − 158 已知 = 42,754 格
分數校準     已知有效藥平均 0.996 / 未知平均 0.016
無洩漏檢查   候選全為 Y=0 的格子,違反 0 個;寵物標籤未被更動 158 = 158
```

⚠️ 候選檔裡 **535 個藥沒有名字,顯示為 DrugBank ID**(佔 2,400 筆中的 1,722 筆)。
`omics/drug.csv` 的 Name 欄目前只填了 91 個 —— 都是會出現在寵物標籤裡的藥。

---

## 獸醫本體 ID(v4 補上)

`omics/disease.csv` 的寵物病掛的是**人類 MeSH ID**(如 `D006973`),那是橋接時借來的。
`data_provenance.csv` 另外記了真正定義寵物疾病的識別碼:

```
mondo_id       MONDO:1013079                              venom_id  VeNom:1036
mondo_label    hypertensive disorder, non-human animal
ontology_match EXACT
```

**為什麼是 MONDO / VeNom**(查證過的現況)

| 系統 | 有無寵物疾病 ID | 取用 |
|---|---|---|
| MeSH | **無** —— 0 個犬/貓專屬 descriptor,動物只有 `/veterinary` 副標題(Q000662) | 公開 |
| FDA Green Book | **無** —— `approved_indication` 是純文字,NADA 編號標的是核准案不是疾病 | 公開 |
| VeNom Coding | **有** —— 獸醫**臨床病歷**的實際編碼標準(VetCompass 用) | 需註冊,下載頁密碼保護 |
| SNOMED CT 獸醫擴充 | **有** | 需 affiliate licence |
| **MONDO** | **有** —— 動物分支 8,318 詞條,其中 5,045 帶 `VeNom:` 交叉引用 | 免費、可程式化 |

補 MONDO 等於同時拿到 VeNom 碼。**48 個寵物病中 46 個有對映,37 個帶 VeNom 臨床碼。**

**核對方法**(以 MeSH ID 為錨點,不靠病名字面猜;MONDO releases/2026-07-06,63,126 詞條)

```
第一輪  MeSH → MONDO 人類詞條(xref)→ 同名動物詞條                       22/48
第二輪  加入 MONDO 人類詞條的同義詞,記錄靠哪個同義詞對上                  27/48
第三輪  逐一精查動物分支關鍵字 + 人工判定粒度,粒度不合的照實標註           46/48
```

`ontology_match` 只有 **EXACT** 代表同概念同粒度:

```
EXACT             36    BROADER    4    NARROWER  3
ANALOGOUS          2    NONE       2    SPECIES_SPECIFIC  1
```

**12 筆不是精確對應** —— 靠純文字病名時這 12 筆全部長得像完美匹配。

其中一筆特別查了文獻確認:`Myoclonic Epilepsy, Juvenile → MONDO:1012511`
(generalized myoclonic epilepsy with photosensitivity, DIRAS1-related, dog)——
PMID 29194766 標題即 *"Juvenile Myoclonic Epilepsy in Rhodesian Ridgeback Dogs"*。

⚠️ 這些 ID **不改變任何訓練行為**。Wang's method 仍用 MeSH DAG 算相似度,因為 MONDO
動物分支掛在 `MONDO:0005583 non-human animal disease` 底下,與人類病只共用最頂層的
root —— 實測犬骨肉瘤 vs 人類骨肉瘤只有 **0.0039**。補 ID 是為了**可追溯性**。

---

## ⚠️ 已知限制(v4 仍未解決)

**① 兩個病沒有標籤,有結構性原因**
- `Carcinoma, Renal Cell` —— 犬用 **toceranib(Palladia)**,獸醫專用藥,**不在 894 人類藥池**
- `Chromoblastomycosis` —— 病種對應本身就錯(真正的犬病是 phaeohyphomycosis);
  MONDO 動物分支也沒有這兩個詞條

**② 核心獸醫藥不在藥池**:carboplatin、cisplatin、cyclophosphamide、toceranib、
pimobendan、carbimazole、L-asparaginase 全都沒有 —— 藥池來自 DrugBank(人類藥物庫)。

**③ 28 條橋接是 ANALOGOUS,卻拿到 `Sim=1.0`**

Wang 沒算錯 —— 它的輸入是兩個 MeSH ID,而我們把人類的 MeSH 借給了寵物節點,
`Sim(同一個 ID, 同一個 ID)` 必然 = 1.0。**問題在建模層,不在演算法層**;
任何以 MeSH ID 為輸入的相似度算法都看不到物種差異。

最嚴重的是肥大細胞:節點掛 `D007946 Leukemia, Mast-Cell`,但犬貓的是
`MONDO:1011441 / VeNom:16441 mast cell tumor`(`ontology_match = ANALOGOUS`),
而且人類 D816V 對 imatinib **抗藥**、犬 exon11 **有效** —— 方向相反。

實測後果:`#502` 的 top10 候選是人類白血病的化療組合(methotrexate · cytarabine ·
mitoxantrone · vincristine · teniposide),不是犬 MCT 的用藥。
**標籤本身沒有對應錯**(掛的是 vinblastine/lomustine/prednisolone/chlorambucil,
確為犬 MCT 標準療法);錯的是候選,經由橋接繼承而來。

要修只有兩條路,都有代價:
- 寵物節點改掛 MONDO ID → 實測掉到 ~0.004,等於切斷所有訊號,更糟
- 對 ANALOGOUS 橋接乘折扣係數 → 找不到客觀的係數來源,隨手填就回到寫死常數的問題

**④ 寵物病節點不分犬貓**:有標籤是**對貓有效、對犬無效**(如 `tramadol→骨關節炎`),
但一個 `寵物-Osteoarthritis` 節點同時代表兩者,模型看到的只是「有效」。

**⑤ GNN 學不到種間毒性**:5-FU 對犬致命,但模型仍會預測它 —— 這是結構性盲點。
