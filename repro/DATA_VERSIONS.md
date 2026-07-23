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
| 有 Evidence 註記 | 121 | **112** | **181** | 181 |
| 正標籤總計 | 2,825 | **2,816** | **2,885** | 2,885 |
| 寵物疾病相似度 | 複製橋接對象整列 | 同左 | 同左 | ✅ **Wang's method 現算** |
| 每筆可回溯來源 | ❌ 只有「標準CHOP」這種字串 | ✅ | ✅ | ✅ + `data_provenance.csv` |

---

## data-v1 — 合併重構(公開包現況)

寵物資料併入原始 Kdataset 的 18 個檔,零新增檔案。演算法 `model.py` 逐行未改。
已發布於 `odek53r/AI-Drug` 的 `repro/`。

⚠️ **後續稽核才發現的問題**:121 筆寵物標籤是憑模型內建知識策展的,
`Evidence` 欄只有「標準CHOP」這種字串,**零引用**,無法驗證。

---

## data-v2 — 新增三陰性乳癌(TNBC)

### 先做的兩次清理(把 v1 的 121 筆清成 89 筆)

**① 移除病種對應錯 / 只有人類證據 / 對犬致命**(121 → 111)
- 整病移除:`Depression`(獸醫無此診斷)、`Colitis, Ulcerative`(同名犬病是 E.coli
  造成的組織球性潰瘍性結腸炎,吃 enrofloxacin 會好 —— 完全不同的病)
- 標籤移除:sunitinib→RCC、sulfasalazine→RA、methotrexate→皮膚狼瘡(人類適應症漏入)、
  rifabutin→NTM(獸醫用 rifampicin)、fluorouracil→大腸癌(**對犬致命**,FDA 安全警告)

**② 移除證據弱的**(111 → 89,全部 SUPPORTED)
- 22 筆 WEAK,其中 9 筆的**最佳證據其實是負面的**:tramadol→骨關節炎(雙盲 RCT 證明無效)、
  sirolimus→骨肉瘤(324 犬 RCT 負面)、vinblastine→淋巴瘤(VCO 2018「minimally
  efficacious」不建議)、doxorubicin→乳腺瘤(前瞻試驗「chemotherapy did not lead to
  improved outcome」)…
- 代價:7 個寵物病變零標籤

### 主體:新增 TNBC

新增人類病 `#454 = D064726 Triple Negative Breast Neoplasms`(資料集原本只有籠統的
`D001943 Breast Neoplasms`)+ **23 筆藥物關聯**。

**Phase 3 六個**:paclitaxel(348 個試驗)· doxorubicin(146)· capecitabine(110)·
gemcitabine(82)· epirubicin(81)· eribulin(56)

**資料來源(全部結構化資料庫)**

| 層 | 來源 | 結果 |
|---|---|---|
| 主來源 | **ChEMBL** `drug_indication?mesh_id=D064726` | 125 筆適應症,含臨床階段 |
| 交叉驗證 | **ClinicalTrials.gov API v2** | **23/23 皆有真實 TNBC 試驗**(附 NCT 編號) |
| ID 驗證 | **PubChem PUG REST** `xref/RegistryID→synonyms` | 剔除 1 筆無法確認(ivermectin) |
| 疾病 ID | **NLM MeSH SPARQL** | D064726 確認 |
| 相似度 | **REDDA 的 Wang's method**(δ=0.5)現算 | TNBC↔Breast Neoplasms = **0.7975**(最高,符合 MeSH 父子關係) |

`disease_disease_baseline.csv` 454×454 → **455×455**;六個檔同步改,寵物病 index 全部 +1。

**產出**:`tnbc_drug_disease.csv`(23 列 × 17 欄,每列 4 個可點來源 URL)

---

## data-v3 — 寵物標籤證據擴充

在 v2 之上,替現有寵物病補**有文獻證據**的標準療法(寵物標籤 89 → 158)。

**+69 筆**,來源:PubMed 獸醫期刊(JVIM/JAVMA/Vet Comp Oncol/J Feline Med Surg)+
ACVIM/AAHA/ICADA/LeishVet/ABCD 指引 + Merck-MSD 手冊

**證據等級**:RCT 8 · 前瞻試驗 13 · 指引 14 · case series 17 · review 17

**補回 5 個零標籤癌**(零標籤病 7 → 2):
```
急性髓白血病  cytarabine + doxorubicin   Vet Comp Oncol 2023(11 犬,7 緩解)
大腸癌        piroxicam                  直腸栓劑,8 犬 4 有效
非小細胞肺癌  vinorelbine                犬肺癌一線,10 犬 8 部分緩解
肝細胞癌      sorafenib                  不可切除犬 HCC,TTP 363 天
甲狀腺癌      doxorubicin
```

**實測成效**:先前零標籤癌被止痛藥霸榜的雜訊消失了 ——
`gabapentin`(score 21~28 佔 top1)被該病真正的化療藥取代
(肝癌 → gemcitabine/eribulin/doxorubicin;AML → vincristine/melphalan)。

**產出**:`pet_labels_expansion.csv`(69 列)

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

**48 條橋接的 `Sim` 仍是 1.0** —— 但那是 Wang 的**計算輸出**(48/48 驗證相符),不再是
先前寫死的常數。寵物病與橋接人類病共用同一個 MeSH ID,公式必然給 1.0。

**圖結構不變**:`load_data.py` 只把 `Sim == 1.0` 的當**邊**(96 條);`Sim < 1` 是給 prop 的
**權重**,不是邊。若一併當邊,寵物邊會從 96 暴增到 4,498。實測 894 藥 / 503 病 /
寵物邊 96 / 寵物特徵 0,與 v3 完全相同。

**重訓驗證**(GPU 14.7 分,seed 42):Loss 0.103 · Train AUC 0.997 · 三項檢查全過。

> ⚠️ **誠實結論:候選排名幾乎沒變。** 48 個病 top50 重疊 **76.2%**,而同設定重跑的基準
> 本來就是 **77~82%**(MetaPath2Vec 非確定性)。改動落在雜訊範圍內,**不能宣稱
> 「Wang 版比較好」** —— 只能說相似度輪廓在生物學上正確了。要證明對結果有益,
> 得跑多 seed + OOF 評估,本包沒做。

**產出**:`stack_candidates_wang.csv`(2,400 候選)

---

## 資料來源與審查(v2/v3 共通)

**證據層**:agent 被明確要求「accuracy 最重要,**寧可回 NOT_FOUND 也不准編 PMID**」。
**PMID 存在性**:37 個 PMID 經 **NCBI E-utilities** 親自 fetch → **37/37 真實、標題相符**。
**ID 層**:不信 agent 也不信資料檔的名字欄,用 **PubChem + PrimeKG** 交叉驗證。

**審查攔下的 3 個 ID 錯位**:
- `DB07615` agent 誤認 deracoxib,PubChem 查是 **tranilast** → 剔除
- `DB00602` ivermectin 的 ID 無法用 PubChem 確認 → 剔除
- `DB00361` 資料檔標「melphalan」,實際是 **vinorelbine** → agent 用對了 ID,錯的是舊資料

**稽核產出**(每筆都有可點的來源 URL):
```
pet_labels_provenance.csv    121 列 × 24 欄  原始標籤的文獻查證
pet_bridges_provenance.csv    50 列 × 20 欄  病-病橋接的跨物種查證
pet_drugs_fda_status.csv      79 列 × 13 欄  FDA Green Book 動物核准狀態
pet_labels_expansion.csv      69 列 × 13 欄  v3 新增標籤
tnbc_drug_disease.csv         23 列 × 17 欄  v2 的 TNBC
```

**v4 加的兩份總表**(把上面五份 + 論文原始資料合起來盤點):
```
sources.csv           25 列 × 14 欄  每個資料來源的 endpoint / 版本 / 授權 /
                                     已知問題 / 驗證方式 / 驗證結果(24/25 逐一實查)
data_provenance.csv  282 列 × 24 欄  逐列掃訓練資料(不是拼來源檔,所以缺口會浮出來)
                                     欄位足以精準找回:pmid · doi · nct_id · chembl_id ·
                                     mesh_id · drugbank_id · source_url · query_used
```

---

## ⚠️ 已知限制(v4 仍未解決)

**① 兩個病補不到標籤,有結構性原因**
- `Carcinoma, Renal Cell` —— 犬用 **toceranib(Palladia)**,獸醫專用藥,**不在 894 人類藥池**
- `Chromoblastomycosis` —— 病種對應本身就錯(真正的犬病是 phaeohyphomycosis)

**② 核心獸醫藥不在藥池**:carboplatin、cisplatin、cyclophosphamide、toceranib、
pimobendan、carbimazole、L-asparaginase 全都沒有 —— 藥池來自 DrugBank(人類藥物庫)。

**③ 28 條橋接是 ANALOGOUS(未修)**:`Sim=1.0` 過度宣稱 —— v4 改用 Wang's method 後
這個值變成「算出來的」而非寫死的,但因為寵物病沿用人類 MeSH ID,公式必然給 1.0,
所以**問題本身沒解決**。要讓它不是 1.0,得給寵物病自己的疾病識別碼(如 MONDO 的
`canine osteosarcoma`),但實測那樣會掉到 0.0039(MONDO 把犬/人分成不同分支),更不合理。
最嚴重的是肥大細胞白血病
(人類 D816V 對 imatinib **抗藥**,犬 exon11 **有效** —— 相反)與黑色素瘤(犬口腔黏膜/
BRAF 野生型 vs 人類皮膚型 UV/BRAF-V600)。詳見 `pet_bridges_provenance.csv`。

**④ 寵物病節點不分犬貓**:v3 有 7 筆(如 `tramadol→骨關節炎`)是**對貓有效、對犬無效**,
但一個 `寵物-Osteoarthritis` 節點同時代表兩者,模型看到的只是「有效」。

**⑤ fluorouracil 清標籤後仍被模型預測**:GNN 學不到種間毒性(5-FU 對犬致命),
這是結構性盲點,不是資料錯誤。
