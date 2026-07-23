# 寵物資料稽核移除紀錄

依據三份稽核 CSV(pet_labels_provenance / pet_bridges_provenance / pet_drugs_fda_status)移除。
人類資料(0~453 號病、2704 筆人類標籤)完全未動。

## 結果

| | 移除前 | 移除後 |
|---|---|---|
| 寵物疾病 | 50 | **48** |
| 寵物標籤 | 121 | **111** |
| 總疾病數 | 504 | **502** |

## 第二輪:移除所有 WEAK(證據弱)標籤

使用者指示「沒證據、證據弱、完全錯誤的都刪掉」→ 只留 SUPPORTED。
只清標籤格(疾病與 index 不動);7 個寵物病因此變零標籤,使用者已知並接受。

**寵物標籤 111 → 89**(刪 22 筆 WEAK)。三類:

### ① 最佳證據為【負面】(文獻打臉標籤)
- sirolimus → Osteosarcoma — Sirolimus was genuinely repurposed and rigorously tested in dogs, but this definitive 324-
- doxorubicin → Breast Neoplasms — Doxorubicin IS given to dogs with mammary carcinoma, but this prospective trial concluded 
- tramadol → Osteoarthritis — Evidence points the OTHER way: a blinded placebo-controlled crossover RCT found tramadol (
- vinblastine → Precursor Cell Lymphoblastic Leukemia-Lymphoma — Evidence actively contradicts the 標準 label: vinblastine was 'minimally efficacious' front-
- hydroxyurea → Leukemia, Myeloid, Acute — DISEASE MISMATCH: the veterinary evidence (and the record's own Chinese label) is for CHRO
- metronidazole → Colitis, Ulcerative — Widely used historically for canine colitis, but this RCT found diet SUPERIOR to metronida
- capecitabine → Colorectal Neoplasms — Off-label/investigational salvage use, NOT standard of care in dogs: in 25 dogs with advan
- itraconazole → Chromoblastomycosis — LABEL MISMATCH: evidence_label says blastomycosis (D001759) but MeSH D002862 is confirmed 
- atenolol → Cardiomyopathy, Hypertrophic, Familial — The '標準貓HCM' label is contradicted by the 2020 ACVIM consensus: 'atenolol has not been sho

### ② 病種對應錯(掛在錯的 MeSH / 亞型不符)
- etoposide → Precursor Cell Lymphoblastic Leukemia-Lymphoma — NOT standard of care and DISEASE MISMATCH: the only canine efficacy evidence is for multic
- amphotericin → Chromoblastomycosis — DOUBT: no chromoblastomycosis-specific dog/cat evidence found; the cited case is phaeohyph

### ③ 只有弱證據(細胞株 / 個案 / off-label salvage)
- metformin → Breast Neoplasms — Only cell-line and xenograft work on canine mammary tumour cells; I found NO clinical tria
- theophylline → Asthma — Merck Vet Manual lists feline dosing (3 mg/kg PO q12h), but efficacy in cats is variable a
- gabapentin → Osteoarthritis — Widely used off-label as a multimodal adjunct but no placebo-controlled trial isolates gab
- methotrexate → Arthritis, Rheumatoid — No primary canine literature found; this review states methotrexate 'has not been evaluate
- methotrexate → Precursor Cell Lymphoblastic Leukemia-Lymphoma — Exact disease match (canine lymphoblastic lymphoma) with intrathecal methotrexate + cytara
- chlorambucil → Precursor Cell Lymphoblastic Leukemia-Lymphoma — Only indirect support: this abstract confirms chlorambucil was a crossover/maintenance age
- azathioprine → Colitis, Ulcerative — Azathioprine is a second-line adjunct to glucocorticoids in canine chronic inflammatory en
- fluorouracil → Colorectal Neoplasms — 5-FU is used in canine carcinomas (24 dogs, 43% response in gross disease) and in canine G
- gemcitabine → Carcinoma, Non-Small-Cell Lung — 37 dogs with mixed carcinomas including 9 respiratory (2 pulmonary carcinoma, 5 bronchoalv
- mitoxantrone → Squamous Cell Carcinoma of Head and Neck — Mitoxantrone has been given to cats with oral SCC (32 cats in this series) but response wa
- doxorubicin → Carcinoma, Hepatocellular — Only solid canine evidence is doxorubicin delivered by drug-eluting bead TACE for NONRESEC
- doxorubicin → Thyroid Neoplasms — Doxorubicin is commonly given for canine thyroid carcinoma (reported response rates 30-50%
- diltiazem → Cardiomyopathy, Hypertrophic, Familial — Real feline HCM data exist (1991 JVIM trial, 12 cats) but the 2020 ACVIM consensus (PMID 3
- azathioprine → Hepatitis, Autoimmune — Consensus names azathioprine as an option some panelists combine with corticosteroids, but

### 因此變零標籤的寵物病(病留著,候選近乎隨機)
Leukemia Myeloid Acute · Colorectal Neoplasms · Carcinoma Renal Cell · NSCLC · HCC · Chromoblastomycosis · Thyroid Neoplasms(7 個)

## A. 整個寵物病移除(病種對應根本錯誤)

- **#468 寵物-Depression** [`NOT_A_VET_DISEASE`]
  - 獸醫無此診斷;連帶移除 fluoxetine 標籤
- **#464 寵物-Colitis, Ulcerative** [`DIFFERENT_ENTITY`]
  - 同名犬病是 E.coli 造成的組織球性潰瘍性結腸炎,吃 enrofloxacin 會好;連帶移除 azathioprine/metronidazole/budesonide/sulfasalazine 4 標籤

## B. 標籤格清除(疾病保留,只有該藥-病關聯錯誤)

- **sunitinib → Carcinoma, Renal Cell** [`HUMAN_ONLY`]
  - 人類 RCC 標準藥;犬用的是 toceranib
- **sulfasalazine → Arthritis, Rheumatoid** [`HUMAN_ONLY`]
  - 人類 RA 的 DMARD;犬用 sulfasalazine 是治結腸炎
- **methotrexate → Lupus Erythematosus, Cutaneous** [`HUMAN_ONLY`]
  - 2018 犬 CLE 回顧整篇未提及
- **rifabutin → NTM Infections** [`NOT_FOUND`]
  - 獸醫指引都用 rifampicin
- **fluorouracil → Colorectal Neoplasms** [`SAFETY`]
  - 對犬【致命】—— FDA 警告所有通報的犬暴露案例全部死亡

## 未移除但已標記存疑(留待人工判斷)

- **WEAK 標籤 25 筆**:有文獻但證據弱,或最佳證據為負面(如 sirolimus→骨肉瘤的 324 犬 RCT 為負)。詳見 pet_labels_provenance.csv 的 verdict 欄。
- **ANALOGOUS 橋接 28 條**:病種存在但機轉/亞型有別,Sim=1.0 過度宣稱(如犬黑色素瘤=口腔黏膜/BRAF 野生型 vs 人類皮膚型)。詳見 pet_bridges_provenance.csv。
- 這兩類【未移除】,因為移除判準設為「病種錯 / 只有人類證據 / 致命」三項,WEAK 與 ANALOGOUS 未達標準。
