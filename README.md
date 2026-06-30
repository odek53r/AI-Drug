# 老藥新用 · 寵物腫瘤 — 跨物種分子證據鏈 Demo

給投資者/合作夥伴看的離線靜態 Demo:輸入寵物癌症 → 系統用「犬腫瘤 driver ∩ 既有犬藥物反應」找線索 → 輸出**每一條都有已發表文獻可追溯**的候選藥。

## 啟動(無需安裝任何套件)
```bash
cd demo
python3 -m http.server 8000
# 瀏覽器開 http://localhost:8000
```
> 若改了證據資料,先重建:`python3 demo/build_demo_data.py`

## 三點故事線(對應 Notion 驗收)
1. **輸入 → 找線索 → 輸出**:左欄選寵物癌 → 中間 graph 展示證據鏈 → 右欄看「為什麼推薦」。
2. **可追溯證據鏈(最重要)**:每個首選藥都附 PubMed/期刊連結;driver、犬反應、機制各自有出處。
3. **可視化**:候選清單(含細胞株活性 % 與證據等級)+ 點擊看證據 + 互動式「疾病 → driver/通路 → 藥」路徑圖。

## 給投資者的一段話
> 「同樣叫黑色素瘤,人那顆是 BRAF 引擎、狗那顆是別的引擎——藥打的是 **driver**,不是病名。我們用**已發表的犬腫瘤基因體 ∩ 既有犬藥物反應**,在不做自己濕實驗的前提下,逐癌種判定哪些人類老藥真能轉移到寵物:膀胱癌(BRAF V595E→sorafenib)、肥大細胞瘤(KIT→toceranib)已被犬證據證實;黑色素瘤則被**主動標為不轉移**——這代表我們的方法會**證偽**,不是事後說故事。」

## 內容(5 個 showcase)
| 寵物癌 | driver(犬頻率) | 證據支持首選 | 等級 |
|---|---|---|---|
| 膀胱泌尿上皮癌 | BRAF V595E (~80%) | sorafenib | ✅ 可轉移 |
| 肥大細胞瘤 | KIT (~33%) | toceranib(已核准) | ✅ 可轉移 |
| 血管肉瘤 | PIK3CA/TP53/NRAS | PI3K / mTOR·HDAC | ✅ 可轉移 |
| 骨肉瘤 | TP53/PTEN/mTOR | mTOR 抑制劑 | ⚠️ 需選病人 |
| 口腔黑色素瘤 | BRAF 野生型 | (無) | ❌ 不轉移 |

## 檔案
- `index.html` / `app.js` / `style.css` — 純靜態前端(Cytoscape.js via CDN)。
- `evidence/curated.json` — **人工核實**的證據與文獻(可信度核心)。
- `build_demo_data.py` — 把策展證據 + `../pet_lab_candidates.csv` 合成 `data/demo.json`。
- `data/demo.json` — 前端唯一讀取的資料。

## 設計原則
- **純策展、零 LLM**:投資者面前不即時生成、不幻覺。
- **誠實**:MRDDA 模型只當「初篩」次要訊號;真正的「為什麼有效」由跨物種 driver 證據鏈提供。
- **不改演算法**:`model.py` 全程未動,本 Demo 不重訓任何模型。
