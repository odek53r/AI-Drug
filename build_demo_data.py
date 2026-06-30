#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the self-contained demo dataset for the static frontend.

Inputs (read-only):
  - demo/evidence/curated.json     人工核實的跨物種證據(核心)
  - pet_lab_candidates.csv         模型初篩候選(rank + 機制標記)

Output:
  - demo/data/demo.json            前端唯一讀取的檔(含每癌的證據鏈 graph 節點/邊)

無 pandas 相依;用標準庫 json/csv 即可。不重訓模型、不改演算法。
"""
import json, csv, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
CURATED = os.path.join(ROOT, "evidence", "curated.json")
# 候選檔:獨立 repo 放根目錄;MRDDA 內則在上層。自動尋找。
CANDS = next((p for p in [os.path.join(ROOT, "pet_lab_candidates.csv"),
                          os.path.join(REPO, "pet_lab_candidates.csv")]
              if os.path.exists(p)), os.path.join(REPO, "pet_lab_candidates.csv"))
OUT_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(OUT_DIR, "demo.json")

# 顯示用同義詞(DB-ID 經 SMILES 核實 / 商品名 → 通用名)
DRUG_DISPLAY = {
    "DB08871": "eribulin", "DB01254": "dasatinib", "DB06176": "romidepsin",
    "naprosyn": "naproxen", "r-vindesine": "vindesine", "mitomycin c": "mitomycin C",
    "mepron": "atovaquone", "emflaza": "deflazacort",
}

def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:40]

def load_candidates():
    """pet_candidate_key -> [ {drug, rank, mechanism_support} ] (依 rank)"""
    by_dis = {}
    with open(CANDS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_dis.setdefault(row["disease"], []).append({
                "drug_raw": row["drug"].strip(),
                "rank": int(row["rank"]),
                "mechanism_support": (row.get("mechanism_support") or "").strip(),
            })
    for k in by_dis:
        by_dis[k].sort(key=lambda r: r["rank"])
    return by_dis

def build_graph(d):
    """從一個 disease 的策展欄位生成 Cytoscape 節點/邊(driver 中心證據鏈)。"""
    did = d["id"]
    nodes, edges = [], []
    def node(nid, label, ntype, **meta):
        nodes.append({"data": dict(id=nid, label=label, ntype=ntype, **meta)})
    def edge(src, tgt, label="", etype="solid"):
        edges.append({"data": {"id": f"{src}__{tgt}", "source": src, "target": tgt,
                                "label": label, "etype": etype}})

    pet_id = f"{did}__pet"
    hum_id = f"{did}__human"
    drv_id = f"{did}__driver"
    pth_id = f"{did}__pathway"
    node(pet_id, d["pet_disease_zh"], "pet", sub=d["pet_disease_en"], species=d["species"])
    node(hum_id, d["human_disease"], "human", sub=d.get("human_mesh", ""))
    edge(pet_id, hum_id, "疾病橋接")

    drv = d["driver"]
    refuted = d["tier"] == "refuted"
    node(drv_id, f"{drv['gene']}", "driver" if not refuted else "driver_refuted",
         variant=drv.get("variant", ""), freq=drv.get("freq", ""),
         badge=f"犬 {drv.get('freq','')}", citation=drv.get("citation"))
    edge(hum_id, drv_id, "driver 基因")

    node(pth_id, d["pathway"], "pathway")
    edge(drv_id, pth_id, "活化" if not refuted else "(非此驅動)",
         etype="solid" if not refuted else "dashed")

    primary_rid = drv_id if refuted else None
    if refuted:
        rf_id = f"{did}__refuted"
        node(rf_id, "✗ 不轉移", "refuted", note=d.get("refuted_note", ""))
        edge(pth_id, rf_id, "閘門判定", etype="dashed")
    else:
        for i, rec in enumerate(d.get("recommended", [])):
            rid = f"{did}__rec-{i}-{slug(rec['drug'])}"
            if primary_rid is None:
                primary_rid = rid
            ev_id = f"{rid}__ev"
            node(rid, rec["drug"], "drug_rec", rationale=rec.get("rationale", ""),
                 evidence=rec.get("evidence", ""), citation=rec.get("citation"))
            edge(pth_id, rid, "抑制 → 治療")
            node(ev_id, "犬證據", "evidence", text=rec.get("evidence", ""),
                 citation=rec.get("citation"))
            edge(rid, ev_id, "已發表")

    # ★ 老藥新用「結構相似」分支:已知相似藥 → 已知適應症,並收斂到推薦藥
    analogy = d.get("analogy") or {}
    for i, sib in enumerate(analogy.get("siblings", [])):
        sib_id = f"{did}__sib-{i}"
        ind_id = f"{did}__ind-{i}"
        broken = bool(sib.get("broken"))
        node(sib_id, sib["drug"], "sibling", relation=sib.get("relation", ""),
             known_for=sib.get("known_for", ""), basis=analogy.get("basis", ""), broken=broken)
        node(ind_id, sib.get("known_for", ""), "indication")
        edge(sib_id, ind_id, "已知治療")
        if broken:
            edge(sib_id, drv_id, "結構相似但無 driver", etype="dashed")
        elif primary_rid:
            edge(sib_id, primary_rid, "結構相似 → 老藥新用", etype="sim")

    # 模型初篩候選(次要,虛線掛在寵物病上)
    scores = d.get("model_candidate_scores", {})
    for mc in d.get("_model_candidates", [])[:5]:
        nm = mc["drug_disp"]
        key = nm.lower()
        sc = scores.get(key, scores.get(nm, {}))
        mid = f"{did}__model-{slug(nm)}"
        lbl = nm + (f"  ·{sc['cellline_pct']}%" if sc.get("cellline_pct") is not None else "")
        node(mid, lbl, "drug_model", cellline_pct=sc.get("cellline_pct"),
             selective=bool(sc.get("selective")), mech=mc.get("mechanism_support", ""))
        edge(pet_id, mid, "模型初篩", etype="dashed")
    return {"nodes": nodes, "edges": edges}

def main():
    cur = json.load(open(CURATED, encoding="utf-8"))
    cands = load_candidates()
    tiers = cur["meta"]["tiers"]

    out_diseases = []
    for d in cur["diseases"]:
        # 併入模型候選清單
        raw = cands.get(d["pet_candidate_key"], [])
        model_cands = []
        for r in raw:
            disp = DRUG_DISPLAY.get(r["drug_raw"], r["drug_raw"])
            sc = d.get("model_candidate_scores", {}).get(disp.lower(), {})
            model_cands.append({
                "drug": disp, "rank": r["rank"],
                "mechanism_support": r["mechanism_support"],
                "cellline_pct": sc.get("cellline_pct"),
                "selective": bool(sc.get("selective")),
                "note": sc.get("note", ""),
            })
        d["_model_candidates"] = [{"drug_disp": m["drug"], **m} for m in model_cands]

        t = tiers[d["tier"]]
        out_diseases.append({
            "id": d["id"],
            "pet_disease_zh": d["pet_disease_zh"],
            "pet_disease_en": d["pet_disease_en"],
            "species": d["species"],
            "human_disease": d["human_disease"],
            "human_mesh": d.get("human_mesh", ""),
            "tier": d["tier"],
            "tier_label": t["label"], "tier_color": t["color"], "tier_icon": t["icon"],
            "storyline": d["storyline"],
            "driver": d["driver"],
            "pathway": d["pathway"],
            "analogy": d.get("analogy", {}),
            "headline": d.get("recommended", []),
            "refuted_note": d.get("refuted_note", ""),
            "model_candidates": model_cands,
            "sources": d.get("sources", []),
            "graph": build_graph(d),
        })

    demo = {"meta": cur["meta"], "diseases": out_diseases}

    # ---- 驗證 ----
    errs = []
    if len(out_diseases) != 5:
        errs.append(f"預期 5 癌,實得 {len(out_diseases)}")
    for d in out_diseases:
        if len(d["model_candidates"]) < 3:
            errs.append(f"{d['id']}: 模型候選 <3")
        if d["tier"] in ("confirmed", "conditional"):
            if not d["headline"]:
                errs.append(f"{d['id']}: 缺 headline 推薦藥")
            for h in d["headline"]:
                u = (h.get("citation") or {}).get("url", "")
                if not u.startswith("http"):
                    errs.append(f"{d['id']}: 推薦藥 {h.get('drug')} 缺有效引用 URL")
        if d["tier"] == "refuted" and not d["refuted_note"]:
            errs.append(f"{d['id']}: refuted 但缺 refuted_note")
        if not d["graph"]["nodes"] or not d["graph"]["edges"]:
            errs.append(f"{d['id']}: graph 空")
        for n in d["graph"]["nodes"]:
            if not n["data"].get("label"):
                errs.append(f"{d['id']}: 有節點缺 label")
    if errs:
        print("✗ 驗證失敗:")
        for e in errs:
            print("  -", e)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(demo, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    n_nodes = sum(len(d["graph"]["nodes"]) for d in out_diseases)
    n_cite = sum(len(d["sources"]) for d in out_diseases)
    print("✓ 驗證通過")
    print(f"✓ 寫出 {OUT}")
    print(f"  {len(out_diseases)} 癌 · {n_nodes} graph 節點 · {n_cite} 來源引用")
    for d in out_diseases:
        print(f"    {d['tier_icon']} {d['pet_disease_zh']:<14} "
              f"推薦 {len(d['headline'])} · 模型候選 {len(d['model_candidates'])} · "
              f"graph {len(d['graph']['nodes'])}節點")

if __name__ == "__main__":
    main()
