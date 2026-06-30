"use strict";
let DEMO = null, CUR = null, cy = null;

const NTYPE_LABEL = {
  pet:"寵物癌", human:"人類癌", driver:"driver 基因", driver_refuted:"driver(野生型)",
  pathway:"靶點/通路", drug_rec:"證據支持首選", evidence:"犬證據",
  drug_model:"模型初篩候選", refuted:"閘門判定",
  sibling:"已知相似老藥", indication:"已知適應症"
};
const NTYPE_COLOR = {
  pet:"#0f9b8e", human:"#2b6cb0", driver:"#1a9850", driver_refuted:"#9aa0a6",
  pathway:"#7b5ea7", drug_rec:"#e8821a", evidence:"#5aa469",
  drug_model:"#b9c0c8", refuted:"#d73027",
  sibling:"#4f8cd9", indication:"#5b6b7b"
};
const WHITE_TEXT = ["human","pathway","refuted","sibling","indication"];

fetch("data/demo.json").then(r=>r.json()).then(data=>{
  DEMO = data;
  document.getElementById("app-subtitle").textContent = data.meta.subtitle;
  document.getElementById("footer-note").textContent =
    data.meta.method_note + "  ·  原始演算法未改,本 Demo 不重訓模型。";
  renderDiseaseList();
  selectDisease(data.diseases[0].id);
});

function renderDiseaseList(){
  const box = document.getElementById("disease-list");
  box.innerHTML = "";
  DEMO.diseases.forEach(d=>{
    const el = document.createElement("div");
    el.className = "disease-item"; el.dataset.id = d.id;
    el.innerHTML = `<span class="di-icon">${d.tier_icon}</span>
      <div><div class="di-name">${d.pet_disease_zh}</div>
      <div class="di-sub">${d.driver.gene} · 犬 ${d.driver.freq||""}</div></div>`;
    el.onclick = ()=>selectDisease(d.id);
    box.appendChild(el);
  });
}

function selectDisease(id){
  CUR = DEMO.diseases.find(d=>d.id===id);
  document.querySelectorAll(".disease-item").forEach(e=>
    e.classList.toggle("active", e.dataset.id===id));
  // 故事線 + tier
  document.getElementById("storyline").textContent = CUR.storyline;
  const tb = document.getElementById("tier-badge");
  tb.textContent = CUR.tier_icon+" "+CUR.tier_label;
  tb.style.background = CUR.tier_color;
  tb.style.color = (CUR.tier==="conditional") ? "#3a2400" : "#04130c";
  renderCandidates();
  buildGraph();
  resetEvidence();
}

function renderCandidates(){
  const box = document.getElementById("candidate-list");
  box.innerHTML = "";
  // headline 推薦(若有)
  CUR.headline.forEach((h,i)=>{
    const el = document.createElement("div");
    el.className = "cand headline";
    el.innerHTML = `<div class="cand-top"><span class="cand-name">⭐ ${h.drug}</span>
      <span class="cand-tag">證據支持</span></div>
      <div class="cand-meta">${CUR.driver.gene} 驅動 · 點看文獻</div>`;
    el.onclick = ()=>{ showEvidenceForRec(h); highlightNodeByLabel(h.drug); };
    box.appendChild(el);
  });
  if(CUR.tier==="refuted"){
    const el=document.createElement("div");
    el.className="cand"; el.style.borderColor="var(--refuted)";
    el.innerHTML=`<div class="cand-top"><span class="cand-name">✗ 無證據支持的推薦</span></div>
      <div class="cand-meta">${CUR.refuted_note}</div>`;
    el.onclick=()=>showRefuted();
    box.appendChild(el);
  }
  // 模型初篩候選
  const head=document.createElement("div");
  head.className="cand-meta"; head.style.margin="10px 2px 2px";
  head.textContent="—— 模型初篩候選(細胞株活性 %)——";
  box.appendChild(head);
  CUR.model_candidates.forEach(m=>{
    const el = document.createElement("div");
    el.className = "cand";
    const pct = (m.cellline_pct!=null) ? m.cellline_pct : null;
    const bar = pct!=null ? `<div class="cand-bar"><i style="width:${pct}%"></i></div>` : "";
    const tag = m.mechanism_support ? `<span class="cand-tag mech">機制</span>` : "";
    el.innerHTML = `<div class="cand-top"><span class="cand-name">${m.drug}</span>${tag}</div>
      ${bar}
      <div class="cand-meta">${pct!=null?("細胞株活性 "+pct+"%"+(m.selective?" · 選擇性":"")):"GDSC/PRISM 無資料"}${m.note?(" · "+m.note):""}</div>`;
    el.onclick = ()=>{ showModelCand(m); highlightNodeByLabel(m.drug); };
    box.appendChild(el);
  });
}

/* ---------- Cytoscape ---------- */
function buildGraph(){
  const g = CUR.graph;
  // 三個水平帶:上帶=結構相似類比分支;中帶=主機制鏈;下方=模型初篩候選
  const layerX = {indication:2, sibling:3, pet:0,human:1,driver:2,driver_refuted:2,pathway:3,drug_rec:4,refuted:4,evidence:5};
  const UPPER = new Set(["sibling","indication"]);
  const cyc = 215, cx0 = 70, cyMid = 280, gapY = 74;
  const mid={}, up={}; const modelNodes=[];
  g.nodes.forEach(n=>{
    const t=n.data.ntype;
    if(t==="drug_model"){ modelNodes.push(n); return; }
    const L = layerX[t] ?? 3;
    const bucket = UPPER.has(t) ? up : mid;
    (bucket[L]=bucket[L]||[]).push(n);
  });
  Object.entries(mid).forEach(([L,arr])=>arr.forEach((n,i)=>{
    n.position={x: cx0 + L*cyc, y: cyMid + (i-(arr.length-1)/2)*gapY};
  }));
  Object.entries(up).forEach(([L,arr])=>arr.forEach((n,i)=>{
    n.position={x: cx0 + L*cyc, y: cyMid - 175 - i*gapY};
  }));
  modelNodes.forEach((n,i)=>{ n.position={x: cx0, y: cyMid+110 + i*48}; });

  const elements = g.nodes.concat(g.edges);
  if(cy) cy.destroy();
  cy = cytoscape({
    container: document.getElementById("cy"),
    elements,
    layout: {name:"preset", fit:true, padding:30},
    style: [
      {selector:"node", style:{
        "label":"data(label)","color":"#06121a","font-size":"12px","font-weight":600,
        "text-valign":"center","text-halign":"center","text-wrap":"wrap","text-max-width":"130px",
        "shape":"round-rectangle","width":"label","height":"label",
        "padding":"9px","background-color":"#ccc","border-width":0
      }},
      ...Object.entries(NTYPE_COLOR).map(([t,c])=>(
        {selector:`node[ntype="${t}"]`, style:{"background-color":c,
          "color": WHITE_TEXT.includes(t)?"#fff":"#06121a"}})),
      {selector:'node[ntype="drug_rec"]', style:{"border-width":3,"border-color":"#ffd9a8","font-size":"13px","font-weight":700}},
      {selector:'node[ntype="drug_model"]', style:{"font-size":"11px","opacity":.9}},
      {selector:'node[ntype="sibling"]', style:{"border-width":2,"border-color":"#bcd6f5","font-weight":700,"shape":"round-rectangle"}},
      {selector:'node[ntype="indication"]', style:{"font-size":"11px","opacity":.92}},
      {selector:'node[ntype="driver_refuted"]', style:{"border-width":2,"border-color":"#d73027"}},
      {selector:'node[?broken]', style:{"border-color":"#d73027","border-width":2,"border-style":"dashed"}},
      {selector:".faded", style:{"opacity":.16}},
      {selector:".hl", style:{"border-width":4,"border-color":"#36c5b0"}},
      {selector:"edge", style:{
        "label":"data(label)","font-size":"9.5px","color":"#9fb0c0","text-background-color":"#0e1620",
        "text-background-opacity":.8,"text-background-padding":"2px",
        "width":2,"line-color":"#3a4d5e","target-arrow-color":"#3a4d5e",
        "target-arrow-shape":"triangle","curve-style":"bezier"
      }},
      {selector:'edge[etype="dashed"]', style:{"line-style":"dashed","line-color":"#55606b","target-arrow-color":"#55606b"}},
      {selector:'edge[etype="sim"]', style:{"line-color":"#e8b04b","target-arrow-color":"#e8b04b","width":3,"color":"#f0c674","font-weight":700,"font-size":"10px"}},
      {selector:"edge.hl", style:{"line-color":"#36c5b0","target-arrow-color":"#36c5b0","width":3}}
    ]
  });
  cy.on("tap","node", evt=>onNodeTap(evt.target));
  cy.on("tap", evt=>{ if(evt.target===cy) clearHighlight(); });
}

function onNodeTap(node){
  const d = node.data();
  highlightPath(node);
  if(d.ntype==="drug_rec") showEvidenceForRec(findRec(d.label));
  else if(d.ntype==="driver"||d.ntype==="driver_refuted") showDriver();
  else if(d.ntype==="evidence") showEvidenceNode(d);
  else if(d.ntype==="drug_model") showModelCand(findModel(d.label));
  else if(d.ntype==="refuted") showRefuted();
  else if(d.ntype==="pet"||d.ntype==="human") showDiseaseInfo(d);
  else if(d.ntype==="pathway") showPathway(d);
  else if(d.ntype==="sibling") showSibling(d);
  else if(d.ntype==="indication") showIndication(d);
}

function highlightPath(node){
  clearHighlight();
  const chain = node.predecessors().union(node.successors()).union(node);
  cy.elements().addClass("faded");
  chain.removeClass("faded").addClass("hl");
}
function clearHighlight(){ if(cy){ cy.elements().removeClass("faded hl"); } }
function highlightNodeByLabel(lbl){
  if(!cy) return;
  const n = cy.nodes().filter(n=>n.data("label").split("  ·")[0].trim()===lbl.trim());
  if(n.length) highlightPath(n[0]);
}

/* ---------- 找對應策展物件 ---------- */
function findRec(lbl){ return CUR.headline.find(h=>h.drug===lbl) || CUR.headline[0]; }
function findModel(lbl){ const nm=lbl.split("  ·")[0].trim();
  return CUR.model_candidates.find(m=>m.drug===nm) || {drug:nm}; }

/* ---------- 證據面板 ---------- */
function setPanel(html){ document.getElementById("evidence-panel").innerHTML = html + sourcesBlock(); }
function resetEvidence(){
  document.getElementById("evidence-panel").innerHTML =
    `<p class="hint">← 點左側候選藥,或中間 graph 的任一節點,這裡會顯示「為什麼推薦」與可追溯的文獻來源。</p>`
    + sourcesBlock();
}
function citeHTML(c){ return c&&c.url ? `<div class="cite">📄 <a href="${c.url}" target="_blank" rel="noopener">${c.title}</a></div>` : ""; }
function badge(){ return `<span class="ev-badge" style="background:${CUR.tier_color};color:${CUR.tier==='conditional'?'#3a2400':'#04130c'}">${CUR.tier_icon} ${CUR.tier_label}</span>`; }
function kind(t){ return `<span class="ev-kind" style="background:${NTYPE_COLOR[t]};color:${WHITE_TEXT.includes(t)?'#fff':'#06121a'}">${NTYPE_LABEL[t]}</span>`; }

function showEvidenceForRec(h){
  setPanel(`<div class="ev-head">${kind("drug_rec")}<span class="ev-title">${h.drug}</span></div>
    ${badge()}
    <div class="ev-row"><span class="k">為什麼推薦(機制)</span>${h.rationale}</div>
    <div class="ev-row"><span class="k">犬反應證據</span>${h.evidence}</div>
    <div class="ev-row"><span class="k">為什麼挑這個老藥(結構/標靶類比)</span>${(CUR.analogy&&CUR.analogy.basis)||"—"}</div>
    <div class="ev-row"><span class="k">證據鏈</span>${CUR.pet_disease_zh} → ${CUR.human_disease} → <b>${CUR.driver.gene} ${CUR.driver.variant||""}</b>(犬 ${CUR.driver.freq||""})→ ${CUR.pathway} → <b>${h.drug}</b></div>
    ${citeHTML(h.citation)}`);
}
function showSibling(d){
  setPanel(`<div class="ev-head">${kind("sibling")}<span class="ev-title">${d.label}</span></div>
    <div class="ev-row"><span class="k">與推薦藥的關係</span>${d.relation||""}</div>
    <div class="ev-row"><span class="k">這個已知藥原本用於</span>${d.known_for||""}</div>
    <div class="ev-row"><span class="k">老藥新用邏輯</span>${d.basis||(CUR.analogy&&CUR.analogy.basis)||""}</div>
    ${d.broken?'<div class="ev-row" style="color:var(--refuted)">⚠️ 此類比在犬身上斷裂:缺對應 driver。</div>':''}`);
}
function showIndication(d){
  setPanel(`<div class="ev-head">${kind("indication")}<span class="ev-title">${d.label}</span></div>
    <div class="ev-row">這是上游「已知相似老藥」原本的適應症;推薦藥即由此結構/標靶類比延伸而來——這就是「老藥新用」。</div>`);
}
function showDriver(){
  const dv=CUR.driver;
  setPanel(`<div class="ev-head">${kind("driver")}<span class="ev-title">${dv.gene} ${dv.variant||""}</span></div>
    ${badge()}
    <div class="ev-row"><span class="k">犬腫瘤頻率</span><b>${dv.freq||"—"}</b></div>
    <div class="ev-row"><span class="k">說明</span>${dv.note||""}</div>
    ${citeHTML(dv.citation)}`);
}
function showEvidenceNode(d){
  setPanel(`<div class="ev-head">${kind("evidence")}<span class="ev-title">已發表的犬證據</span></div>
    ${badge()}
    <div class="ev-row">${d.text||""}</div>
    ${citeHTML(d.citation)}`);
}
function showModelCand(m){
  const pct=(m.cellline_pct!=null)?m.cellline_pct+"%":"無資料";
  setPanel(`<div class="ev-head">${kind("drug_model")}<span class="ev-title">${m.drug}</span></div>
    <div class="ev-row"><span class="k">來源</span>MRDDA 圖模型初篩候選(非證據鏈首選)。</div>
    <div class="ev-row"><span class="k">細胞株活性(GDSC/PRISM)</span><b>${pct}</b>${m.selective?" · 具選擇性":""}</div>
    ${m.note?`<div class="ev-row"><span class="k">註</span>${m.note}</div>`:""}
    <div class="ev-row" style="color:var(--muted)">模型多半挑出廣譜化療;真正的『為什麼對寵物有效』由上方 driver 證據鏈提供。</div>`);
}
function showRefuted(){
  setPanel(`<div class="ev-head">${kind("refuted")}<span class="ev-title">✗ 不轉移</span></div>
    ${badge()}
    <div class="ev-row"><span class="k">閘門判定</span>${CUR.refuted_note}</div>
    <div class="ev-row"><span class="k">為什麼這很重要</span>這個案例證明方法會「主動證偽」——人類有的 driver,犬沒有就不推薦,不是事後合理化。</div>
    ${citeHTML(CUR.driver.citation)}`);
}
function showDiseaseInfo(d){
  setPanel(`<div class="ev-head">${kind(d.ntype)}<span class="ev-title">${d.label}</span></div>
    <div class="ev-row"><span class="k">${d.ntype==="pet"?"物種":"MeSH"}</span>${d.sub||d.species||""}</div>
    <div class="ev-row">${CUR.storyline}</div>`);
}
function showPathway(d){
  setPanel(`<div class="ev-head">${kind("pathway")}<span class="ev-title">${d.label}</span></div>
    <div class="ev-row">由 <b>${CUR.driver.gene}</b> 驅動的訊號通路;候選藥即作用於此通路。</div>`);
}

function sourcesBlock(){
  if(!CUR||!CUR.sources||!CUR.sources.length) return "";
  return `<div class="sources-block"><h4>本疾病所有文獻來源</h4>`+
    CUR.sources.map(s=>`<div class="cite">📄 <a href="${s.url}" target="_blank" rel="noopener">${s.title}</a></div>`).join("")+
    `</div>`;
}
