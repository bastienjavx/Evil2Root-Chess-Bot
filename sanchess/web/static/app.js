"use strict";
/* San-o1 console — frontend autonome (aucune dépendance externe).
   Toute la logique d'échecs (légalité, SAN, fin de partie) est côté serveur. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const PIECES = { K:"♔",Q:"♕",R:"♖",B:"♗",N:"♘",P:"♙",
                 k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟" };
const FILES = "abcdefgh";
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

function conn(ok){ const c = $("#conn"); c.classList.toggle("ok", ok); }

async function api(path, opts){
  try{
    const r = await fetch(path, opts);
    conn(r.ok);
    return await r.json();
  }catch(e){ conn(false); throw e; }
}
function esc(s){ return String(s??"").replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function clamp(x, a, b){ return Math.max(a, Math.min(b, Number(x)||0)); }
function pct(x){ return `${Math.round(clamp(x, 0, 1)*100)}%`; }
function fmtInt(x){ return Number(x||0).toLocaleString("fr-FR"); }
function fmtFloat(x, d=1){ return Number(x||0).toLocaleString("fr-FR", {maximumFractionDigits:d, minimumFractionDigits:d}); }
function sideName(s){ return s==="white" ? "Blancs" : "Noirs"; }
async function copyText(txt, msg){
  try{
    await navigator.clipboard.writeText(txt);
    const st = $("#play-status");
    const old = st.textContent;
    st.textContent = msg || "Copié";
    setTimeout(()=>{ if(st.textContent === (msg || "Copié")) st.textContent = old; }, 1400);
  }catch(e){
    $("#play-status").textContent = "Copie impossible dans ce navigateur.";
  }
}

/* ---------------- Rendu de l'échiquier ---------------- */
function fenToBoard(fen){
  const rows = fen.split(" ")[0].split("/");
  const g = [];
  for(const row of rows){
    const r = [];
    for(const ch of row){
      if(/\d/.test(ch)) for(let i=0;i<+ch;i++) r.push(null);
      else r.push(ch);
    }
    g.push(r);
  }
  return g; // g[rankFromTop 0..7][file 0..7], rankFromTop 0 = rang 8
}

function renderBoard(el, fen, opt={}){
  const {flip=false, lastMove=null, sel=null, targets=[], checkSq=null, onClick=null} = opt;
  const g = fenToBoard(fen);
  el.innerHTML = "";
  const targetSet = new Set(targets.map(t=>t.to));
  for(let vr=0; vr<8; vr++){
    for(let vf=0; vf<8; vf++){
      const r = flip ? 7-vr : vr;          // rankFromTop affiché
      const f = flip ? 7-vf : vf;
      const piece = g[r][f];
      const sqName = FILES[f] + (8-r);
      const d = document.createElement("div");
      const dark = (r+f)%2===1;
      d.className = "sq " + (dark?"dark":"light");
      if(piece) d.classList.add(piece===piece.toUpperCase()?"w":"b");
      if(sqName===sel) d.classList.add("sel");
      if(lastMove && (sqName===lastMove.slice(0,2)||sqName===lastMove.slice(2,4))) d.classList.add("last");
      if(sqName===checkSq) d.classList.add("check");
      if(targetSet.has(sqName)){ d.classList.add(piece?"cap":""); const dot=document.createElement("div"); dot.className="dot"; d.appendChild(dot); }
      if(piece){ const p=document.createElement("span"); p.className="piece"; p.textContent=PIECES[piece]; d.appendChild(p); }
      d.dataset.sq = sqName;
      if(onClick) d.addEventListener("click", ()=>onClick(sqName, piece));
      el.appendChild(d);
    }
  }
}

function setEval(el, winProb){ // winProb du point de vue des Blancs
  el.style.height = Math.max(2, Math.min(98, winProb*100)).toFixed(1) + "%";
}
function fmtCp(cp){ if(cp===null||cp===undefined) return "—"; const s=(cp/100).toFixed(2); return cp>0?"+"+s:s; }
function statusLabel(result, reason){
  if(result && result!=="*") return result==="1-0"?"Blancs gagnent":result==="0-1"?"Noirs gagnent":"Nulle";
  if(reason==="max_plies") return "limite atteinte";
  if(reason==="no_move") return "aucun coup";
  return "en attente";
}

/* ---------------- Modèles (selects + table) ---------------- */
let MODELS = [];
async function loadModels(){
  const data = await api("/api/models");
  MODELS = data.models;
  $("#device-badge").textContent = data.device;
  const opts = '<option value="latest">latest.pt (courant)</option>' +
    MODELS.filter(m=>!m.is_latest).map(m=>`<option value="${esc(m.name)}">${esc(m.name)}${m.step!=null?` · step ${m.step}`:""}</option>`).join("");
  for(const id of ["#play-model","#watch-white","#watch-black","#analyze-model"]) $(id).innerHTML = opts;
  renderModelsTable();
}
function renderModelsTable(){
  const tb = $("#models-table tbody");
  const summary = $("#models-summary");
  if(summary){
    const total = MODELS.reduce((a,m)=>a+(m.size||0),0)/1e6;
    summary.innerHTML = card("Checkpoints", fmtInt(MODELS.length))+
      card("Latest step", MODELS.find(m=>m.is_latest)?.step ?? "—")+
      card("Taille totale", total.toFixed(1)+" Mo");
  }
  if(!MODELS.length){ tb.innerHTML='<tr><td colspan="8" class="empty">Aucun checkpoint.</td></tr>'; return; }
  const q = ($("#models-filter")?.value || "").trim().toLowerCase();
  const list = MODELS.filter(m=>!q || m.name.toLowerCase().includes(q) || String(m.step??"").includes(q));
  if(!list.length){ tb.innerHTML='<tr><td colspan="8" class="empty">Aucun résultat.</td></tr>'; return; }
  tb.innerHTML = list.map(m=>`<tr>
    <td>${esc(m.name)} ${m.is_latest?'<span class="badge latest">latest</span>':''}</td>
    <td>${m.step??"—"}</td>
    <td>${m.channels?`${m.channels}×${m.blocks}${m.error?" ⚠":""}`:"—"}</td>
    <td>${esc(m.policy_head??"—")}</td>
    <td>${esc(m.activation??"—")}</td>
    <td>${m.size_mb} Mo</td>
    <td>${new Date(m.mtime*1000).toLocaleString()}</td>
    <td><button class="btn" data-test="${esc(m.name)}">Tester</button></td>
  </tr>`).join("");
  $$("#models-table button[data-test]").forEach(b=>b.addEventListener("click",()=>{
    $("#play-model").value = b.dataset.test; switchTab("play"); newGame();
  }));
}

/* ---------------- Réflexion (table candidats) ---------------- */
function renderThinking(el, top){
  const tb = el.querySelector("tbody");
  if(!top || !top.length){ tb.innerHTML=""; return; }
  const max = Math.max(...top.map(t=>t.visits), 1);
  tb.innerHTML = top.map((t,i)=>`<tr>
    <td class="rank">${i+1}</td>
    <td class="san">${esc(t.san)}</td>
    <td>${fmtCp(t.cp)}</td>
    <td>${fmtInt(t.visits)}</td>
    <td style="width:40%"><div class="bar" style="width:${(t.visits/max*100).toFixed(0)}%"></div></td>
  </tr>`).join("");
}
function pushMoveList(el, list){
  let html=""; for(let i=0;i<list.length;i+=2){
    html+=`<span class="num">${i/2+1}.</span> <button class="mv" data-ply="${i+1}">${esc(list[i]||"")}</button> <button class="mv" data-ply="${i+2}">${esc(list[i+1]||"")}</button> `;
  }
  el.innerHTML = html; el.scrollTop = el.scrollHeight;
}

/* ================= JOUER (humain vs modèle) ================= */
const play = { startFen:null, moves:[], sans:[], side:"white", flip:false, legal:[], sel:null,
               fen:new Chess0(), busy:false };
// État minimal : on garde le FEN courant renvoyé par le serveur.
function Chess0(){ this.fen = START_FEN; }

function budget(id){ const [k,v]=$(id).value.split(":"); return k==="nodes"?{nodes:+v}:{movetime:+v}; }

async function newGame(){
  play.moves=[]; play.sans=[]; play.sel=null; play.busy=false;
  play.side = $("#play-side").value; play.flip = play.side==="black";
  play.fen.fen = START_FEN;
  pushMoveList($("#play-movelist"), []);
  renderThinking($("#play-thinking"), []);
  $("#play-cp").textContent="—"; $("#play-wp").textContent="50%"; $("#play-nps").textContent="0";
  $("#play-clock").textContent="0.0s"; $("#play-depth").textContent="0 nœud";
  setEval($("#play-eval"),.5);
  await refreshLegal();
  $("#play-status").textContent = "À vous de jouer.";
  drawPlay();
  if(play.side==="black") await modelMove();
}

async function refreshLegal(){
  const q = new URLSearchParams({moves: play.moves.join(",")});
  const d = await api("/api/legal?"+q.toString());
  play.legal = d.legal; play.fen.fen = d.fen; play.turn = d.turn;
  play.checkSq = d.in_check ? kingSquare(d.fen, d.turn) : null;
  play.over = d.is_game_over; play.result = d.result;
  updatePlayMeta();
  return d;
}
function kingSquare(fen, turn){
  const g = fenToBoard(fen); const k = turn==="white"?"K":"k";
  for(let r=0;r<8;r++)for(let f=0;f<8;f++) if(g[r][f]===k) return FILES[f]+(8-r);
  return null;
}
function drawPlay(){
  const last = play.moves.length? play.moves[play.moves.length-1]:null;
  const targets = play.sel? play.legal.filter(m=>m.from===play.sel):[];
  renderBoard($("#play-board"), play.fen.fen, {
    flip:play.flip, lastMove:last, sel:play.sel, targets, checkSq:play.checkSq,
    onClick:onPlayClick });
  $("#play-fen-box").value = play.fen.fen;
}
function updatePlayMeta(){
  $("#play-turn").textContent = `Trait ${sideName(play.turn || "white")}`;
  $("#play-current-model").textContent = $("#play-model").value || "latest";
  $("#play-fen-box").value = play.fen.fen;
}
async function onPlayClick(sq, piece){
  if(play.busy||play.over) return;
  if(play.turn!==play.side) return;            // pas votre tour
  if(play.sel){
    const mv = play.legal.find(m=>m.from===play.sel && m.to===sq);
    if(mv){ await applyHuman(mv); return; }
  }
  // sélection d'une pièce à soi qui a des coups
  const has = play.legal.some(m=>m.from===sq);
  play.sel = has? sq : null;
  drawPlay();
}
async function applyHuman(mv){
  // promotion : choisir la dame par défaut s'il y a ambiguïté
  let uci = mv.uci;
  const promos = play.legal.filter(m=>m.from===mv.from && m.to===mv.to && m.uci.length===5);
  if(promos.length){ const q = promos.find(p=>p.uci.endsWith("q")); uci = (q||promos[0]).uci; }
  play.moves.push(uci); play.sans.push(sanFor(mv,uci)); play.sel=null;
  await refreshLegal(); pushMoveList($("#play-movelist"), play.sans); drawPlay();
  if(play.over){ endPlay(); return; }
  await modelMove();
}
function sanFor(mv, uci){ return uci.length===5 ? mv.san.replace(/=?[QRBN]?[+#]?$/, "")+"=Q" : mv.san; }

async function modelMove(){
  if(play.over) return;
  play.busy=true; $("#play-status").textContent = "Le modèle réfléchit…";
  const b = budget("#play-budget");
  let res;
  try{ res = await api("/api/move", {method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({moves:play.moves, model:$("#play-model").value, ...b})}); }
  catch(e){ $("#play-status").textContent="Erreur de connexion au moteur."; play.busy=false; return; }
  if(res.error){ $("#play-status").textContent=res.error; play.busy=false; return; }
  renderThinking($("#play-thinking"), res.top_moves);
  // éval du point de vue des Blancs
  const wp = play.turn==="white"? res.win_prob : 1-res.win_prob;
  setEval($("#play-eval"), wp);
  const cpWhite = play.turn==="white"? res.cp : -res.cp;
  $("#play-cp").textContent = fmtCp(cpWhite) + (res.step?`  ·  step ${res.step}`:"");
  $("#play-wp").textContent = pct(wp);
  $("#play-nps").textContent = fmtInt(res.nps);
  $("#play-clock").textContent = `${(res.elapsed||0).toFixed(1)}s`;
  $("#play-depth").textContent = `${fmtInt(res.root_visits)} nœuds`;
  if(res.bestmove){
    play.moves.push(res.bestmove); play.sans.push(res.bestmove_san);
    await refreshLegal(); pushMoveList($("#play-movelist"), play.sans);
  }
  play.busy=false; drawPlay();
  $("#play-status").textContent = play.over? "" : "À vous de jouer.";
  if(play.over) endPlay();
}
function endPlay(){
  const r = play.result;
  const txt = r==="1-0"?"Victoire des Blancs":r==="0-1"?"Victoire des Noirs":"Partie nulle";
  $("#play-status").textContent = `Fin de partie · ${txt} (${r})`;
}

async function copyPlayPgn(){
  const white = play.side==="white" ? "Humain" : ($("#play-model").value || "San-o1");
  const black = play.side==="black" ? "Humain" : ($("#play-model").value || "San-o1");
  const d = await api("/api/pgn", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({moves:play.moves, white, black, result:play.result})});
  if(d.error){ $("#play-status").textContent=d.error; return; }
  await copyText(d.pgn, "PGN copié");
}

/* ================= ANALYSER (position libre) ================= */
const analyze = { flip:false, fen:START_FEN };

function drawAnalyze(fen=analyze.fen, lastMove=null){
  analyze.fen = fen || START_FEN;
  renderBoard($("#analyze-board"), analyze.fen, {flip:analyze.flip, lastMove});
  $("#analyze-fen").value = analyze.fen;
}

async function runAnalyze(){
  const fen = $("#analyze-fen").value.trim() || START_FEN;
  $("#analyze-status").textContent = "Analyse en cours…";
  renderThinking($("#analyze-thinking"), []);
  let res;
  try{
    res = await api("/api/analyze", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({fen, model:$("#analyze-model").value, ...budget("#analyze-budget"), top_k:8})});
  }catch(e){
    $("#analyze-status").textContent = "Erreur de connexion au moteur.";
    return;
  }
  if(res.error){ $("#analyze-status").textContent = res.error; return; }
  drawAnalyze(res.fen, res.bestmove);
  const wp = res.turn==="white" ? res.win_prob : 1-res.win_prob;
  setEval($("#analyze-eval"), wp);
  $("#analyze-cp").textContent = fmtCp(res.turn==="white" ? res.cp : -res.cp);
  $("#analyze-best").textContent = res.bestmove_san ? `meilleur coup ${res.bestmove_san}` : "aucun coup";
  $("#analyze-wp").textContent = pct(wp);
  $("#analyze-nps").textContent = fmtInt(res.nps);
  $("#analyze-time").textContent = `${(res.elapsed||0).toFixed(1)}s`;
  renderThinking($("#analyze-thinking"), res.top_moves);
  $("#analyze-status").textContent = `${fmtInt(res.root_visits)} visites · ${sideName(res.turn)} au trait`;
}

/* ================= REGARDER (selfplay streaming) ================= */
let watchWs=null;
const watch = {state:"idle", moves:[], cp:[], wp:[], nps:[], visits:[]};

function setWatchState(state, label){
  watch.state = state;
  $("#watch-state").textContent = label || state;
  $("#watch-start").disabled = state==="connecting" || state==="running" || state==="stopping";
  $("#watch-stop").disabled = !(state==="connecting" || state==="running");
}
function resetWatchCharts(){
  for(const id of ["#watch-chart-cp","#watch-chart-wp","#watch-chart-nps","#watch-chart-visits"]){
    lineChart($(id), [], [], "");
  }
}
function drawWatchCharts(){
  const xs = watch.cp.map((_,i)=>i+1);
  lineChart($("#watch-chart-cp"), xs, watch.cp, "");
  lineChart($("#watch-chart-wp"), xs, watch.wp, "v", {unit:"%", fixed:0, yMin:0, yMax:100});
  lineChart($("#watch-chart-nps"), xs, watch.nps, "o", {fixed:0});
  lineChart($("#watch-chart-visits"), xs, watch.visits, "p", {fixed:0});
}
function resetWatchUi(){
  watch.moves=[]; watch.cp=[]; watch.wp=[]; watch.nps=[]; watch.visits=[];
  pushMoveList($("#watch-movelist"), []);
  renderThinking($("#watch-thinking"), []);
  renderBoard($("#watch-board"), START_FEN, {});
  $("#watch-ply").textContent = "0 demi-coup";
  $("#watch-result").textContent = "en attente";
  $("#watch-cp").textContent = "—";
  $("#watch-nps").textContent = "0";
  $("#watch-visits").textContent = "0";
  $("#watch-wp").textContent = "50%";
  $("#watch-time").textContent = "0.0s";
  setEval($("#watch-eval"), .5);
  resetWatchCharts();
}
function watchStart(){
  watchStop("reset");
  resetWatchUi();
  setWatchState("connecting", "connexion");
  $("#watch-result").textContent = "en direct";
  $("#watch-status").textContent = "Connexion au flux self-play…";
  const proto = location.protocol==="https:"?"wss:":"ws:";
  watchWs = new WebSocket(`${proto}//${location.host}/ws/selfplay`);
  watchWs.onopen = ()=>{
    conn(true);
    setWatchState("running", "en direct");
    watchWs.send(JSON.stringify({
      model_white:$("#watch-white").value,
      model_black:$("#watch-black").value,
      nodes:+$("#watch-nodes").value,
      max_plies:+$("#watch-max-plies").value
    }));
  };
  watchWs.onmessage = (ev)=>{
    let m;
    try{ m = JSON.parse(ev.data); }
    catch(e){ $("#watch-status").textContent = "Message WebSocket illisible."; return; }
    if(m.type==="start"){
      $("#watch-status").textContent=`Blancs: ${m.white} · Noirs: ${m.black} · limite ${m.max_plies} demi-coups`;
    } else if(m.type==="move"){
      renderBoard($("#watch-board"), m.fen, {lastMove:m.uci});
      const moverWasWhite = (m.turn==="black");
      const wpWhite = moverWasWhite ? m.win_prob : 1-m.win_prob;
      const cpWhite = moverWasWhite ? m.cp : -m.cp;
      setEval($("#watch-eval"), wpWhite);
      $("#watch-cp").textContent = fmtCp(cpWhite);
      $("#watch-nps").textContent = fmtInt(m.nps);
      $("#watch-visits").textContent = fmtInt(m.visits);
      $("#watch-wp").textContent = pct(wpWhite);
      $("#watch-time").textContent = `${fmtFloat(m.elapsed||0, 1)}s`;
      $("#watch-ply").textContent = `${m.ply} demi-coup${m.ply>1?"s":""}`;
      renderThinking($("#watch-thinking"), m.top_moves);
      watch.moves.push(m.san);
      watch.cp.push(cpWhite);
      watch.wp.push(Math.round(wpWhite*100));
      watch.nps.push(m.nps||0);
      watch.visits.push(m.visits||0);
      pushMoveList($("#watch-movelist"), watch.moves);
      drawWatchCharts();
    } else if(m.type==="gameover"){
      $("#watch-status").textContent = `Fin · ${statusLabel(m.result, m.reason)} · ${m.plies} demi-coups`;
      $("#watch-result").textContent = statusLabel(m.result, m.reason);
      watchStop("ended");
    } else if(m.type==="error"){
      $("#watch-status").textContent = "Erreur : " + (m.error || "inconnue");
      watchStop("error");
    }
  };
  watchWs.onclose = ()=>{
    if(watch.state==="running" || watch.state==="connecting") setWatchState("idle", "arrêté");
    if(watchWs) watchWs = null;
  };
  watchWs.onerror = ()=>{
    conn(false);
    $("#watch-status").textContent = "Erreur de connexion WebSocket.";
  };
}
function watchStop(mode="user"){
  const ws = watchWs;
  watchWs = null;
  if(ws){ try{ ws.close(); }catch(e){} }
  if(mode==="ended") setWatchState("ended", "terminé");
  else if(mode==="error") setWatchState("error", "erreur");
  else if(mode==="reset") setWatchState("idle", "prêt");
  else {
    setWatchState("stopping", "arrêt…");
    setTimeout(()=>{ if(!watchWs && watch.state==="stopping") setWatchState("idle", "arrêté"); }, 150);
  }
}

/* ================= ENTRAINEMENT ================= */
function lineChart(svg, xs, ys, klass="", opt={}){
  svg.innerHTML="";
  if(!ys || ys.length<2){ svg.innerHTML='<text x="300" y="100" text-anchor="middle">aucune donnée</text>'; return; }
  const vb = svg.viewBox && svg.viewBox.baseVal ? svg.viewBox.baseVal : null;
  const W = vb?.width || 600, H = vb?.height || 200, pad=30;
  const xmin=xs[0], xmax=xs[xs.length-1]||1;
  const rawMin=Math.min(...ys), rawMax=Math.max(...ys);
  const spread = Math.max(1e-9, rawMax-rawMin);
  const ymin=opt.yMin!==undefined ? opt.yMin : rawMin - spread*.08;
  const ymax=opt.yMax!==undefined ? opt.yMax : rawMax + spread*.08;
  const sx=v=>pad+(W-2*pad)*((v-xmin)/((xmax-xmin)||1));
  const sy=v=>pad+(H-2*pad)*(1-((v-ymin)/((ymax-ymin)||1)));
  const fmt = v => opt.fixed===0 ? Math.round(v).toString() : Number(v).toFixed(opt.fixed ?? 3);
  let g="";
  for(let i=0;i<=4;i++){
    const y=pad+(H-2*pad)*i/4;
    g+=`<line class="grid" x1="${pad}" y1="${y}" x2="${W-pad}" y2="${y}"/>`;
  }
  const pts = xs.map((x,i)=>`${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(" ");
  svg.innerHTML = g +
    `<polyline class="area ${klass}" points="${pad},${H-pad} ${pts} ${W-pad},${H-pad}"/>`+
    `<polyline class="line ${klass}" points="${pts}"/>`+
    `<circle class="point ${klass}" cx="${sx(xs[xs.length-1]).toFixed(1)}" cy="${sy(ys[ys.length-1]).toFixed(1)}" r="3"/>`+
    `<text x="${pad}" y="16">${fmt(rawMax)}${opt.unit||""}</text>`+
    `<text x="${pad}" y="${H-7}">${fmt(rawMin)}${opt.unit||""}</text>`+
    `<text x="${W-pad}" y="16" text-anchor="end">${fmt(ys[ys.length-1])}${opt.unit||""}</text>`+
    `<text x="${W-pad}" y="${H-7}" text-anchor="end">#${xmax}</text>`;
}
async function loadTraining(){
  const d = await api("/api/training?points=400");
  const cards = $("#train-cards"); cards.innerHTML="";
  const p=d.pretrain, o=d.online;
  if(p.steps.length){
    cards.innerHTML += card("Pretrain step", p.steps[p.steps.length-1]);
    cards.innerHTML += card("Policy loss", p.policy[p.policy.length-1].toFixed(4));
    cards.innerHTML += card("Value loss", p.value[p.value.length-1].toFixed(4));
  }
  if(o.steps.length){
    cards.innerHTML += card("Online step", o.steps[o.steps.length-1]);
    cards.innerHTML += card("Online loss", o.loss[o.loss.length-1].toFixed(4));
    cards.innerHTML += card("Buffer", fmtInt(o.buffer[o.buffer.length-1]));
  }
  if(!cards.innerHTML) cards.innerHTML='<div class="empty">Aucun log d\'entraînement détecté dans data/.</div>';
  $("#train-logs").textContent = [d.pretrain_log && `pretrain: ${d.pretrain_log}`, d.online_log && `online: ${d.online_log}`].filter(Boolean).join(" · ");
  lineChart($("#chart-policy"), p.steps, p.policy, "");
  lineChart($("#chart-value"), p.steps, p.value, "v");
  lineChart($("#chart-online"), o.steps, o.loss, "o");
}
function card(k,v,sub=""){ return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div>${sub}</div>`; }

/* ================= SYSTEME ================= */
const sysHist = {x:[], gpu:[], mem:[], t:0};
async function loadSystem(){
  const d = await api("/api/system");
  const gpu = $("#sys-gpu");
  gpu.innerHTML = d.gpu.length? d.gpu.map(g=>{
    const mem = g.mem_total_mb? (g.mem_used_mb/g.mem_total_mb*100):0;
    return `<div class="card"><div class="k">${g.name}</div>
      <div class="v">${g.util_pct??"—"}%<span class="muted"> GPU</span></div>
      <div class="muted">${g.mem_used_mb??"?"} / ${g.mem_total_mb??"?"} Mo · ${g.temp_c??"?"}°C · ${g.power_w??"?"} W</div>
      <div class="gauge"><span style="width:${mem.toFixed(0)}%"></span></div></div>`;
  }).join("") : '<div class="empty">Pas de GPU NVIDIA détecté.</div>';

  $("#sys-services").innerHTML = d.services.length? d.services.map(s=>
    `<span class="chip ${s.active?"on":"off"}"><span class="led"></span>${s.name} · ${s.state}</span>`).join("")
    : '<div class="empty">systemctl indisponible.</div>';

  const da=d.data;
  $("#sys-data").innerHTML =
    card("Replay buffer", da.replay_buffer_files+" fichiers", `<div class="muted">${da.replay_buffer_mb} Mo</div>`)+
    card("Shards", fmtInt(da.shard_files));

  const tr = d.training || {};
  $("#sys-training").innerHTML =
    (tr.pretrain ? card("Pretrain", `step ${fmtInt(tr.pretrain.step)}`, `<div class="muted">policy ${tr.pretrain.policy} · value ${tr.pretrain.value}</div>`) : "")+
    (tr.online ? card("Online", `step ${fmtInt(tr.online.step)}`, `<div class="muted">loss ${tr.online.loss} · buffer ${fmtInt(tr.online.buffer)}</div>`) : "") ||
    '<div class="empty">Aucun entraînement récent détecté.</div>';

  $("#sys-procs").innerHTML = d.processes.length? d.processes.map(p=>
    `<span class="chip on"><span class="led"></span>${esc(p.module)} · pid ${p.pid}</span>`).join("")
    : '<div class="empty">Aucun process sanchess en cours.</div>';

  sysHist.t += 1;
  const firstGpu = d.gpu[0];
  sysHist.x.push(sysHist.t);
  sysHist.gpu.push(firstGpu?.util_pct ?? 0);
  sysHist.mem.push(firstGpu?.mem_total_mb ? firstGpu.mem_used_mb / firstGpu.mem_total_mb * 100 : 0);
  for(const k of ["x","gpu","mem"]) if(sysHist[k].length > 80) sysHist[k].shift();
  lineChart($("#sys-chart-gpu"), sysHist.x, sysHist.gpu, "", {unit:"%", fixed:0, yMin:0, yMax:100});
  lineChart($("#sys-chart-mem"), sysHist.x, sysHist.mem, "v", {unit:"%", fixed:0, yMin:0, yMax:100});
}

/* ================= Onglets + auto-refresh ================= */
let timers={};
function switchTab(name){
  $$(".tab").forEach(t=>t.classList.toggle("active", t.dataset.tab===name));
  $$(".panel").forEach(p=>p.classList.toggle("active", p.id===name));
  for(const k in timers){ clearInterval(timers[k]); } timers={};
  if(name==="models") loadModels();
  if(name==="analyze") drawAnalyze($("#analyze-fen").value || play.fen.fen || START_FEN);
  if(name==="train"){ loadTraining(); if($("#train-auto").checked) timers.t=setInterval(loadTraining,4000); }
  if(name==="system"){ loadSystem(); if($("#sys-auto").checked) timers.s=setInterval(loadSystem,3000); }
  if(name==="watch"){ /* prêt */ }
}

/* ================= Init ================= */
function init(){
  $$(".tab").forEach(t=>t.addEventListener("click",()=>switchTab(t.dataset.tab)));
  $("#play-new").addEventListener("click", newGame);
  $("#play-flip").addEventListener("click", ()=>{ play.flip=!play.flip; drawPlay(); });
  $("#play-copy-fen").addEventListener("click", ()=>copyText(play.fen.fen, "FEN copié"));
  $("#play-copy-pgn").addEventListener("click", copyPlayPgn);
  $("#play-undo").addEventListener("click", async ()=>{
    if(play.busy||play.moves.length<2) return;
    play.moves.splice(-2); play.sans.splice(-2); play.sel=null;
    await refreshLegal(); pushMoveList($("#play-movelist"),play.sans); drawPlay();
  });
  $("#play-side").addEventListener("change", newGame);
  $("#play-model").addEventListener("change", updatePlayMeta);
  $("#analyze-run").addEventListener("click", runAnalyze);
  $("#analyze-startpos").addEventListener("click", ()=>{ drawAnalyze(START_FEN); setEval($("#analyze-eval"),.5); });
  $("#analyze-flip").addEventListener("click", ()=>{ analyze.flip=!analyze.flip; drawAnalyze(analyze.fen); });
  $("#analyze-fen").addEventListener("keydown", e=>{ if((e.ctrlKey||e.metaKey) && e.key==="Enter") runAnalyze(); });
  $("#watch-start").addEventListener("click", watchStart);
  $("#watch-stop").addEventListener("click", watchStop);
  $("#models-refresh").addEventListener("click", loadModels);
  $("#models-filter").addEventListener("input", renderModelsTable);
  $("#train-auto").addEventListener("change", ()=>switchTab("train"));
  $("#sys-auto").addEventListener("change", ()=>switchTab("system"));
  document.addEventListener("keydown", e=>{
    if(e.target.matches("input,textarea,select")) return;
    if(e.key==="n") newGame();
    if(e.key==="f"){ play.flip=!play.flip; drawPlay(); }
    if(e.key==="u") $("#play-undo").click();
  });
  drawAnalyze(START_FEN);
  loadModels().then(newGame);
}
document.addEventListener("DOMContentLoaded", init);
