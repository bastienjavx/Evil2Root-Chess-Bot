"use strict";
/* San-o1 console — frontend autonome (aucune dépendance externe).
   Toute la logique d'échecs (légalité, SAN, fin de partie) est côté serveur. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const PIECES = { K:"♔",Q:"♕",R:"♖",B:"♗",N:"♘",P:"♙",
                 k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟" };
const FILES = "abcdefgh";

function conn(ok){ const c = $("#conn"); c.classList.toggle("ok", ok); }

async function api(path, opts){
  try{
    const r = await fetch(path, opts);
    conn(r.ok);
    return await r.json();
  }catch(e){ conn(false); throw e; }
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

/* ---------------- Modèles (selects + table) ---------------- */
let MODELS = [];
async function loadModels(){
  const data = await api("/api/models");
  MODELS = data.models;
  $("#device-badge").textContent = data.device;
  const opts = '<option value="latest">latest.pt (courant)</option>' +
    MODELS.filter(m=>!m.is_latest).map(m=>`<option value="${m.name}">${m.name}${m.step!=null?` · step ${m.step}`:""}</option>`).join("");
  for(const id of ["#play-model","#watch-white","#watch-black"]) $(id).innerHTML = opts;
  renderModelsTable();
}
function renderModelsTable(){
  const tb = $("#models-table tbody");
  if(!MODELS.length){ tb.innerHTML='<tr><td colspan="8" class="empty">Aucun checkpoint.</td></tr>'; return; }
  tb.innerHTML = MODELS.map(m=>`<tr>
    <td>${m.name} ${m.is_latest?'<span class="badge latest">latest</span>':''}</td>
    <td>${m.step??"—"}</td>
    <td>${m.channels?`${m.channels}×${m.blocks}${m.error?" ⚠":""}`:"—"}</td>
    <td>${m.policy_head??"—"}</td>
    <td>${m.activation??"—"}</td>
    <td>${m.size_mb} Mo</td>
    <td>${new Date(m.mtime*1000).toLocaleString()}</td>
    <td><button class="btn" data-test="${m.name}">Tester</button></td>
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
  tb.innerHTML = top.map(t=>`<tr>
    <td class="san">${t.san}</td>
    <td>${fmtCp(t.cp)}</td>
    <td>${t.visits}</td>
    <td style="width:40%"><div class="bar" style="width:${(t.visits/max*100).toFixed(0)}%"></div></td>
  </tr>`).join("");
}
function pushMoveList(el, list){
  let html=""; for(let i=0;i<list.length;i+=2){
    html+=`<span class="num">${i/2+1}.</span> <span class="mv">${list[i]||""}</span> <span class="mv">${list[i+1]||""}</span> `;
  }
  el.innerHTML = html; el.scrollTop = el.scrollHeight;
}

/* ================= JOUER (humain vs modèle) ================= */
const play = { startFen:null, moves:[], sans:[], side:"white", flip:false, legal:[], sel:null,
               fen:new Chess0(), busy:false };
// État minimal : on garde le FEN courant renvoyé par le serveur.
function Chess0(){ this.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"; }

function budget(id){ const [k,v]=$(id).value.split(":"); return k==="nodes"?{nodes:+v}:{movetime:+v}; }

async function newGame(){
  play.moves=[]; play.sans=[]; play.sel=null; play.busy=false;
  play.side = $("#play-side").value; play.flip = play.side==="black";
  play.fen.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  pushMoveList($("#play-movelist"), []);
  renderThinking($("#play-thinking"), []);
  $("#play-cp").textContent="—"; setEval($("#play-eval"),.5);
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

/* ================= REGARDER (selfplay streaming) ================= */
let watchWs=null, watchMoves=[];
function watchStart(){
  watchStop();
  watchMoves=[]; pushMoveList($("#watch-movelist"), []);
  renderBoard($("#watch-board"), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", {});
  const proto = location.protocol==="https:"?"wss:":"ws:";
  watchWs = new WebSocket(`${proto}//${location.host}/ws/selfplay`);
  $("#watch-start").disabled=true; $("#watch-stop").disabled=false;
  watchWs.onopen = ()=>{ conn(true); watchWs.send(JSON.stringify({
    model_white:$("#watch-white").value, model_black:$("#watch-black").value,
    nodes:+$("#watch-nodes").value, max_plies:300 })); };
  watchWs.onmessage = (ev)=>{
    const m = JSON.parse(ev.data);
    if(m.type==="start"){ $("#watch-status").textContent=`Blancs: ${m.white} · Noirs: ${m.black}`; }
    else if(m.type==="move"){
      renderBoard($("#watch-board"), m.fen, {lastMove:m.uci});
      // m.value est du point de vue du joueur QUI VIENT DE JOUER... non : analyze
      // renvoie la valeur côté trait AVANT de jouer. m.turn = trait après le coup.
      const moverWasWhite = (m.turn==="black");
      const wp = moverWasWhite? m.win_prob : 1-m.win_prob;
      setEval($("#watch-eval"), wp);
      const cpW = moverWasWhite? m.cp : -m.cp;
      $("#watch-cp").textContent = fmtCp(cpW); $("#watch-nps").textContent = m.nps;
      renderThinking($("#watch-thinking"), m.top_moves);
      watchMoves.push(m.san); pushMoveList($("#watch-movelist"), watchMoves);
    } else if(m.type==="gameover"){
      $("#watch-status").textContent = `Fin · ${m.result} (${m.plies} demi-coups)`;
      watchStop();
    } else if(m.type==="error"){ $("#watch-status").textContent="Erreur : "+m.error; watchStop(); }
  };
  watchWs.onclose = ()=>{ $("#watch-start").disabled=false; $("#watch-stop").disabled=true; };
  watchWs.onerror = ()=>conn(false);
}
function watchStop(){ if(watchWs){ try{watchWs.close();}catch(e){} watchWs=null; }
  $("#watch-start").disabled=false; $("#watch-stop").disabled=true; }

/* ================= ENTRAINEMENT ================= */
function lineChart(svg, xs, ys, klass=""){
  svg.innerHTML="";
  if(!ys || ys.length<2){ svg.innerHTML='<text x="300" y="100" text-anchor="middle">aucune donnée</text>'; return; }
  const W=600,H=200,pad=28;
  const xmin=xs[0], xmax=xs[xs.length-1]||1;
  const ymin=Math.min(...ys), ymax=Math.max(...ys);
  const sx=v=>pad+(W-2*pad)*((v-xmin)/((xmax-xmin)||1));
  const sy=v=>pad+(H-2*pad)*(1-((v-ymin)/((ymax-ymin)||1)));
  let g="";
  for(let i=0;i<=4;i++){ const y=pad+(H-2*pad)*i/4; g+=`<line class="grid" x1="${pad}" y1="${y}" x2="${W-pad}" y2="${y}"/>`; }
  const pts = xs.map((x,i)=>`${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(" ");
  svg.innerHTML = g +
    `<polyline class="line ${klass}" points="${pts}"/>`+
    `<text x="${pad}" y="14">${ymax.toFixed(3)}</text>`+
    `<text x="${pad}" y="${H-6}">${ymin.toFixed(3)}</text>`+
    `<text x="${W-pad}" y="${H-6}" text-anchor="end">step ${xmax}</text>`;
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
    cards.innerHTML += card("Buffer", o.buffer[o.buffer.length-1].toLocaleString());
  }
  if(!cards.innerHTML) cards.innerHTML='<div class="empty">Aucun log d\'entraînement détecté dans data/.</div>';
  lineChart($("#chart-policy"), p.steps, p.policy, "");
  lineChart($("#chart-value"), p.steps, p.value, "v");
  lineChart($("#chart-online"), o.steps, o.loss, "o");
}
function card(k,v,sub=""){ return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div>${sub}</div>`; }

/* ================= SYSTEME ================= */
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
    card("Shards", da.shard_files.toLocaleString());

  $("#sys-procs").innerHTML = d.processes.length? d.processes.map(p=>
    `<span class="chip on"><span class="led"></span>${p.module} · pid ${p.pid}</span>`).join("")
    : '<div class="empty">Aucun process sanchess en cours.</div>';
}

/* ================= Onglets + auto-refresh ================= */
let timers={};
function switchTab(name){
  $$(".tab").forEach(t=>t.classList.toggle("active", t.dataset.tab===name));
  $$(".panel").forEach(p=>p.classList.toggle("active", p.id===name));
  for(const k in timers){ clearInterval(timers[k]); } timers={};
  if(name==="models") loadModels();
  if(name==="train"){ loadTraining(); if($("#train-auto").checked) timers.t=setInterval(loadTraining,4000); }
  if(name==="system"){ loadSystem(); if($("#sys-auto").checked) timers.s=setInterval(loadSystem,3000); }
  if(name==="watch"){ /* prêt */ }
}

/* ================= Init ================= */
function init(){
  $$(".tab").forEach(t=>t.addEventListener("click",()=>switchTab(t.dataset.tab)));
  $("#play-new").addEventListener("click", newGame);
  $("#play-flip").addEventListener("click", ()=>{ play.flip=!play.flip; drawPlay(); });
  $("#play-undo").addEventListener("click", async ()=>{
    if(play.busy||play.moves.length<2) return;
    play.moves.splice(-2); play.sans.splice(-2); play.sel=null;
    await refreshLegal(); pushMoveList($("#play-movelist"),play.sans); drawPlay();
  });
  $("#play-side").addEventListener("change", newGame);
  $("#watch-start").addEventListener("click", watchStart);
  $("#watch-stop").addEventListener("click", watchStop);
  $("#models-refresh").addEventListener("click", loadModels);
  $("#train-auto").addEventListener("change", ()=>switchTab("train"));
  $("#sys-auto").addEventListener("change", ()=>switchTab("system"));
  loadModels().then(newGame);
}
document.addEventListener("DOMContentLoaded", init);
