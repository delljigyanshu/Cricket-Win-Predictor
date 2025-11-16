// static/app.js

async function postPredict(payload){
  const res = await fetch('/api/predict', {
    method:'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  if(!res.ok){
    const txt = await res.text();
    throw new Error('Server: ' + res.status + ' - ' + txt);
  }
  return res.json();
}

function oversToBalls(oversStr){
  const s = String(oversStr).trim();
  if(s.indexOf('.') === -1) return Math.max(0, Math.floor(Number(s)))*6;
  const parts = s.split('.');
  const o = parseInt(parts[0]) || 0;
  const b = parseInt(parts[1]) || 0;
  return o*6 + b;
}

function computeDerived(){
  const score_before = Number(document.getElementById('score_before').value) || 0;
  const wickets_before = Number(document.getElementById('wickets_before').value) || 0;
  const overs_completed_raw = document.getElementById('overs_completed').value || '0';
  const balls_elapsed = oversToBalls(overs_completed_raw);
  const balls_remaining = Number(document.getElementById('balls_remaining').value) || 0;
  const runs_required = Number(document.getElementById('runs_required').value) || 0;
  const current_run_rate = Number(document.getElementById('current_run_rate').value) || 0;
  const req_run_rate = (balls_remaining>0) ? (runs_required / (balls_remaining/6.0)) : 0;

  const wickets_in_hand = Math.max(0, 10 - wickets_before);
  const frac_innings_complete = (balls_elapsed) / 120.0;
  const runs_required_norm = runs_required / Math.max(1, (score_before + runs_required));
  const rr_diff = current_run_rate - req_run_rate;
  const pressure = runs_required / ((balls_remaining + 1) * (wickets_in_hand + 1));

  return {
    score_before, wickets_before, overs_completed: Number(overs_completed_raw) || 0,
    balls_elapsed, balls_remaining, runs_required, req_run_rate,
    current_run_rate, wickets_in_hand, frac_innings_complete, runs_required_norm, rr_diff, pressure,
    over_int: Math.floor(Number(overs_completed_raw)) || 0
  };
}

function updateFacts(derived){
  const facts = document.getElementById('facts');
  facts.innerHTML = `
    <li>Runs required per ball: ${(derived.balls_remaining>0 ? (derived.runs_required/derived.balls_remaining).toFixed(2) : '—')}</li>
    <li>Wickets in hand: ${derived.wickets_in_hand}</li>
    <li>Pressure metric: ${derived.pressure.toFixed(4)}</li>
  `;
}

function setIndicator(percent, derived){
  const indicator = document.getElementById('indicator');
  let status = 'Toss-up';
  if(percent >= 70) status = 'Strong advantage';
  else if(percent >= 55) status = 'Advantage';
  else if(percent >= 45) status = 'Evenly poised';
  else status = 'Under pressure';
  indicator.innerText = `Status: ${status}`;
}

function animateGauge(percent){
  const arc = document.getElementById('gauge-arc');
  const text = document.getElementById('gauge-text');
  const radius = 85;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(100, percent));
  const dash = (pct/100) * circumference;
  arc.style.transition = 'stroke-dasharray 700ms ease';
  arc.setAttribute('stroke-dasharray', `${dash} ${circumference - dash}`);
  text.textContent = Math.round(pct) + '%';
}

function formatNumber(n){
  if(n === null || n === undefined) return '—';
  if(typeof n === 'number') return Number(n).toFixed(2);
  return n;
}

document.getElementById('predictBtn').addEventListener('click', async function(){
  const derived = computeDerived();
  updateFacts(derived);

  const batsman_form = Number(document.getElementById('batsman_form').value) || 0;
  const bowler_form = Number(document.getElementById('bowler_form').value) || 0;

  const payload = {
    score_before: derived.score_before,
    wickets_before: derived.wickets_before,
    overs_completed: derived.overs_completed,
    balls_remaining: derived.balls_remaining,
    runs_required: derived.runs_required,
    req_run_rate: derived.req_run_rate,
    current_run_rate: derived.current_run_rate,
    wickets_in_hand: derived.wickets_in_hand,
    frac_innings_complete: derived.frac_innings_complete,
    runs_required_norm: derived.runs_required_norm,
    rr_diff: derived.rr_diff,
    pressure: derived.pressure,
    over_int: derived.over_int,
    batsman_form: batsman_form,
    bowler_form: bowler_form
  };

  // loading UI
  document.getElementById('lr').innerText = '...';
  document.getElementById('gb').innerText = '...';
  document.getElementById('combined').innerText = '...';
  animateGauge(0);
  document.getElementById('summary').innerText = 'Predicting...';

  try{
    const r = await postPredict(payload);

    // format & display
    if(r.lr !== undefined) document.getElementById('lr').innerText = formatNumber(r.lr) + ' %';
    else document.getElementById('lr').innerText = '—';

    if(r.gb !== undefined) document.getElementById('gb').innerText = formatNumber(r.gb) + ' %';
    else document.getElementById('gb').innerText = '—';

    const combined = Number(r.combined) || 0;
    document.getElementById('combined').innerText = formatNumber(combined) + ' %';
    animateGauge(combined);
    setIndicator(combined, derived);

    // summary
    const parts = [];
    if(derived.runs_required <= 0) parts.push('Target already reached or invalid.');
    else {
      if(combined >= 70) parts.push('Batting team in very strong position.');
      else if(combined >= 55) parts.push('Batting team has advantage.');
      else if(combined >= 45) parts.push('Match is even.');
      else parts.push('Batting team under pressure.');
      parts.push(`RRR ≈ ${derived.req_run_rate.toFixed(2)} vs CRR ${derived.current_run_rate.toFixed(2)}.`);
      parts.push(`${derived.wickets_in_hand} wickets in hand.`);
    }
    document.getElementById('summary').innerText = parts.join(' ');
  }catch(err){
    console.error(err);
    alert('Prediction failed: ' + err.message);
    document.getElementById('summary').innerText = 'Prediction failed — check server/models.';
  }
});

document.getElementById('calcDerived').addEventListener('click', function(){
  const d = computeDerived();
  document.getElementById('req_run_rate').value = Number(d.req_run_rate.toFixed(3));
  if(!document.getElementById('current_run_rate').value || document.getElementById('current_run_rate').value === '0'){
    document.getElementById('current_run_rate').value = Number(d.current_run_rate || 0).toFixed(2);
  }
  updateFacts(d);
});

document.getElementById('calcRR').addEventListener('click', function(){
  const d = computeDerived();
  document.getElementById('req_run_rate').value = Number(d.req_run_rate.toFixed(3));
});
