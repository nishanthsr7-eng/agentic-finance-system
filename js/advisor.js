/* ══════════════════════════════════════════════════════════════
   FLUX · Smart Advisor (Prediction Cockpit)
   Built container-by-container. Shared state lives in `state`.
   ══════════════════════════════════════════════════════════════ */
(() => {
  'use strict';

  const API = window.FLUX_API || 'http://localhost:8000';

  // Assets whose live candles /market/candles can serve (yfinance-backed).
  const CHARTABLE = { BTC: 'btc', ETH: 'eth' };

  // Crypto labels in the training universe — used to set asset_type on paper trades.
  const CRYPTO_LABELS = new Set(['BTC','ETH','USDT','BNB','SOL','XRP','DOGE','ADA','AVAX','DOT','LINK','UNI','LTC','SHIB','TRX']);

  const state = {
    symbol: 'BTC',     // active chart / prediction symbol
    tf: '7D',          // active timeframe (1D|7D|1M|1Y)
    candles: [],       // [{t,o,h,l,c,v}] from /market/candles
    prediction: null,  // /predict/{sym}/forecast .prediction
    series: [],        // /predict/{sym}/forecast .series — daily closes
    live: false,       // candle feed healthy?
    lineMode: false,   // true when no candles (non-chartable symbol → forecast line)
    verdict: null,     // /predict/{sym}/verdict .verdict — latest verifier verdict, if any
    pointForecast: null, // /predict/{sym}/forecast .point_forecast — OOF skill of the point estimate
  };

  const $ = (id) => document.getElementById(id);

  /* ─────────────── formatters ─────────────── */
  function fmtPrice(v) {
    if (v == null || !isFinite(v)) return '—';
    const a = Math.abs(v);
    const dp = a >= 1000 ? 0 : a >= 1 ? 2 : 4;
    return (v < 0 ? '−' : '') + a.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
  }
  function fmtPct(v, dp = 2) {
    if (v == null || !isFinite(v)) return '—';
    return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(dp) + '%';
  }
  function fmtDate(s) {
    if (!s) return '';
    const d = new Date(s);
    if (isNaN(d)) return String(s);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }
  // Candle axis label: intraday feeds show the time, daily candles the date.
  function candleLbl(t) { return state.daily ? fmtDate(t) : fmtTime(t); }
  function fmtTime(t) {
    const d = new Date(t);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  async function getJSON(url, ms = 25000) {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), ms);
    try {
      const r = await fetch(url, { signal: ctrl.signal });
      if (!r.ok) throw new Error(`${r.status}`);
      return await r.json();
    } finally { clearTimeout(to); }
  }

  /* ══════════════════════════════════════════════════════════════
     CONTAINER 1 — LIVE FORECAST CHART
     Candles on the left; a "now" divider; then the agent's forecast
     RE-BASED onto the live price (relative return + relative band).
     ══════════════════════════════════════════════════════════════ */
  const NOW_FRAC = 0.66;          // candles occupy 0..66% of plot width
  const PAD = { t: 16, r: 66, b: 24, l: 12 };
  const UP = '#00e5a0', DOWN = '#ff4d4d';
  let chartW = 800, chartH = 440;
  let _chartGeom = null; // last drawChart() geometry, used by the hover crosshair/tooltip

  function syncViewport() {
    const wrap = $('chart-wrap');
    const w = Math.max(wrap.clientWidth || 800, 360);
    const h = Math.max(wrap.clientHeight || 440, 140);
    chartW = w; chartH = h;
    $('chart-svg').setAttribute('viewBox', `0 0 ${w} ${h}`);
    $('chart-svg').setAttribute('preserveAspectRatio', 'none');
  }

  // The point estimate is only promoted when the shipped return model beat "no change" out of fold.
  function pointHasSkill() { return !!(state.pointForecast && state.pointForecast.has_skill); }

  // Build the re-based forecast geometry from the live anchor price.
  // The range is a RETURN range, so it follows the live price rather than the model's last close.
  function rebasedForecast(anchor) {
    const p = state.prediction;
    if (!p || anchor == null || !isFinite(anchor)) return null;
    const base = p.last_close;
    if (!base || !isFinite(base)) return null;
    const predFrac = p.pred_return != null ? (1 + p.pred_return) : (p.pred_price / base);
    const lowFrac = p.conf_low != null ? p.conf_low / base : predFrac;
    const highFrac = p.conf_high != null ? p.conf_high / base : predFrac;
    const has90 = p.conf_low_90 != null && p.conf_high_90 != null;
    return {
      projected: anchor * predFrac,
      bandLow: anchor * lowFrac,
      bandHigh: anchor * highFrac,
      band90Low: has90 ? anchor * (p.conf_low_90 / base) : null,
      band90High: has90 ? anchor * (p.conf_high_90 / base) : null,
      deltaPct: (predFrac - 1) * 100,
      up: p.direction !== 'DOWN',
    };
  }

  function drawChart() {
    syncViewport();
    const svg = $('chart-svg');
    const candles = state.candles;
    const lineMode = state.lineMode;
    const src = lineMode ? state.series : candles;
    if (!src || src.length < 2) { svg.innerHTML = ''; _chartGeom = null; hideChartHover(); return; }

    const pw = chartW - PAD.l - PAD.r;
    const ph = chartH - PAD.t - PAD.b;
    const hasPred = !!state.prediction;
    const xNow = hasPred ? PAD.l + pw * NOW_FRAC : PAD.l + pw;
    const xEnd = PAD.l + pw;

    // Last actual price = anchor for the re-based forecast.
    const anchor = lineMode ? src[src.length - 1].close : candles[candles.length - 1].c;
    const fc = hasPred ? rebasedForecast(anchor) : null;

    // ── Y domain: actual highs/lows + the forecast point/anchor (always shown) ──
    let lo = Infinity, hi = -Infinity;
    if (lineMode) {
      for (const d of src) { lo = Math.min(lo, d.close); hi = Math.max(hi, d.close); }
    } else {
      for (const c of candles) { lo = Math.min(lo, c.l); hi = Math.max(hi, c.h); }
    }
    const range0 = (hi - lo) || (hi || 1);
    lo -= range0 * 0.06; hi += range0 * 0.06;
    if (fc) { [fc.projected, anchor].forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); }); }

    // The price range is shown in the 5-day outlook panel below the chart, not as a cone:
    // it is often far wider than recent price action and would crush the price line.
    const range = hi - lo || 1;
    const py = v => PAD.t + ph - ((v - lo) / range) * ph;
    const N = src.length;
    const px = i => PAD.l + (N === 1 ? 0 : i / (N - 1)) * (xNow - PAD.l);

    let out = '';

    // ── Y grid + price axis (right) ──
    for (let i = 0; i <= 4; i++) {
      const val = lo + (range / 4) * i;
      const y = py(val);
      out += `<line x1="${PAD.l}" y1="${y.toFixed(1)}" x2="${xEnd.toFixed(1)}" y2="${y.toFixed(1)}" stroke="rgba(255,255,255,.04)" stroke-width="1"/>`;
      out += `<text x="${(chartW - PAD.r + 6).toFixed(1)}" y="${(y + 3).toFixed(1)}" font-family="'JetBrains Mono',monospace" font-size="9.5" fill="rgba(255,255,255,.4)">${fmtPrice(val)}</text>`;
    }

    // ── Actual price: candles or (lineMode) area+line ──
    if (lineMode) {
      const pts = src.map((d, i) => [px(i), py(d.close)]);
      const line = pts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
      const area = `M ${pts[0][0].toFixed(1)},${(PAD.t + ph).toFixed(1)} ` +
        pts.map(p => `L ${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ') +
        ` L ${pts[pts.length - 1][0].toFixed(1)},${(PAD.t + ph).toFixed(1)} Z`;
      out += `<path d="${area}" fill="rgba(0,229,160,.07)"/>`;
      out += `<polyline points="${line}" fill="none" stroke="${UP}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>`;
    } else {
      const cW = Math.max((xNow - PAD.l) / N * 0.62, 1.4);
      let body = '';
      candles.forEach((c, i) => {
        const x = px(i), yO = py(c.o), yC = py(c.c), yH = py(c.h), yL = py(c.l);
        const bull = c.c >= c.o;
        const col = bull ? UP : DOWN;
        const top = Math.min(yO, yC);
        const bh = Math.max(Math.abs(yO - yC), 0.8);
        body += `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${yH.toFixed(1)}" y2="${yL.toFixed(1)}" stroke="${col}" stroke-width=".7" opacity=".55"/>`;
        body += `<rect x="${(x - cW / 2).toFixed(1)}" y="${top.toFixed(1)}" width="${cW.toFixed(1)}" height="${bh.toFixed(1)}" fill="${col}" opacity="${bull ? '.9' : '.7'}" rx=".4"/>`;
      });
      out += body;
    }

    // ── Forecast zone (re-based) ──
    if (fc) {
      const yA = py(anchor), yP = py(fc.projected);
      // muted unless the point estimate has out-of-fold skill over "no change"
      const col = pointHasSkill() ? (fc.up ? UP : DOWN) : 'rgba(255,255,255,.45)';
      // shaded forecast-zone background
      out += `<rect x="${xNow.toFixed(1)}" y="${PAD.t}" width="${(xEnd - xNow).toFixed(1)}" height="${ph.toFixed(1)}" fill="rgba(255,255,255,.015)"/>`;
      // now divider
      out += `<line x1="${xNow.toFixed(1)}" y1="${PAD.t}" x2="${xNow.toFixed(1)}" y2="${(PAD.t + ph).toFixed(1)}" stroke="rgba(255,255,255,.18)" stroke-width="1" stroke-dasharray="2 3"/>`;
      // flat current-price reference across the zone
      out += `<line x1="${xNow.toFixed(1)}" y1="${yA.toFixed(1)}" x2="${xEnd.toFixed(1)}" y2="${yA.toFixed(1)}" stroke="rgba(255,255,255,.22)" stroke-width="1" stroke-dasharray="1 4"/>`;
      // projection (dotted, glowing)
      out += `<line x1="${xNow.toFixed(1)}" y1="${yA.toFixed(1)}" x2="${xEnd.toFixed(1)}" y2="${yP.toFixed(1)}" stroke="${col}" stroke-width="2.2" stroke-dasharray="0.5 4.5" stroke-linecap="round"/>`;
      // markers
      out += `<circle cx="${xNow.toFixed(1)}" cy="${yA.toFixed(1)}" r="3" fill="#fff"/>`;
      out += `<circle cx="${xEnd.toFixed(1)}" cy="${yP.toFixed(1)}" r="7" fill="${col}" fill-opacity=".22"/>`;
      out += `<circle cx="${xEnd.toFixed(1)}" cy="${yP.toFixed(1)}" r="3.4" fill="${col}"/>`;
      // endpoint label
      const lblY = Math.max(PAD.t + 8, Math.min(PAD.t + ph - 4, yP + (fc.deltaPct >= 0 ? -8 : 13)));
      out += `<text x="${(xEnd - 2).toFixed(1)}" y="${lblY.toFixed(1)}" fill="${col}" font-size="9" font-weight="700" font-family="'JetBrains Mono',monospace" text-anchor="end">${fmtPrice(fc.projected)} ${fc.deltaPct >= 0 ? '▲' : '▼'}${Math.abs(fc.deltaPct).toFixed(1)}%</text>`;
    }

    // ── X labels: first · now · target ──
    const xlbl = (x, t, a) => `<text x="${x.toFixed(1)}" y="${(chartH - 7).toFixed(1)}" fill="rgba(255,255,255,.32)" font-size="8.5" font-family="'JetBrains Mono',monospace" text-anchor="${a}">${t}</text>`;
    const firstLabel = lineMode ? fmtDate(src[0].date) : candleLbl(candles[0].t);
    out += xlbl(PAD.l, firstLabel, 'start');
    if (hasPred) {
      out += xlbl(xNow, 'NOW', 'middle');
      if (state.prediction.target_date) out += xlbl(xEnd, fmtDate(state.prediction.target_date), 'end');
    } else {
      const lastLabel = lineMode ? fmtDate(src[src.length - 1].date) : candleLbl(candles[candles.length - 1].t);
      out += xlbl(xEnd, lastLabel, 'end');
    }

    // hover dot, repositioned by the crosshair handler — hidden by default
    out += `<circle id="chart-hover-dot" r="3.5" fill="#fff" stroke="rgba(0,0,0,.35)" stroke-width="1" style="display:none" pointer-events="none"/>`;

    svg.innerHTML = out;

    _chartGeom = {
      N, lineMode, src, candles, fc, anchor, hasPred,
      PAD, pw, ph, xNow, xEnd, lo, hi, range, py,
      prediction: state.prediction,
    };
  }

  /* ─────────────── chart hover: crosshair + tooltip ─────────────── */
  function hideChartHover() {
    $('chart-crosshair').style.display = 'none';
    $('chart-tooltip').style.display = 'none';
    const dot = $('chart-svg').querySelector('#chart-hover-dot');
    if (dot) dot.style.display = 'none';
  }

  function onChartHover(e) {
    const g = _chartGeom;
    if (!g) { hideChartHover(); return; }
    const svg = $('chart-svg'), wrap = $('chart-wrap');
    const svgRect = svg.getBoundingClientRect(), wrapRect = wrap.getBoundingClientRect();
    if (!svgRect.width || !svgRect.height) { hideChartHover(); return; }
    const scaleX = chartW / svgRect.width, scaleY = chartH / svgRect.height;
    const vbX = (e.clientX - svgRect.left) * scaleX;
    const vbY = (e.clientY - svgRect.top) * scaleY;
    if (vbX < g.PAD.l - 1 || vbX > g.xEnd + 1 || vbY < g.PAD.t - 1 || vbY > g.PAD.t + g.ph + 1) {
      hideChartHover(); return;
    }

    let label, priceVal, col = null, dotX = vbX;
    if (vbX <= g.xNow || !g.fc) {
      // hover over actual price history — snap to the nearest candle/close
      const frac = g.N === 1 ? 0 : (Math.min(vbX, g.xNow) - g.PAD.l) / (g.xNow - g.PAD.l);
      const i = Math.max(0, Math.min(g.N - 1, Math.round(frac * (g.N - 1))));
      const d = g.lineMode ? g.src[i] : g.candles[i];
      priceVal = g.lineMode ? d.close : d.c;
      label = g.lineMode ? fmtDate(d.date) : candleLbl(d.t);
      dotX = g.PAD.l + (g.N === 1 ? 0 : i / (g.N - 1)) * (g.xNow - g.PAD.l);
    } else {
      // hover over the forecast zone — interpolate along the projection line
      const frac = Math.max(0, Math.min(1, (vbX - g.xNow) / (g.xEnd - g.xNow)));
      priceVal = g.anchor + (g.fc.projected - g.anchor) * frac;
      const tgt = g.prediction && g.prediction.target_date ? fmtDate(g.prediction.target_date) : '';
      label = tgt ? `Forecast → ${tgt}` : 'Forecast';
      col = g.fc.up ? 'up' : 'down';
    }
    const dotY = g.py(priceVal);

    // crosshair (HTML overlay, in chart-wrap-relative pixels)
    const cross = $('chart-crosshair');
    const screenX = (svgRect.left - wrapRect.left) + dotX / scaleX;
    cross.style.left = `${screenX}px`;
    cross.style.top = `${(svgRect.top - wrapRect.top) + g.PAD.t / scaleY}px`;
    cross.style.height = `${g.ph / scaleY}px`;
    cross.style.display = 'block';

    // hover dot (SVG, in viewBox units)
    const dot = $('chart-svg').querySelector('#chart-hover-dot');
    if (dot) {
      dot.setAttribute('cx', dotX.toFixed(1));
      dot.setAttribute('cy', dotY.toFixed(1));
      dot.setAttribute('fill', col === 'down' ? DOWN : col === 'up' ? UP : '#fff');
      dot.style.display = 'block';
    }

    // tooltip
    const tip = $('chart-tooltip');
    const priceCls = col ? ` ${col}` : '';
    tip.innerHTML = `<div class="ct-date">${label}</div><div class="ct-price${priceCls}">${fmtPrice(priceVal)}</div>`;
    tip.style.left = `${screenX}px`;
    const tipTop = (svgRect.top - wrapRect.top) + g.PAD.t / scaleY + 6;
    tip.style.top = `${tipTop}px`;
    tip.style.display = 'block';
    // keep the tooltip within the chart bounds horizontally
    const tipW = tip.offsetWidth;
    const minX = tipW / 2 + 2, maxX = wrapRect.width - tipW / 2 - 2;
    tip.style.left = `${Math.max(minX, Math.min(maxX, screenX))}px`;
  }

  function initChartHover() {
    const wrap = $('chart-wrap');
    wrap.addEventListener('mousemove', onChartHover);
    wrap.addEventListener('mouseleave', hideChartHover);
  }

  /* ─────────────── live price readout ─────────────── */
  function updateLiveReadout() {
    const src = state.lineMode ? state.series : state.candles;
    const priceEl = $('cl-price'), chgEl = $('cl-chg'), dot = $('cl-dot'), lbl = $('cl-live-lbl');
    if (!src || src.length < 2) { priceEl.textContent = '—'; chgEl.textContent = '—'; chgEl.className = 'cl-chg neu'; return; }
    const last = state.lineMode ? src[src.length - 1].close : src[src.length - 1].c;
    const first = state.lineMode ? src[0].close : src[0].o;
    const pct = (last - first) / first * 100;
    priceEl.textContent = fmtPrice(last);
    const up = pct >= 0;
    chgEl.textContent = `${up ? '▲' : '▼'} ${Math.abs(pct).toFixed(2)}% · ${state.daily || state.lineMode ? src.length + 'D' : state.tf}`;
    chgEl.className = 'cl-chg ' + (up ? 'up' : 'down');
    const daily = state.lineMode || state.daily;
    dot.className = 'cl-dot ' + (state.live ? 'live' : (daily ? '' : 'stale'));
    lbl.textContent = state.live ? 'LIVE' : (daily ? 'DAILY' : 'STALE');
  }

  /* ─────────────── loaders ─────────────── */
  async function loadCandles() {
    const asset = CHARTABLE[state.symbol];
    if (!asset) { state.candles = []; return false; }
    try {
      const d = await getJSON(`${API}/market/candles/${asset}?tf=${state.tf}`);
      const c = d.candles || [];
      if (c.length > 1) { state.candles = c; state.live = true; return true; }
      return false;
    } catch (e) { return false; }
  }

  function dailyCandles() {
    const s = state.series;
    if (s.length < 2 || !s.every(p => p.open != null && p.high != null && p.low != null)) return null;
    return s.map(p => ({ t: new Date(p.date).getTime(), o: p.open, h: p.high, l: p.low, c: p.close }));
  }

  async function loadForecast() {
    try {
      const d = await getJSON(`${API}/predict/${state.symbol}/forecast?lookback=60`);
      state.prediction = d.prediction || null;
      state.series = d.series || [];
      state.pointForecast = d.point_forecast || null;
      return true;
    } catch (e) { state.prediction = null; state.series = []; state.pointForecast = null; return false; }
  }

  // Latest verifier verdict for the active symbol, if the daily cycle has logged one.
  async function loadVerdict() {
    try {
      const d = await getJSON(`${API}/predict/${state.symbol}/verdict`);
      state.verdict = d.verdict || null;
    } catch (e) { state.verdict = null; }
  }

  function showChartStatus(msg, isErr, retry) {
    const s = $('chart-status');
    s.style.display = 'flex';
    s.className = 'chart-status' + (isErr ? ' err' : '');
    s.innerHTML = msg + (retry ? `<br><button class="chart-retry" id="chart-retry">Retry</button>` : '');
    if (retry) $('chart-retry').addEventListener('click', () => loadAll(false));
  }
  // Shimmering bar-chart placeholder shown while the first fetch for a symbol is in flight.
  function showChartSkeleton() {
    const s = $('chart-status');
    s.style.display = 'flex';
    s.className = 'chart-status loading';
    const bars = Array.from({ length: 28 }, () => `<span class="skel" style="height:${(18 + Math.random() * 72).toFixed(0)}%"></span>`).join('');
    s.innerHTML = `<div class="chart-skel-bars">${bars}</div><div class="chart-skel-label">Loading market data…</div>`;
  }
  function hideChartStatus() { $('chart-status').style.display = 'none'; }

  // Full (re)load for the active symbol/timeframe.
  let _loading = false;
  async function loadAll(quiet) {
    if (_loading) return;
    _loading = true;
    if (!quiet) showChartSkeleton();
    const [okC] = await Promise.all([loadCandles(), loadForecast(), loadVerdict()]);
    // No live candle feed (everything but BTC/ETH): draw daily candles from
    // the stored history; fall back to a close-price line if OHLC is missing.
    const daily = okC ? null : dailyCandles();
    if (daily) state.candles = daily;
    state.daily = !!daily;
    state.lineMode = !okC && !daily && state.series.length > 1;
    state.live = okC;
    if (okC || daily || state.lineMode) {
      hideChartStatus();
      drawChart();
      updateLiveReadout();
    } else if (!quiet) {
      showChartStatus('No market data for this asset. Is the API running on :8000?', true, true);
    }
    renderVerdictCard();
    syncSymbolLabels();
    _loading = false;
  }

  // Lightweight live tick — refresh candles only, keep forecast.
  async function liveTick() {
    if (document.visibilityState !== 'visible') return;
    if (state.lineMode) return; // daily-only symbols don't tick intraday
    const ok = await loadCandles();
    state.live = ok;
    if (ok) { drawChart(); updateLiveReadout(); }
  }

  /* ─────────────── tab wiring ─────────────── */
  function initChart() {
    syncViewport();
    $('asset-tabs').addEventListener('click', (e) => {
      const b = e.target.closest('.asset-tab'); if (!b) return;
      setSymbol(b.dataset.asset);
    });
    $('tf-tabs').addEventListener('click', (e) => {
      const b = e.target.closest('.tf-tab'); if (!b) return;
      state.tf = b.dataset.tf;
      document.querySelectorAll('.tf-tab').forEach(t => t.classList.toggle('active', t === b));
      loadAll(false);
    });
    // redraw on container resize
    let rt = null;
    new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(() => { if (state.candles.length > 1 || state.series.length > 1) drawChart(); }, 120); }).observe($('chart-wrap'));
    initChartHover();
    // live polling (paused when hidden)
    setInterval(liveTick, 30000);
    document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') liveTick(); });
    loadAll(false);
  }

  // Switch active symbol (from tabs or leaderboard). Highlights the right tab.
  function setSymbol(sym) {
    state.symbol = sym;
    document.querySelectorAll('.asset-tab').forEach(t => t.classList.toggle('active', t.dataset.asset === sym));
    if (typeof markLeaderboardActive === 'function') markLeaderboardActive();
    syncSymbolLabels();
    loadAll(false);
  }

  function syncSymbolLabels() {
    const s = state.symbol;
    if ($('chat-ctx-sym')) $('chat-ctx-sym').textContent = s;
  }

  /* ══════════════════════════════════════════════════════════════
     CONTAINER 2 — SIGNAL CONSOLE STRIP
     The agent's current call, with the target/band RE-BASED to the
     live price so it reads consistently with the chart projection.
     ══════════════════════════════════════════════════════════════ */
  function liveAnchor() {
    if (state.lineMode) return state.series.length ? state.series[state.series.length - 1].close : null;
    return state.candles.length ? state.candles[state.candles.length - 1].c : null;
  }

  const REGIME = { trend: 'Trending', chop: 'Sideways' };
  const _track = {};   // symbol -> { acc, n } past-call hit rate

  async function loadTrack(sym) {
    if (_track[sym] !== undefined) return;
    _track[sym] = null;
    try {
      const h = await getJSON(`${API}/predict/${sym}/history?limit=100`);
      const n = (h.history || []).filter(r => r.correct === 0 || r.correct === 1).length;
      if (h.realized_accuracy != null && n) _track[sym] = { acc: Math.round(h.realized_accuracy * 100), n };
    } catch (e) { /* no track record line */ }
    if (sym === state.symbol) renderVerdictCard();
  }

  // The second check (an LLM reading recent news) as one line: it can only
  // lower the model's confidence or veto the call, never raise it.
  function checkLine(v) {
    if (_verifying) return `<span class="vd-check-txt">Checking against recent news…</span>`;
    if (!v || !v.label || v.verifier === 'unavailable' || v.verifier === 'parse_error') {
      return `<span class="vd-check-txt">Not checked against the news yet.</span>`;
    }
    const veto = v.label === 'VETO' || v.veto;
    const agree = !veto && (v.label === 'agree' || v.agree);
    const [cls, icon, txt] = veto ? ['veto', ICON.veto, 'News check: <b>vetoed</b> this forecast']
      : agree ? ['agree', ICON.agree, 'News check: <b>agrees</b>']
      : ['downgrade', ICON.down, `News check: lowered confidence to <b>${v.final_confidence}%</b>`];
    const why = v.rationale ? ` <span class="vd-why">${escapeHtml(v.rationale)}</span>` : '';
    const tip = [v.rationale, v.risks && `Main risk: ${v.risks}`].filter(Boolean).join(' — ');
    return `<span class="vd-check-ic ${cls}">${icon}</span><span class="vd-check-txt" title="${escapeHtml(tip)}">${txt}.${why}</span>`;
  }

  function renderVerdictCard() {
    const el = $('verdict');
    if (!el) return;
    const p = state.prediction, sym = escapeHtml(state.symbol);
    if (!p) {
      el.innerHTML = `<div class="vd-head"><span class="vd-title">No forecast for ${sym} yet.</span></div>`;
      return;
    }
    const anchor = liveAnchor();
    const fc = rebasedForecast(anchor);
    const up = p.direction !== 'DOWN';
    const days = p.horizon_days || 5;
    const conf = state.verdict && state.verdict.final_confidence != null && state.verdict.label !== 'agree'
      ? state.verdict.final_confidence : p.confidence;

    const range = fc && p.conf_low != null
      ? `Likely between <b>${fmtPrice(fc.bandLow)}</b> and <b>${fmtPrice(fc.bandHigh)}</b> by ${fmtDate(p.target_date)}`
      : '';
    let bar = '';
    if (fc && anchor && fc.bandHigh > fc.bandLow) {
      const pad = (fc.bandHigh - fc.bandLow) * 0.25, lo = fc.bandLow - pad, span = fc.bandHigh + pad - lo;
      const pos = v => Math.max(0, Math.min(100, (v - lo) / span * 100)).toFixed(1);
      bar = `<div class="vd-bar"><span class="vd-bar-in" style="left:${pos(fc.bandLow)}%;right:${(100 - pos(fc.bandHigh)).toFixed(1)}%"></span><span class="vd-bar-now" style="left:${pos(anchor)}%" title="Now ${fmtPrice(anchor)}"></span></div>`;
    }

    const vetoed = !!(state.verdict && state.verdict.label === 'VETO');
    const size = vetoed ? 'Stay out (vetoed by news check)'
      : p.act && p.kelly_frac != null ? `${(p.kelly_frac * 100).toFixed(1)}% of portfolio`
      : 'Stay out (signal too weak)';
    const t = _track[state.symbol];
    const facts = [
      ['Suggested size', size],
      ['Market', REGIME[p.regime] || (p.regime ? p.regime[0].toUpperCase() + p.regime.slice(1) : '—')],
      ['Track record', t ? `Right ${t.acc}% of ${t.n} past calls` : 'Not enough past calls yet'],
    ];

    el.innerHTML = `
      <div class="vd-head">
        <span class="vd-dir ${up ? 'up' : 'down'}">${up ? ICON.up : ICON.dn}</span>
        <span class="vd-title">${vetoed
          ? `${sym}: model says <b class="${up ? 'up' : 'down'}">${up ? 'up' : 'down'}</b>, but the news check vetoed it`
          : `${sym} likely <b class="${up ? 'up' : 'down'}">${up ? 'up' : 'down'}</b> over the next ${days} days`}</span>
        <span class="vd-conf">${conf}% confidence</span>
      </div>
      ${range ? `<div class="vd-range">${range}</div>${bar}` : ''}
      <div class="vd-facts">${facts.map(([k, v]) => `<div class="vd-fact"><span>${k}</span><b>${v}</b></div>`).join('')}</div>
      <div class="vd-check">${checkLine(state.verdict)}<button class="mini-btn" id="vf-btn"${_verifying ? ' disabled' : ''}>Check now</button></div>`;
    $('vf-btn').addEventListener('click', runVerifier);
    loadTrack(state.symbol);
  }

  /* ══════════════════════════════════════════════════════════════
     CONTAINER 3 — CONVICTION LEADERBOARD
     The agent's highest-conviction calls across every symbol it
     covers. Click a row to load it into the chart.
     ══════════════════════════════════════════════════════════════ */
  // Plain stroked line icons (inherit colour from the badge class).
  const svg = (d) => `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
  const ICON = {
    up:    svg('<polyline points="6 15 12 9 18 15"/>'),
    dn:    svg('<polyline points="6 9 12 15 18 9"/>'),
    veto:  svg('<line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/>'),
    down:  svg('<line x1="12" y1="5" x2="12" y2="19"/><polyline points="6 13 12 19 18 13"/>'),
    agree: svg('<polyline points="5 12 10 17 19 7"/>'),
  };
  let _leaderboard = [];
  let _verdicts = {}; // symbol -> latest verifier verdict (for VETO/downgrade badges)

  function renderLeaderboard() {
    const body = $('lb-body');
    if (!_leaderboard.length) { body.innerHTML = `<div class="mini-empty">No agent calls logged yet.<br>Trigger predictions on the API to populate.</div>`; return; }
    body.innerHTML = _leaderboard.map((r, i) => {
      const up = r.direction !== 'DOWN';
      const conf = r.confidence != null ? r.confidence : 0;
      const chartable = !!CHARTABLE[r.symbol];
      const v = _verdicts[r.symbol];
      let badge = '<span class="lb-verdict"></span>';
      if (v && v.label === 'VETO') badge = `<span class="lb-verdict veto" title="News check vetoed this (confidence → ${v.final_confidence}%): ${escapeHtml(v.rationale || '')}">${ICON.veto}</span>`;
      else if (v && v.label === 'downgrade') badge = `<span class="lb-verdict downgrade" title="News check lowered confidence to ${v.final_confidence}%: ${escapeHtml(v.rationale || '')}">${ICON.down}</span>`;
      return `<div class="lb-row${r.symbol === state.symbol ? ' active' : ''}" data-sym="${r.symbol}" title="${chartable ? 'Live chart' : 'Forecast (no live candles)'}">
        <span class="lb-rank">${i + 1}</span>
        <span class="lb-sym">${r.symbol}</span>
        <span class="lb-dir${up ? '' : ' down'}">${up ? ICON.up : ICON.dn}</span>
        <span class="lb-bar-wrap"><span class="lb-bar" style="width:${conf}%"></span></span>
        <span class="lb-conf">${conf}%</span>
        ${badge}
      </div>`;
    }).join('');
    body.querySelectorAll('.lb-row').forEach(row => {
      row.addEventListener('click', () => setSymbol(row.dataset.sym));
    });
  }

  function markLeaderboardActive() {
    document.querySelectorAll('.lb-row').forEach(r => r.classList.toggle('active', r.dataset.sym === state.symbol));
  }

  async function initLeaderboard() {
    try {
      const d = await getJSON(`${API}/predict/leaderboard?limit=12`);
      _leaderboard = d.leaderboard || [];
      $('lb-meta').textContent = `${_leaderboard.length} calls`;
    } catch (e) {
      _leaderboard = [];
      $('lb-meta').textContent = 'offline';
    }
    try {
      const vd = await getJSON(`${API}/predict/verdicts?limit=50`);
      _verdicts = {};
      (vd.verdicts || []).forEach(v => { _verdicts[v.symbol] = v; });
    } catch (e) { _verdicts = {}; }
    renderLeaderboard();
  }
  /* ── News check, on demand: re-runs the LLM verifier for the active symbol ── */
  let _verifying = false;

  async function runVerifier() {
    if (_verifying) return;
    _verifying = true;
    renderVerdictCard();
    try {
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 60000);
      const r = await fetch(`${API}/predict/${state.symbol}/verify`, { method: 'POST', signal: ctrl.signal });
      clearTimeout(to);
      if (!r.ok) throw new Error(r.status);
      const d = await r.json();
      const v = d.verdict || d;
      if (v && v.verifier !== 'unavailable' && v.verifier !== 'parse_error') state.verdict = v;
    } catch (e) { /* keep the last stored verdict */ }
    _verifying = false;
    renderVerdictCard();
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  /* ══════════════════════════════════════════════════════════════
     CONTAINER 6 — SYMBOL-AWARE AURA CHAT
     Streaming chat that knows (a) the user's finances from
     localStorage and (b) the agent's LIVE call on the active symbol,
     so "explain this call" / "what's the risk" are grounded.
     ══════════════════════════════════════════════════════════════ */
  const HISTORY_KEY = 'flux_advisor_history';
  let chatHistory = [];

  function getFinancialContext() {
    const txs = JSON.parse(localStorage.getItem('flux_transactions') || '[]');
    const now = new Date();
    const isMonth = t => { const d = new Date(t.date); return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth(); };
    const monthTxs = txs.filter(isMonth);
    const income = monthTxs.filter(t => t.amount > 0).reduce((s, t) => s + t.amount, 0);
    const expense = monthTxs.filter(t => t.amount < 0).reduce((s, t) => s + Math.abs(t.amount), 0);
    const savingsRate = income > 0 ? Math.round(((income - expense) / income) * 100) : null;
    const thirtyAgo = new Date(now - 30 * 864e5);
    const catTotals = {};
    txs.filter(t => new Date(t.date) >= thirtyAgo && t.amount < 0).forEach(t => {
      const c = (t.category || 'other').toLowerCase(); catTotals[c] = (catTotals[c] || 0) + Math.abs(t.amount);
    });
    const topCats = Object.entries(catTotals).sort(([, a], [, b]) => b - a).slice(0, 5);
    return { txs, income, expense, savingsRate, topCats, hasTxData: txs.length > 0 };
  }

  function signalBlock() {
    const p = state.prediction;
    if (!p) return `No live model signal for ${state.symbol} right now.`;
    const anchor = liveAnchor();
    const fc = rebasedForecast(anchor);
    const tgt = (fc ? `${fmtPrice(fc.projected)} (${fmtPct(fc.deltaPct)})` : fmtPrice(p.pred_price)) +
      (pointHasSkill() ? '' : ' (central estimate, not better than assuming no change)');
    return [
      `LIVE AGENT SIGNAL — ${state.symbol}:`,
      `  Direction: ${p.direction} | Calibrated confidence: ${p.confidence}%${p.base_confidence != null ? ` (base ${p.base_confidence}%)` : ''}`,
      `  Live price: ${anchor != null ? fmtPrice(anchor) : 'n/a'} | ${p.horizon_days || 5}-day target: ${tgt}`,
      `  ${p.band_pct || 80}% range: ${fc ? fmtPrice(fc.bandLow) + ' – ' + fmtPrice(fc.bandHigh) : 'n/a'}` +
        (fc && fc.band90Low != null ? ` | 90% range: ${fmtPrice(fc.band90Low)} – ${fmtPrice(fc.band90High)}` : '') +
        ' (historical coverage, not a promise)',
      `  Regime: ${p.regime || 'n/a'} | News sentiment: ${p.sentiment_label || 'n/a'} (${p.sentiment != null ? p.sentiment.toFixed(2) : 'n/a'})`,
      `  Meta act-gate: ${p.act ? `ACT (Kelly ${(p.kelly_frac * 100).toFixed(1)}%)` : 'NO-TRADE (conviction below action threshold)'}`,
    ].join('\n');
  }

  function buildSystemPrompt() {
    const { income, expense, savingsRate, topCats, hasTxData } = getFinancialContext();
    const cats = topCats.map(([c, a]) => `  - ${c}: ₹${Math.round(a).toLocaleString('en-IN')}`).join('\n') || '  - (none)';
    const fin = hasTxData
      ? `USER FINANCES (this month): income ₹${Math.round(income).toLocaleString('en-IN')}, expenses ₹${Math.round(expense).toLocaleString('en-IN')}, savings rate ${savingsRate != null ? savingsRate + '%' : 'n/a'}.\nTop categories (30d):\n${cats}`
      : `USER FINANCES: no transaction data yet (guide them to the Payments page if relevant).`;

    return `You are the advisor inside FLUX. Use plain, everyday language and avoid jargon (no 'Kelly', 'regime', 'calibrated', 'conformal'). You sit next to a quantitative trading agent and the live chart for ${state.symbol}.
Be concise, specific and data-driven. Use **bold** for key numbers. Keep answers under 180 words unless asked for depth.
When explaining the model's call, ground it in the LIVE AGENT SIGNAL below — never invent numbers. The calibrated confidence already reflects the model's real hit-rate; the meta act-gate decides whether the edge is worth trading. Be honest about uncertainty and the no-trade case.

SPECIAL RULE — INVESTMENT PROPOSALS: when the user explicitly asks for a "trade proposal" / "investment proposal" / "generate a proposal" for an asset, reply ONLY with valid JSON (no markdown) matching:
{"type":"investment_proposal","asset":"<name>","ticker":"<TICKER>","action":"BUY"|"SELL","entry":"<str>","target":"<str>","stop_loss":"<str>","timeframe":"<e.g. 5-10 days>","rationale":"<2 sentences>","risk_level":"Low"|"Medium"|"High"}

=== CONTEXT ===
${signalBlock()}

${fin}
=== END CONTEXT ===`;
  }

  function ts() { return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
  function mdToHtml(t) {
    return escapeHtml(t)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  }
  function tryParseProposal(text) {
    let t = text.trim();
    // Ollama often wraps the reply in a ```json ... ``` fence despite the
    // "no markdown" instruction — strip a fence wrapping the whole reply.
    const fence = t.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
    if (fence) t = fence[1].trim();
    if (!t.startsWith('{')) return null;
    try { const o = JSON.parse(t); if (o.type === 'investment_proposal') return o; } catch (_) {}
    return null;
  }
  // Execute the proposal as a paper trade via POST /db/trades (requires login;
  // window.fetch already attaches the session token — see flux-data.js).
  async function acceptProposal(p, btn) {
    const symbol = String(p.ticker || p.asset || state.symbol || '').toUpperCase().replace(/-USD$/, '').trim();
    const side = (p.action || 'BUY').toUpperCase();
    const price = liveAnchor();
    if (!symbol || !price) { btn.textContent = 'No live price'; btn.disabled = true; return; }

    btn.disabled = true; btn.textContent = 'Placing…';
    try {
      const res = await fetch(`${API}/db/trades`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol, name: p.asset || symbol, side, price,
          amount: 1000,   // fixed $1,000 paper notional per accepted proposal
          asset_type: CRYPTO_LABELS.has(symbol) ? 'crypto' : 'stocks',
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      btn.textContent = `Filled ✓ ${data.trade.quantity.toFixed(4)} ${symbol} @ ${fmtPrice(data.trade.price)}`;
    } catch (e) {
      btn.classList.add('error');
      btn.textContent = 'Failed — ' + (e.message || 'try again');
      setTimeout(() => { btn.disabled = false; btn.textContent = 'Accept'; btn.classList.remove('error'); }, 3000);
    }
  }
  function renderProposalCard(p) {
    const ac = (p.action || 'BUY').toLowerCase(), rc = (p.risk_level || 'Medium').toLowerCase();
    const card = document.createElement('div');
    card.className = 'msg-row aura';
    card.innerHTML = `<div class="msg-sender">Advisor</div>
      <div class="proposal-card">
        <div class="proposal-hd">
          <div><div class="proposal-type-tag">Investment Proposal</div><div class="proposal-name">${escapeHtml(p.asset || p.ticker || '')}</div></div>
          <div class="proposal-badges"><div class="proposal-badge ${ac}">${escapeHtml(p.action || 'BUY')}</div><div class="proposal-risk ${rc}">${escapeHtml(p.risk_level || 'Medium')} Risk</div></div>
        </div>
        <div class="proposal-levels">
          <div class="p-level"><div class="p-level-lbl">Entry</div><div class="p-level-val">${escapeHtml(p.entry || '—')}</div></div>
          <div class="p-level"><div class="p-level-lbl">Target</div><div class="p-level-val">${escapeHtml(p.target || '—')}</div></div>
          <div class="p-level"><div class="p-level-lbl">Stop</div><div class="p-level-val">${escapeHtml(p.stop_loss || '—')}</div></div>
          <div class="p-level"><div class="p-level-lbl">Horizon</div><div class="p-level-val">${escapeHtml(p.timeframe || '—')}</div></div>
        </div>
        <div class="proposal-rationale">${escapeHtml(p.rationale || '')}</div>
        <div class="proposal-actions">
          <button class="prop-btn accept">Accept · $1,000 paper</button>
          <button class="prop-btn dismiss">Dismiss</button>
        </div>
      </div><div class="msg-ts">${ts()}</div>`;
    card.querySelector('.prop-btn.accept').addEventListener('click', (e) => acceptProposal(p, e.currentTarget));
    card.querySelector('.prop-btn.dismiss').addEventListener('click', (e) => { e.currentTarget.closest('.msg-row').style.opacity = '.4'; });
    return card;
  }
  function appendRow(role, html, timestamp) {
    const feed = $('msg-feed');
    const row = document.createElement('div');
    row.className = `msg-row ${role}`;
    row.innerHTML = `<div class="msg-sender">${role === 'user' ? 'You' : 'Advisor'}</div><div class="bubble ${role}">${html}</div><div class="msg-ts">${timestamp || ts()}</div>`;
    feed.appendChild(row); feed.scrollTop = feed.scrollHeight;
    return row.querySelector('.bubble');
  }
  function appendTyping() {
    const feed = $('msg-feed');
    const row = document.createElement('div'); row.className = 'msg-row aura'; row.id = 'typing-row';
    row.innerHTML = `<div class="msg-sender">Advisor</div><div class="typing-bubble"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
    feed.appendChild(row); feed.scrollTop = feed.scrollHeight; return row;
  }
  function loadHistory() { try { chatHistory = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch (_) { chatHistory = []; } return chatHistory; }
  function saveHistory() { localStorage.setItem(HISTORY_KEY, JSON.stringify(chatHistory.slice(-40))); }
  function renderChatHistory() {
    const feed = $('msg-feed'); feed.innerHTML = '';
    chatHistory.filter(m => m.role !== 'system').forEach(m => appendRow(m.role === 'user' ? 'user' : 'aura', mdToHtml(m.content), m.ts));
  }
  function autoResize(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 120) + 'px'; }

  async function sendMessage(text) {
    if (!text || !text.trim()) return;
    const sendBtn = $('send-btn'), inputEl = $('chat-input');
    sendBtn.disabled = true; inputEl.value = ''; autoResize(inputEl);
    appendRow('user', escapeHtml(text));
    chatHistory.push({ role: 'user', content: text, ts: ts() });

    const messages = [{ role: 'system', content: buildSystemPrompt() },
      ...chatHistory.filter(m => m.role !== 'system').map(m => ({ role: m.role, content: m.content }))];
    const typingRow = appendTyping();
    let full = '', bubble = null;
    try {
      const res = await fetch(`${API}/ai/chat/stream`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ messages }) });
      if (!res.ok) throw new Error(res.status);
      const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '';
      while (true) {
        const { done, value } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n'); buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const pl = line.slice(6).trim();
          if (pl === '[DONE]') break;
          try {
            const ch = JSON.parse(pl);
            if (ch.error) throw new Error(ch.error);
            if (ch.content) {
              full += ch.content;
              if (!bubble) { typingRow.remove(); bubble = appendRow('aura', mdToHtml(full)); }
              else { bubble.innerHTML = mdToHtml(full); $('msg-feed').scrollTop = $('msg-feed').scrollHeight; }
            }
          } catch (_) {}
        }
      }
    } catch (e) {
      typingRow.remove();
      appendRow('aura', `<span style="color:var(--T3)">The advisor can't answer right now. Please try again in a minute.</span>`);
    }
    if (full) {
      const proposal = tryParseProposal(full);
      if (proposal) {
        if (bubble) bubble.closest('.msg-row')?.remove();
        const feed = $('msg-feed'); feed.appendChild(renderProposalCard(proposal)); feed.scrollTop = feed.scrollHeight;
      }
      chatHistory.push({ role: 'assistant', content: full, ts: ts() }); saveHistory();
    }
    sendBtn.disabled = false; inputEl.focus();
  }

  function bootGreeting() {
    const hour = new Date().getHours();
    const greet = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';
    const intro = `${greet}, Nishanth. Ask me why the forecast says what it does, what could go wrong, or how it fits your spending and savings.`;
    appendRow('aura', mdToHtml(intro));
  }

  // Expand dynamic, symbol-aware chips into full prompts.
  function expandPrompt(chip) {
    const dyn = chip.dataset.dyn;
    const p = state.prediction;
    const dir = p ? p.direction : 'current';
    const conf = p ? p.confidence + '%' : '';
    if (dyn === 'explain') return `Explain the agent's ${state.symbol} call in plain English: it's ${dir} at ${conf} calibrated confidence${p && !p.act ? ' but flagged NO-TRADE' : ''}. What's driving it?`;
    if (dyn === 'risk') return `What is the single biggest risk to the ${state.symbol} ${dir} call right now, given the ${p ? p.regime : ''} regime and the news sentiment?`;
    return chip.dataset.prompt || chip.textContent;
  }

  function initChat() {
    const inputEl = $('chat-input'), sendBtn = $('send-btn');
    loadHistory();
    if (chatHistory.length) renderChatHistory(); else bootGreeting();

    sendBtn.addEventListener('click', () => sendMessage(inputEl.value));
    inputEl.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(inputEl.value); } });
    inputEl.addEventListener('input', () => autoResize(inputEl));
    $('prompts-bar').addEventListener('click', e => {
      const chip = e.target.closest('.prompt-chip'); if (!chip) return;
      sendMessage(expandPrompt(chip));
    });
    $('prompts-bar').addEventListener('keydown', e => {
      const chip = e.target.closest('.prompt-chip'); if (!chip) return;
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sendMessage(expandPrompt(chip)); }
    });
    $('clear-btn').addEventListener('click', () => {
      chatHistory = []; localStorage.removeItem(HISTORY_KEY); $('msg-feed').innerHTML = ''; bootGreeting();
    });
  }

  // expose for cross-container use
  window.__advisor = { state, setSymbol, CHARTABLE };

  /* ─────────────── BOOT ─────────────── */
  window.addEventListener('DOMContentLoaded', () => {
    initChart();
    initLeaderboard();
    initChat();
  });

})();
