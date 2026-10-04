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
    const h = Math.max(wrap.clientHeight || 440, 240);
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
    const firstLabel = lineMode ? fmtDate(src[0].date) : fmtTime(candles[0].t);
    out += xlbl(PAD.l, firstLabel, 'start');
    if (hasPred) {
      out += xlbl(xNow, 'NOW', 'middle');
      if (state.prediction.target_date) out += xlbl(xEnd, fmtDate(state.prediction.target_date), 'end');
    } else {
      const lastLabel = lineMode ? fmtDate(src[src.length - 1].date) : fmtTime(candles[candles.length - 1].t);
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
      label = g.lineMode ? fmtDate(d.date) : fmtTime(d.t);
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
    chgEl.textContent = `${up ? '▲' : '▼'} ${Math.abs(pct).toFixed(2)}% · ${state.tf}`;
    chgEl.className = 'cl-chg ' + (up ? 'up' : 'down');
    dot.className = 'cl-dot ' + (state.live ? 'live' : (state.lineMode ? '' : 'stale'));
    lbl.textContent = state.live ? 'LIVE' : (state.lineMode ? 'DAILY' : 'STALE');
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
    const [okC, okF] = await Promise.all([loadCandles(), loadForecast(), loadVerdict()]);
    state.lineMode = !okC && state.series.length > 1;
    state.live = okC;
    if (okC || state.lineMode) {
      hideChartStatus();
      drawChart();
      updateLiveReadout();
    } else if (!quiet) {
      showChartStatus('No market data for this asset. Is the API running on :8000?', true, true);
    }
    // notify dependent containers (signal strip, track record, verifier label, chat ctx)
    renderSignalStrip();
    if (typeof renderTrust === 'function') renderTrust();
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
    if ($('vf-sym')) $('vf-sym').textContent = s;
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

  function renderSignalStrip() {
    const p = state.prediction;
    const set = (id, html, cls) => { const el = $(id); if (!el) return; el.innerHTML = html; if (cls != null) el.className = cls; };

    if (!p) {
      set('sig-dir', '—', 'sig-val neu');
      ['sig-conf', 'sig-size', 'sig-regime'].forEach(id => set(id, '—', 'sig-val neu'));
      set('sig-conf-sub', 'no model signal yet'); set('sig-size-sub', '—'); set('sig-sent', '—');
      set('sig-asof', '—');
      renderOutlook();
      return;
    }

    const anchor = liveAnchor();
    const fc = rebasedForecast(anchor);
    const up = p.direction !== 'DOWN';
    const col = up ? 'up' : 'down';

    // Direction
    set('sig-dir', `<span class="sig-badge ${col}">${up ? '▲' : '▼'} ${p.direction || (up ? 'UP' : 'DOWN')}</span>`, 'sig-val');

    // Verifier verdict tag (from the daily prediction cycle, if logged) + as-of stamp
    const v = state.verdict;
    let vfTag = '';
    if (v && v.label) {
      const vcls = v.label === 'VETO' ? 'veto' : (v.label === 'downgrade' ? 'downgrade' : 'agree');
      const vtxt = v.label === 'VETO' ? `veto → ${v.final_confidence}%`
        : v.label === 'downgrade' ? `↓ verifier ${v.final_confidence}%`
        : '✓ verifier agrees';
      vfTag = `<span class="sig-vf-tag ${vcls}" title="${escapeHtml(v.rationale || '')}">${vtxt}</span>`;
    }
    const asOf = p.generated_at ? `as of ${fmtDate(p.generated_at)} ${fmtTime(p.generated_at)}` : '—';
    set('sig-asof', vfTag + asOf);

    // Confidence (calibrated). Note base→calibrated downgrade if present.
    set('sig-conf', `${p.confidence != null ? p.confidence : '—'}%`, 'sig-val');
    const baseNote = (p.base_confidence != null && p.base_confidence !== p.confidence)
      ? `base ${p.base_confidence}% → calibrated` : 'calibrated';
    set('sig-conf-sub', baseNote);

    renderOutlook();

    // Position size — honest about the meta act-gate.
    if (p.act) {
      set('sig-size', (p.kelly_frac != null ? (p.kelly_frac * 100).toFixed(1) + '%' : '—'), 'sig-val');
      set('sig-size-sub', 'Kelly stake');
    } else {
      set('sig-size', 'No-trade', 'sig-val neu');
      set('sig-size-sub', 'below act gate');
    }

    // Regime + sentiment
    set('sig-regime', p.regime || '—', 'sig-val');
    const sl = p.sentiment_label || (p.sentiment != null ? (p.sentiment >= 0 ? 'positive' : 'negative') : null);
    const ss = p.sentiment != null ? ` ${p.sentiment >= 0 ? '+' : ''}${p.sentiment.toFixed(2)}` : '';
    set('sig-sent', sl ? `${sl}${ss}` : '—');
  }

  /* ══════════════════════════════════════════════════════════════
     5-DAY OUTLOOK — the calibrated price range, re-based to live.
     The range is the headline: its coverage is measured (80% / 90%
     out of fold). The central estimate is shown muted, with its
     average miss, until a retrain beats "no change".
     ══════════════════════════════════════════════════════════════ */
  function renderOutlook() {
    const body = $('ol-body'), meta = $('ol-meta');
    if (!body) return;
    const p = state.prediction;
    const anchor = liveAnchor();
    const fc = p ? rebasedForecast(anchor) : null;
    if ($('ol-h')) $('ol-h').textContent = p ? (p.horizon_days || 5) : 5;
    if (!p || !fc || p.conf_low == null || p.conf_high == null) {
      body.innerHTML = `<div class="ol-note">No price range for ${escapeHtml(state.symbol)} yet.</div>`;
      if (meta) meta.textContent = '—';
      return;
    }
    const pct = p.band_pct || 80;
    const chg = v => fmtPct((v / anchor - 1) * 100, 1);
    const skill = pointHasSkill();
    const pf = state.pointForecast || {};
    const miss = pf.mae != null ? `avg miss ±${(pf.mae * 100).toFixed(1)}%` : '';
    if (meta) meta.textContent = `${pct}% range · by ${fmtDate(p.target_date)}`;

    const has90 = fc.band90Low != null;
    let html = `<div class="ol-levels">
      <div class="ol-lvl">
        <span class="ol-lvl-lbl">Low</span>
        <span class="ol-lvl-val">${fmtPrice(fc.bandLow)}<small>${chg(fc.bandLow)}</small></span>
        ${has90 ? `<span class="ol-lvl-sub">90%: ${fmtPrice(fc.band90Low)} (${chg(fc.band90Low)})</span>` : ''}
      </div>
      <div class="ol-lvl${skill ? '' : ' muted'}">
        <span class="ol-lvl-lbl">Central estimate</span>
        <span class="ol-lvl-val">${fmtPrice(fc.projected)}<small>${chg(fc.projected)}</small></span>
        <span class="ol-lvl-sub">${skill ? miss : `not better than no-change${miss ? ' · ' + miss : ''}`}</span>
      </div>
      <div class="ol-lvl">
        <span class="ol-lvl-lbl">High</span>
        <span class="ol-lvl-val">${fmtPrice(fc.bandHigh)}<small>${chg(fc.bandHigh)}</small></span>
        ${has90 ? `<span class="ol-lvl-sub">90%: ${fmtPrice(fc.band90High)} (${chg(fc.band90High)})</span>` : ''}
      </div>
    </div>`;

    // Bar: the track spans the 90% range (or the 80% one padded), the inner block is the
    // 80% range, the tick is the live price.
    const lo = has90 ? fc.band90Low : fc.bandLow - (fc.bandHigh - fc.bandLow) * 0.15;
    const hi = has90 ? fc.band90High : fc.bandHigh + (fc.bandHigh - fc.bandLow) * 0.15;
    const span = hi - lo;
    if (span > 0) {
      const pos = v => Math.max(0, Math.min(100, (v - lo) / span * 100));
      html += `<div class="ol-bar" title="${pct}% range (inner) within the 90% range (track); tick = live price">
        <span class="ol-bar-80" style="left:${pos(fc.bandLow).toFixed(1)}%;right:${(100 - pos(fc.bandHigh)).toFixed(1)}%"></span>
        <span class="ol-bar-now" style="left:${pos(anchor).toFixed(1)}%"></span>
      </div>
      <div class="ol-bar-ends"><span>${has90 ? '90% low' : ''}</span><span>live ${fmtPrice(anchor)}</span><span>${has90 ? '90% high' : ''}</span></div>`;
    }
    html += `<div class="ol-note">In ${pct}% of past cases, the price ${p.horizon_days || 5} trading days later was inside this range. That rate is an average over all predictions, not a promise for this one.</div>`;
    body.innerHTML = html;
  }

  /* ══════════════════════════════════════════════════════════════
     CONTAINER 3 — CONVICTION LEADERBOARD
     The agent's highest-conviction calls across every symbol it
     covers. Click a row to load it into the chart.
     ══════════════════════════════════════════════════════════════ */
  let _leaderboard = [];
  let _verdicts = {}; // symbol -> latest verifier verdict (for VETO/downgrade badges)

  function renderLeaderboard() {
    const body = $('lb-body');
    if (!_leaderboard.length) { body.innerHTML = `<div class="mini-empty">No agent calls logged yet.<br>Trigger predictions on the API to populate.</div>`; return; }
    body.innerHTML = _leaderboard.map((r, i) => {
      const up = r.direction !== 'DOWN';
      const col = up ? UP : DOWN;
      const conf = r.confidence != null ? r.confidence : 0;
      const chartable = !!CHARTABLE[r.symbol];
      const v = _verdicts[r.symbol];
      let badge = '<span class="lb-verdict"></span>';
      if (v && v.label === 'VETO') badge = `<span class="lb-verdict veto" title="Verifier VETO → ${v.final_confidence}%: ${escapeHtml(v.rationale || '')}">⛔</span>`;
      else if (v && v.label === 'downgrade') badge = `<span class="lb-verdict downgrade" title="Verifier downgraded → ${v.final_confidence}%: ${escapeHtml(v.rationale || '')}">↓</span>`;
      return `<div class="lb-row${r.symbol === state.symbol ? ' active' : ''}" data-sym="${r.symbol}" title="${chartable ? 'Live chart' : 'Forecast (no live candles)'}">
        <span class="lb-rank">${i + 1}</span>
        <span class="lb-sym">${r.symbol}</span>
        <span class="lb-dir" style="color:${col}">${up ? '▲' : '▼'}</span>
        <span class="lb-bar-wrap"><span class="lb-bar" style="width:${conf}%;background:${col}"></span></span>
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
  /* ══════════════════════════════════════════════════════════════
     CONTAINER 4 — VERIFIER (2nd opinion)
     On-demand LLM risk-manager: red-teams the model's call against
     fresh news/RAG. Can only DOWNGRADE or VETO — never raise.
     ══════════════════════════════════════════════════════════════ */
  let _verifying = false;

  async function runVerifier() {
    if (_verifying) return;
    const sym = state.symbol;
    const btn = $('vf-btn'), body = $('vf-body');
    _verifying = true; btn.disabled = true; btn.textContent = '…';
    body.innerHTML = `<div class="mini-empty">Red-teaming <b>${sym}</b> against news &amp; context… <br>(LLM verifier, ~20s)</div>`;
    try {
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 60000);
      const r = await fetch(`${API}/predict/${sym}/verify`, { method: 'POST', signal: ctrl.signal });
      clearTimeout(to);
      if (!r.ok) throw new Error(r.status);
      const d = await r.json();
      renderVerdict(d.verdict || d, sym);
    } catch (e) {
      body.innerHTML = `<div class="mini-empty">Verifier offline — needs Ollama running.<br><code>ollama serve</code></div>`;
    } finally {
      _verifying = false; btn.disabled = false; btn.textContent = 'Run';
    }
  }

  function renderVerdict(v, sym) {
    const body = $('vf-body');
    if (!v) { body.innerHTML = `<div class="mini-empty">No prediction to verify for ${sym}.</div>`; return; }

    if (v.verifier === 'unavailable' || v.verifier === 'parse_error') {
      body.innerHTML = `<div class="vf-verdict">
        <div class="vf-top"><span class="vf-stamp downgrade">Unavailable</span>
          <span class="vf-conf">model held at <b>${v.model_confidence ?? '—'}%</b></span></div>
        <div class="vf-text">The LLM verifier ${v.verifier === 'unavailable' ? 'is offline' : 'returned unparseable output'}; the calibrated model signal stands unchanged.</div>
      </div>`;
      return;
    }

    const veto = !!v.veto;
    const agree = !!v.agree && !veto;
    const stampCls = veto ? 'veto' : (agree ? 'agree' : 'downgrade');
    const stampTxt = veto ? 'VETO' : (agree ? 'Agree' : 'Downgrade');
    const mc = v.model_confidence ?? v.confidence ?? '—';
    const fc = v.final_confidence ?? mc;
    const changed = fc !== mc;

    body.innerHTML = `<div class="vf-verdict">
      <div class="vf-top">
        <span class="vf-stamp ${stampCls}">${stampTxt}</span>
        <span class="vf-conf">${sym} · <b>${mc}%</b>${changed ? `<span class="arrow">→</span><b>${fc}%</b>` : ''}</span>
      </div>
      ${v.rationale ? `<div class="vf-text">${escapeHtml(v.rationale)}</div>` : ''}
      ${v.risks ? `<div class="vf-risk"><b>Top risk:</b> ${escapeHtml(v.risks)}</div>` : ''}
    </div>`;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function initVerifier() {
    $('vf-btn').addEventListener('click', runVerifier);
  }
  /* ══════════════════════════════════════════════════════════════
     CONTAINER 5 — MODEL TRUST & TRACK RECORD
     Global calibration (reliability curve) when data exists, else an
     honest "accumulating" state. Plus the active symbol's resolved
     win/loss timeline — never a fabricated curve.
     ══════════════════════════════════════════════════════════════ */
  let _calib = [];

  function calibSvg(buckets) {
    // Reliability curve: stated confidence (x) vs realized hit-rate (y), with the
    // perfect-calibration diagonal for reference.
    const W = 240, H = 80, p = 10;
    const sx = v => p + (v / 100) * (W - 2 * p);
    const sy = v => (H - p) - (v / 100) * (H - 2 * p);
    let s = `<svg class="calib-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
    s += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(100)}" y2="${sy(100)}" stroke="rgba(255,255,255,.18)" stroke-width="1" stroke-dasharray="3 3"/>`;
    const pts = buckets.filter(b => b.n > 0).map(b => [sx(b.stated_conf), sy(b.realized_hit * 100)]);
    if (pts.length > 1) {
      s += `<polyline points="${pts.map(p => p.map(n => n.toFixed(1)).join(',')).join(' ')}" fill="none" stroke="${UP}" stroke-width="2" stroke-linejoin="round"/>`;
    }
    pts.forEach(pt => { s += `<circle cx="${pt[0].toFixed(1)}" cy="${pt[1].toFixed(1)}" r="2.6" fill="${UP}"/>`; });
    s += `</svg>`;
    return s;
  }

  async function renderTrust() {
    const body = $('trust-body');
    const sym = state.symbol;
    let hist = null;
    try { hist = await getJSON(`${API}/predict/${sym}/history?limit=100`); } catch (e) { hist = null; }

    const rows = (hist && hist.history) || [];
    const resolved = rows.filter(r => r.correct !== null && r.correct !== undefined);
    const acc = hist && hist.realized_accuracy != null ? Math.round(hist.realized_accuracy * 100) : null;

    // Header number = realized hit-rate for this symbol (honest — '—' until resolved).
    let html = `<div class="trust-acc">
      <span class="trust-acc-val">${acc != null ? acc + '%' : '—'}</span>
      <span class="trust-acc-lbl">${sym} hit rate</span>
    </div>`;
    const bandHit = hist && hist.band_hit_rate != null ? Math.round(hist.band_hit_rate * 100) : null;
    html += `<div class="trust-note">Range hit rate: ${bandHit != null
      ? `<b style="color:var(--T1)">${bandHit}%</b> over the last ${hist.band_resolved} prediction${hist.band_resolved === 1 ? '' : 's'} (target ${state.prediction ? state.prediction.band_pct || 80 : 80}%)`
      : 'pending, needs matured predictions'}</div>`;
    html += `<div class="trust-note">${rows.length} call${rows.length === 1 ? '' : 's'} logged · <b style="color:var(--T1)">${resolved.length}</b> resolved${acc == null ? ' — accuracy pending' : ''}</div>`;

    // Global calibration: real curve if we have data, else honest accumulating note.
    if (_calib && _calib.some(b => b.n > 0)) {
      html += `<div class="trust-note" style="margin-bottom:4px;">Model reliability — stated vs realized</div>` + calibSvg(_calib);
    } else {
      html += `<div class="trust-note">Calibration curve is still accumulating: it needs predictions whose ${state.prediction ? (state.prediction.horizon_days || 5) : 5}-day horizon has matured and resolved.</div>`;
    }

    // Track strip — recent calls as win / loss / pending cells (newest last).
    if (rows.length) {
      const cells = rows.slice(0, 18).reverse().map(r => {
        if (r.correct === 1) return `<span class="track-cell win" title="hit">✓</span>`;
        if (r.correct === 0) return `<span class="track-cell loss" title="miss">✗</span>`;
        return `<span class="track-cell pending" title="awaiting outcome">·</span>`;
      }).join('');
      html += `<div class="track-strip">${cells}</div>`;
    }

    body.innerHTML = html;
  }

  async function initTrust() {
    try {
      const d = await getJSON(`${API}/predict/calibration`);
      _calib = d.buckets || [];
      const live = _calib.filter(b => b.n > 0).length;
      $('trust-meta').textContent = live ? `${live} buckets` : 'accumulating';
    } catch (e) {
      _calib = [];
      $('trust-meta').textContent = 'offline';
    }
    renderTrust();
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

    return `You are AURA, the AI advisor inside FLUX's prediction cockpit. You sit next to a quantitative trading agent and the live chart for ${state.symbol}.
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
    card.innerHTML = `<div class="msg-sender">AURA</div>
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
    row.innerHTML = `<div class="msg-sender">${role === 'user' ? 'You' : 'AURA'}</div><div class="bubble ${role}">${html}</div><div class="msg-ts">${timestamp || ts()}</div>`;
    feed.appendChild(row); feed.scrollTop = feed.scrollHeight;
    return row.querySelector('.bubble');
  }
  function appendTyping() {
    const feed = $('msg-feed');
    const row = document.createElement('div'); row.className = 'msg-row aura'; row.id = 'typing-row';
    row.innerHTML = `<div class="msg-sender">AURA</div><div class="typing-bubble"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
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
      appendRow('aura', `<span style="color:var(--T3)">AURA is offline — is Ollama running? <code>ollama serve</code></span>`);
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
    const intro = `${greet}, Nishanth. I'm **AURA**, wired into the prediction agent.\n\nI can see the live model call on whatever asset you load in the chart — ask me to **explain this call**, stress-test **the risk**, or compare assets. I also know your finances from the Payments page.`;
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

  /* ─────────────── HEADER: AGENT HEALTH ─────────────── */
  // Reflects whether the Ollama-backed LLM (verifier + AURA chat) is actually
  // reachable — the calibrated model signal itself doesn't depend on this.
  async function loadAgentHealth() {
    const dot = $('model-dot'), txt = $('model-badge-text');
    if (!dot || !txt) return;
    try {
      const d = await getJSON(`${API}/health`, 4000);
      const up = !!(d.agent && d.agent.available);
      dot.className = 'model-dot' + (up ? '' : ' offline');
      txt.textContent = up ? 'AURA v4.2 · Agent online' : 'AURA v4.2 · Agent offline';
    } catch (e) {
      dot.className = 'model-dot offline';
      txt.textContent = 'AURA v4.2 · API offline';
    }
  }

  // expose for cross-container use
  window.__advisor = { state, setSymbol, CHARTABLE };

  /* ─────────────── BOOT ─────────────── */
  window.addEventListener('DOMContentLoaded', () => {
    initChart();
    initLeaderboard();
    initVerifier();
    initTrust();
    initChat();
    loadAgentHealth();
    setInterval(loadAgentHealth, 60000);
  });

})();
