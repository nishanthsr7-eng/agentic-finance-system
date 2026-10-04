// Resolved once by js/flux-data.js (FLUX_CONFIG override → :8000 on
    // localhost → same-origin in production). Loaded before this script.
    const API = window.FLUX_API;
    const USER_ID = window.FLUX_USER_ID;

    /* ══════════════════════════════════════════════
       0. SHARED localStorage HELPERS
    ══════════════════════════════════════════════ */
    // Local-timezone YYYY-MM-DD key (toISOString would bucket evening
    // transactions onto the wrong day for IST and other non-UTC zones).
    function localDateKey(d) {
      const x = new Date(d);
      return x.getFullYear() + '-' + String(x.getMonth() + 1).padStart(2, '0') + '-' + String(x.getDate()).padStart(2, '0');
    }

    // Memoized — several widgets call this per refresh cycle and parsing
    // ~2000 transactions repeatedly is wasted work. Invalidated whenever
    // fresh DB data lands (refreshLocalDataWidgets).
    let _storageCache = null;
    function invalidateStorageCache() { _storageCache = null; }

    function getStorageData() {
      if (_storageCache) return _storageCache;
      const txs  = JSON.parse(localStorage.getItem('flux_transactions') || '[]');
      const now  = new Date();
      const prev = new Date(now.getFullYear(), now.getMonth() - 1, 1);
      const yrAgo    = new Date(now.getFullYear() - 1, now.getMonth(), now.getDate());
      const twoYrAgo = new Date(now.getFullYear() - 2, now.getMonth(), now.getDate());

      const isToday  = t => new Date(t.date).toDateString() === now.toDateString();
      const isMonth  = t => { const d = new Date(t.date); return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth(); };
      const isPrevMo = t => { const d = new Date(t.date); return d.getFullYear() === prev.getFullYear() && d.getMonth() === prev.getMonth(); };
      const isYear   = t => new Date(t.date) >= yrAgo;
      // Prior rolling-365d window so the year % compares like-for-like periods.
      const isPrevYr = t => { const d = new Date(t.date); return d >= twoYrAgo && d < yrAgo; };
      const net = arr => arr.reduce((s, t) => s + t.amount, 0);

      const todayNet  = net(txs.filter(isToday));
      const monthNet  = net(txs.filter(isMonth));
      const prevMoNet = net(txs.filter(isPrevMo));
      const yearNet   = net(txs.filter(isYear));
      const prevYearNet = net(txs.filter(isPrevYr));

      const thirtyAgo = new Date(now - 30 * 24 * 60 * 60 * 1000);
      const recent30  = txs.filter(t => new Date(t.date) >= thirtyAgo && t.amount < 0);
      const catTotals = {};
      recent30.forEach(t => {
        const c = (t.category || 'other').toLowerCase();
        catTotals[c] = (catTotals[c] || 0) + Math.abs(t.amount);
      });

      const invest     = (catTotals['investment'] || catTotals['invest'] || 0);
      const essentials = (catTotals['essentials'] || catTotals['essential'] || catTotals['utilities'] ||
                          catTotals['rent'] || catTotals['groceries'] || catTotals['grocery'] || 0);
      const lifestyle  = (catTotals['lifestyle'] || catTotals['food'] || catTotals['dining'] ||
                          catTotals['entertainment'] || catTotals['subscription'] || catTotals['travel'] || 0);
      const other30    = Object.entries(catTotals)
        .filter(([k]) => !['investment','invest','essentials','essential','utilities','rent',
                           'groceries','grocery','lifestyle','food','dining','entertainment',
                           'subscription','travel'].includes(k))
        .reduce((s, [, v]) => s + v, 0);
      const catTotal = invest + essentials + lifestyle + other30 || 1;

      const dailySpend = {};
      txs.filter(t => t.amount < 0).forEach(t => {
        const key = localDateKey(t.date);
        dailySpend[key] = (dailySpend[key] || 0) + Math.abs(t.amount);
      });

      _storageCache = {
        txs, todayNet, monthNet, prevMoNet, yearNet, prevYearNet,
        invest, essentials, lifestyle, other30, catTotal,
        dailySpend, hasTxData: txs.length > 0
      };
      return _storageCache;
    }

    /* ══════════════════════════════════════════════
       1. CANDLESTICK CHART — live yfinance data
    ══════════════════════════════════════════════ */
    let DAILY_HISTORY = [];
    let allCandles = [];
    let chartW = 900, chartH = 260;
    let currentTimeframe = '1D';
    let currentAsset = 'btc';

    const mainSVG   = document.getElementById('main-svg');
    const candleGrp = document.getElementById('candle-group');
    const yGrid     = document.getElementById('y-grid');
    const xLabels   = document.getElementById('x-labels');
    const livePriceEl = document.getElementById('live-price');
    const liveChgEl   = document.getElementById('live-chg');

    const ASSETS = {
      btc:   { name: 'BTC / CRYPTO',  icon: '₿', desc: 'Sovereign Digital Asset',   fmtPrefix: '$',  fmtLocale: 'en-US' },
      eth:   { name: 'ETH / CRYPTO',  icon: 'Ξ', desc: 'Smart Contract Utility',    fmtPrefix: '$',  fmtLocale: 'en-US' },
    };

    function fmtPrice(val) {
      const a = ASSETS[currentAsset];
      return a.fmtPrefix + val.toLocaleString(a.fmtLocale, { maximumFractionDigits: 2 });
    }

    // Header badge tells the truth: LIVE only while market data is flowing.
    // On an OFFLINE→LIVE transition (the 60s candle poll succeeding again),
    // re-hydrate the DB-driven widgets that failed while the backend was down.
    let badgeState = null;
    function setLiveBadge(ok) {
      const badge = document.getElementById('live-badge');
      const dot   = document.getElementById('live-badge-dot');
      const text  = document.getElementById('live-badge-text');
      if (!badge || !dot || !text) return;
      if (ok && badgeState === false) {
        window.FluxHydrate?.refresh();        // re-pulls /db/* → flux:data-updated
        if (typeof intelRefresh === 'function') intelRefresh();
      }
      badgeState = ok;
      if (ok) {
        badge.style.color = 'var(--BRIGHT)';
        dot.style.background = 'var(--A)';
        dot.style.boxShadow = '0 0 6px var(--A)';
        dot.style.animation = 'pd 2s ease-in-out infinite';
        text.textContent = 'LIVE';
      } else {
        badge.style.color = 'var(--T3)';
        dot.style.background = 'var(--T3)';
        dot.style.boxShadow = 'none';
        dot.style.animation = 'none';
        text.textContent = 'OFFLINE';
      }
    }

    async function loadCandles(asset, tf) {
      try {
        const res = await fetch(`${API}/market/candles/${asset}?tf=${tf}`);
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();
        return data.candles;   // [{t,o,h,l,c,v}, ...]
      } catch (e) {
        console.warn('loadCandles failed:', e);
        return null;
      }
    }

    // Match the SVG coordinate system to the rendered pixel size so text and
    // strokes are never stretched (the old fixed 900×280 viewBox with
    // preserveAspectRatio="none" distorted axis labels).
    function syncChartViewport() {
      const area = mainSVG.parentElement;
      const w = Math.max(area.clientWidth || 900, 320);
      const h = Math.max(area.clientHeight || 280, 200);
      chartW = w;
      chartH = h - 20; // reserve a strip below the plot for x-axis labels
      mainSVG.setAttribute('viewBox', `0 0 ${w} ${h}`);
      const clip = document.getElementById('chart-clip-rect');
      if (clip) { clip.setAttribute('width', w); clip.setAttribute('height', chartH); }
      const hov = document.getElementById('hover-area');
      if (hov) { hov.setAttribute('width', w); hov.setAttribute('height', chartH); }
      chV.setAttribute('y2', chartH);
      chH.setAttribute('x2', chartW);
    }

    function drawChart() {
      const slice = allCandles;
      if (slice.length < 2) return;
      syncChartViewport();
      const N = slice.length;

      const allHi = slice.map(c => c.h);
      const allLo = slice.map(c => c.l);
      // Use reduce to avoid spread stack-overflow on large arrays
      const maxP  = allHi.reduce((m, v) => v > m ? v : m, -Infinity);
      const minP  = allLo.reduce((m, v) => v < m ? v : m, Infinity);
      const range = maxP - minP || 1;
      const pad   = { t: 20, b: 30, l: 10, r: 70 };
      const pw = chartW - pad.l - pad.r;
      const ph = chartH - pad.t - pad.b;

      const px = i => pad.l + (i / (N - 1)) * pw;
      const py = v => pad.t + ph - ((v - minP) / range) * ph;

      // Y-grid
      let yHtml = '';
      for (let i = 0; i <= 5; i++) {
        const val = minP + (range / 5) * i;
        yHtml += `<text x="${chartW - pad.r + 5}" y="${py(val) + 4}" font-family="var(--F)"
          font-size="10" fill="rgba(255,255,255,.5)">${fmtPrice(val)}</text>`;
      }
      yGrid.innerHTML = yHtml;

      // Candlesticks
      const cW = Math.max(pw / N * 0.6, 2);
      let cHtml = '';
      slice.forEach((c, i) => {
        const x  = px(i), yO = py(c.o), yC = py(c.c), yH = py(c.h), yL = py(c.l);
        const bull = c.c >= c.o;
        const bodyH = Math.max(Math.abs(yO - yC), 1);
        cHtml += `<line x1="${x}" x2="${x}" y1="${yH}" y2="${yL}" stroke="${bull ? '#fff' : '#666'}" stroke-width=".5" opacity=".6"/>
        <rect id="candle-${i}" x="${x - cW / 2}" y="${Math.min(yO, yC)}" width="${cW}" height="${bodyH}"
          fill="${bull ? 'rgba(255,255,255,0.8)' : 'rgba(100,100,100,0.6)'}" rx="0.5" style="transition:opacity .2s"/>`;
      });
      candleGrp.innerHTML = cHtml;

      // Token bar: last close & period change
      const lastPrice  = slice[slice.length - 1].c;
      const firstPrice = slice[0].o;
      const pctChg = ((lastPrice - firstPrice) / firstPrice * 100).toFixed(2);
      livePriceEl.textContent = fmtPrice(lastPrice);
      // Suffix the change with its window — without it a 1Y change reads as
      // an intraday move next to the live price.
      if (parseFloat(pctChg) >= 0) {
        liveChgEl.textContent = `▲ ${pctChg}% · ${currentTimeframe}`;
        liveChgEl.className   = 'token-change-up';
        liveChgEl.style.color = '';
      } else {
        liveChgEl.textContent = `▼ ${Math.abs(pctChg)}% · ${currentTimeframe}`;
        liveChgEl.className   = 'token-change-down';
        liveChgEl.style.color = '';
      }

      // X-axis labels (6 evenly spaced)
      let xlHtml = '';
      const lblStep = Math.ceil(N / 6);
      slice.forEach((c, i) => {
        if (i % lblStep !== 0 && i !== N - 1) return;
        const d   = new Date(c.t);
        let label = '';
        if (currentTimeframe === '1D') {
          label = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } else if (currentTimeframe === '7D') {
          label = d.toLocaleString('default', { weekday: 'short', hour: '2-digit' });
        } else if (currentTimeframe === '1M') {
          label = d.getDate() + ' ' + d.toLocaleString('default', { month: 'short' });
        } else {
          label = d.toLocaleString('default', { month: 'short', year: '2-digit' });
        }
        xlHtml += `<text x="${px(i)}" y="${chartH + 14}" text-anchor="middle"
          font-family="var(--F)" font-size="10" fill="rgba(255,255,255,.5)">${label}</text>`;
      });
      xLabels.innerHTML = xlHtml;
    }

    async function reloadChart(asset, tf, quiet = false) {
      if (!quiet) mainSVG.style.opacity = '0.4';
      const candles = await loadCandles(asset, tf);
      mainSVG.style.opacity = '1';
      if (candles && candles.length > 1) {
        allCandles = candles;
        drawChart();
        setLiveBadge(true);
      } else if (!quiet || !allCandles.length) {
        // Never leave a stale/fake price next to an empty chart.
        allCandles = [];
        livePriceEl.textContent = '—';
        liveChgEl.textContent = '';
        syncChartViewport();
        const cx = chartW / 2, cy = chartH / 2;
        candleGrp.innerHTML = `
          <text x="${cx}" y="${cy - 14}" text-anchor="middle" font-family="var(--F)" font-size="12" fill="rgba(255,255,255,.45)">No data available</text>
          <g id="chart-retry" style="cursor:pointer;">
            <rect x="${cx - 40}" y="${cy + 2}" width="80" height="24" rx="6" fill="rgba(255,255,255,.06)" stroke="rgba(255,255,255,.15)"/>
            <text x="${cx}" y="${cy + 18}" text-anchor="middle" font-family="var(--F)" font-size="11" fill="rgba(255,255,255,.7)">Retry</text>
          </g>`;
        document.getElementById('chart-retry')?.addEventListener('click', () => reloadChart(currentAsset, currentTimeframe));
        setLiveBadge(false);
      }
    }

    // ── Hover crosshair ──
    const hoverArea = document.getElementById('hover-area');
    const chV = document.getElementById('ch-v'), chH = document.getElementById('ch-h');
    const chTipGrp = document.getElementById('ch-tip-grp');
    const chTip    = document.getElementById('ch-tip');

    hoverArea.addEventListener('mousemove', e => {
      if (!allCandles.length) return;
      const rect  = mainSVG.getBoundingClientRect();
      const mx    = (e.clientX - rect.left) * (chartW / rect.width);
      const N     = allCandles.length;
      const pad   = { t: 20, b: 30, l: 10, r: 70 };
      const pw    = chartW - pad.l - pad.r;
      const ph    = chartH - pad.t - pad.b;
      const maxP  = allCandles.reduce((m, c) => c.h > m ? c.h : m, -Infinity);
      const minP  = allCandles.reduce((m, c) => c.l < m ? c.l : m, Infinity);
      const range = maxP - minP || 1;

      const idx     = Math.max(0, Math.min(N - 1, Math.round(((mx - pad.l) / pw) * (N - 1))));
      const c       = allCandles[idx];
      const cx      = pad.l + (idx / (N - 1)) * pw;
      const cy      = pad.t + ph - ((c.c - minP) / range) * ph;

      chV.setAttribute('x1', cx); chV.setAttribute('x2', cx); chV.setAttribute('opacity', 1);
      chH.setAttribute('y1', cy); chH.setAttribute('y2', cy); chH.setAttribute('opacity', 1);
      chTipGrp.setAttribute('transform', `translate(${Math.min(cx + 8, chartW - 85)}, ${Math.max(cy - 30, 10)})`);
      chTipGrp.setAttribute('opacity', 1);
      chTip.textContent = fmtPrice(c.c);
    });

    hoverArea.addEventListener('mouseleave', () => {
      [chV, chH, chTipGrp].forEach(el => el.setAttribute('opacity', 0));
    });

    /* ══════════════════════════════════════════════
       2. PORTFOLIO TRANSACTION TREND GRAPH
    ══════════════════════════════════════════════ */
    function buildTxGraph() {
      const svg  = document.getElementById('tx-svg');
      const path = document.getElementById('tx-path');
      const area = document.getElementById('tx-area');
      const W = 310, H = 100, pad = 10;
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

      const data = DAILY_HISTORY.length > 1 ? DAILY_HISTORY : [0, 0];
      const min  = Math.min(...data);
      const max  = Math.max(...data);
      const rng  = max - min || 1;
      const step = W / (data.length - 1);
      const drawH = H - pad * 2;

      let d = '', ad = '';
      data.forEach((v, i) => {
        const x = i * step, y = pad + drawH - ((v - min) / rng) * drawH;
        if (i === 0) { d = `M ${x} ${y}`; ad = `M ${x} ${H} L ${x} ${y}`; }
        else { d += ` L ${x} ${y}`; ad += ` L ${x} ${y}`; }
      });
      ad += ` L ${W} ${H} Z`;
      path.setAttribute('d', d);
      area.setAttribute('d', ad);
    }

    /* ══════════════════════════════════════════════
       3. HEATMAP — real daily spend from localStorage
    ══════════════════════════════════════════════ */
    function buildHeatmap() {
      const el = document.getElementById('heatmap');
      let tooltip = document.getElementById('heatmap-tooltip');
      if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'heatmap-tooltip';
        document.body.appendChild(tooltip);
      }

      const { dailySpend, hasTxData } = getStorageData();
      const now = new Date();
      DAILY_HISTORY = [];
      let grandTotal = 0, html = '';

      const days = [];
      for (let i = 0; i < 30; i++) {
        const date = new Date(now);
        date.setDate(now.getDate() - (29 - i));
        const key   = localDateKey(date);
        const total = hasTxData ? Math.round(dailySpend[key] || 0) : 0;
        DAILY_HISTORY.push(total);
        grandTotal += total;
        days.push({ date, total });
      }

      const maxDay = Math.max(...days.map(d => d.total), 1);
      days.forEach(({ date, total }) => {
        const intensity = total / maxDay;
        const alpha = 0.05 + intensity * 0.85;
        const col = intensity > 0.7
          ? `rgba(0,229,160,${alpha.toFixed(2)})`
          : intensity > 0.3
            ? `rgba(0,229,160,${(alpha * 0.5).toFixed(2)})`
            : `rgba(255,255,255,.04)`;
        const dateStr = date.toLocaleString('default', { month: 'short', day: 'numeric' });
        const srLabel = `${dateStr}: ${total > 0 ? '₹' + total.toLocaleString('en-IN') + ' spent' : 'no spend'}`;
        html += `<div class="hm-cell" role="img" aria-label="${srLabel}" style="background:${col};aspect-ratio:1/1;border-radius:4px;" data-tx='${JSON.stringify({ d: dateStr, t: total })}'></div>`;
      });

      el.innerHTML = html;

      const totalEl = document.getElementById('summary-total');
      if (totalEl) totalEl.textContent = '₹' + grandTotal.toLocaleString('en-IN');

      const trendEl = document.getElementById('summary-trend');
      if (trendEl) {
        const first = DAILY_HISTORY.slice(0, 15).reduce((s, v) => s + v, 0);
        const last  = DAILY_HISTORY.slice(15).reduce((s, v) => s + v, 0);
        const rising  = last > first * 1.1;
        const falling = last < first * 0.9;
        trendEl.textContent = rising ? '↑ Rising' : falling ? '↓ Falling' : '→ Stable';
        trendEl.style.color = rising ? 'var(--A)' : falling ? 'var(--T2)' : 'var(--T3)';
      }

      el.querySelectorAll('.hm-cell').forEach(cell => {
        cell.addEventListener('mouseenter', e => {
          const data = JSON.parse(e.target.dataset.tx);
          const valStr = data.t > 0 ? `₹${data.t.toLocaleString('en-IN')}` : 'No spend';
          tooltip.innerHTML = `<div class="tt-total" style="border:none;margin:0;padding:0;">${data.d}: ${valStr}</div>`;
          tooltip.style.display = 'block';
        });
        cell.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
        cell.addEventListener('mousemove', e => {
          tooltip.style.left = (e.pageX + 15) + 'px';
          tooltip.style.top  = (e.pageY + 15) + 'px';
        });
      });
    }

    /* ══════════════════════════════════════════════
       4. KINETIC TICKER FEED — real transactions
    ══════════════════════════════════════════════ */
    // Live Ledger now shows the BUY/SELL trade book for stocks available in the
    // marketplace (served from MySQL via /db/trades). Fetched once and cached.
    let LEDGER_TRADES = null;
    let tickerObserver = null;

    const CCY_SYMBOL = { USD: '$', INR: '₹', EUR: '€', GBP: '£' };

    async function loadLedgerTrades() {
      if (LEDGER_TRADES) return LEDGER_TRADES;
      try {
        const res = await fetch(`${API}/db/trades?limit=200`);
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();
        LEDGER_TRADES = data.trades || [];
      } catch (e) {
        console.warn('[FLUX] Live Ledger trades fetch failed:', e.message);
        LEDGER_TRADES = [];
      }
      return LEDGER_TRADES;
    }

    async function buildTickerFeed() {
      const el = document.getElementById('vol-ticker');
      if (!el) return;
      const trades = await loadLedgerTrades();
      const countEl = document.getElementById('ticker-count');

      let cards;
      if (trades.length) {
        cards = trades.slice(0, 100).map(t => {
          const ccy = CCY_SYMBOL[t.currency] || '$';
          const qty = Number(t.quantity);
          return {
            side:  t.side,                                  // BUY | SELL
            asset: `${t.symbol} · ${t.name}`,
            qty:   `${qty % 1 === 0 ? qty : qty.toFixed(2)} sh @ ${ccy}${Number(t.price).toLocaleString('en-US', { maximumFractionDigits: 2 })}`,
            ccy,
            val:   Number(t.amount).toLocaleString('en-US', { maximumFractionDigits: 2 }),
            time:  new Date(t.trade_date).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }),
          };
        });
        if (countEl) {
          const dates = trades.map(t => new Date(t.trade_date)).filter(d => !isNaN(d));
          const fmt = d => d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: '2-digit' });
          // Flag when amounts mix currencies so values aren't read as one unit.
          const ccys = [...new Set(trades.map(t => CCY_SYMBOL[t.currency] || '$'))];
          const ccyNote = ccys.length > 1 ? ` · ${ccys.join('/')}` : '';
          countEl.textContent = (dates.length
            ? `${trades.length} TRADES · ${fmt(new Date(Math.min(...dates)))}–${fmt(new Date(Math.max(...dates)))}`
            : `${trades.length} TRADES`) + ccyNote;
        }
      } else {
        cards = [{ side: 'BUY', asset: 'No trades yet — trade stocks in the Marketplace', qty: '—', ccy: '$', val: '0', time: '—' }];
        if (countEl) countEl.textContent = '—';
      }

      // How many cards fit in the visible container area (approx 84px per card)
      const visibleCount = Math.ceil((el.clientHeight || 295) / 84) + 1;
      el.innerHTML = cards.map((t, i) => {
        const cls = t.side === 'BUY' ? 'buy' : 'sell';
        return `
        <div class="tx-ticker-card ${cls}${i < visibleCount ? ' active' : ''}">
          <div class="tx-ticker-status ${cls}">${escapeHtml(t.side)}</div>
          <div class="tx-ticker-asset">${escapeHtml(t.asset)}</div>
          <div class="tx-ticker-meta">
            <span style="color:var(--T3);font-size:10px;letter-spacing:.5px;">${escapeHtml(t.qty)}</span>
            <span>${escapeHtml(t.ccy)}<span class="tx-ticker-val">${escapeHtml(t.val)}</span> · ${escapeHtml(t.time)}</span>
          </div>
        </div>`;
      }).join('');

      // Shrink the observed area to the unmasked center (the ::before gradient
      // fades out the top/bottom ~15%) so cards don't get full "active" styling
      // (shadow/border/scale) while still visually clipped by the mask.
      // Disconnect the previous observer — this runs again on every data
      // refresh and the old one would otherwise leak with its dead targets.
      if (tickerObserver) tickerObserver.disconnect();
      tickerObserver = new IntersectionObserver(entries => {
        entries.forEach(e => e.target.classList.toggle('active', e.isIntersecting));
      }, { root: el, rootMargin: '-15% 0px -15% 0px', threshold: 1.0 });
      el.querySelectorAll('.tx-ticker-card').forEach(c => tickerObserver.observe(c));
    }

    /* ══════════════════════════════════════════════
       5. SPENDING DONUT — real category split
    ══════════════════════════════════════════════ */
    function buildUnlockDonut() {
      const arcs   = [1,2,3,4].map(n => document.getElementById('u-arc' + n));
      const center = document.querySelector('.unlock-pct');
      const C      = 201; // 2π × 32 ≈ 201.06
      const { invest, essentials, lifestyle, other30, catTotal, hasTxData } = getStorageData();

      const subEl = document.querySelector('.unlock-sub');
      if (!hasTxData || catTotal <= 0) {
        // Honest empty state — never render fabricated percentages.
        if (center) center.textContent = '—';
        if (subEl)  subEl.textContent  = 'No data';
        arcs.forEach(arc => { if (arc) { arc.style.strokeDasharray = `0 ${C}`; } });
        ['donut-pct-invest','donut-pct-essentials','donut-pct-lifestyle','donut-pct-other']
          .forEach(id => { const el = document.getElementById(id); if (el) el.textContent = '—'; });
        return;
      }

      const pctI = Math.round(invest     / catTotal * 100);
      const pctE = Math.round(essentials / catTotal * 100);
      const pctL = Math.round(lifestyle  / catTotal * 100);
      const pctO = Math.max(0, 100 - pctI - pctE - pctL);
      // Show top category % in center
      const topPct = Math.max(pctI, pctE, pctL, pctO);
      if (center) center.textContent = topPct + '%';
      const topLabel = topPct === pctI ? 'Invest' : topPct === pctE ? 'Essentials' : topPct === pctL ? 'Lifestyle' : 'Other';
      if (subEl) subEl.textContent = topLabel;

      // stroke-dasharray: segLen (C - segLen)
      // stroke-dashoffset: C - cumulativeStart  (positions each arc after the previous ones)
      const segments = [pctI, pctE, pctL, pctO];
      let cumulative = 0;
      segments.forEach((pct, i) => {
        const arc = arcs[i];
        if (!arc) return;
        const len   = (pct / 100) * C;
        const start = cumulative;
        arc.style.strokeDasharray  = `${len.toFixed(2)} ${(C - len).toFixed(2)}`;
        arc.style.strokeDashoffset = (C - start).toFixed(2);
        cumulative += len;
      });

      const setL = (id, pct) => { const el = document.getElementById(id); if (el) el.textContent = pct + '%'; };
      setL('donut-pct-invest', pctI); setL('donut-pct-essentials', pctE);
      setL('donut-pct-lifestyle', pctL); setL('donut-pct-other', pctO);
    }

    /* ══════════════════════════════════════════════
       6. PORTFOLIO P&L PANEL — from localStorage
    ══════════════════════════════════════════════ */
    function updatePortfolioPnl(final = false) {
      const pnlIds = ['pnl-today', 'pnl-month', 'pnl-year'];
      const { txs, todayNet, monthNet, prevMoNet, yearNet, prevYearNet, hasTxData } = getStorageData();
      if (!hasTxData) {
        if (final) {
          // Hydration finished with no data — resolve to an honest empty state
          // instead of shimmering forever.
          pnlIds.forEach(id => {
            const el = document.getElementById(id);
            if (el) { el.classList.remove('skeleton'); el.textContent = '₹0'; }
          });
          ['pnl-today-chg', 'pnl-month-chg', 'pnl-year-chg'].forEach(id => {
            const el = document.getElementById(id);
            if (el) { el.className = 'p-num-chg'; el.style.color = 'var(--T3)'; el.textContent = 'No transactions yet'; }
          });
        } else {
          // Show skeleton shimmer while hydration is still in flight
          pnlIds.forEach(id => { const el = document.getElementById(id); if (el) el.classList.add('skeleton'); });
        }
        return;
      }
      pnlIds.forEach(id => { const el = document.getElementById(id); if (el) el.classList.remove('skeleton'); });

      const fmt = v => {
        const abs = Math.abs(v);
        const str = abs >= 100000 ? '₹' + (abs / 100000).toFixed(2) + 'L'
                  : abs >= 1000   ? '₹' + Math.round(abs).toLocaleString('en-IN')
                  : '₹' + Math.round(abs);
        return (v < 0 ? '−' : '') + str;
      };

      const setChg = (id, val, base) => {
        const el = document.getElementById(id);
        if (!el) return;
        if (base !== 0) {
          const pct = ((val - base) / Math.abs(base) * 100).toFixed(1);
          const up  = parseFloat(pct) >= 0;
          el.className  = 'p-num-chg ' + (up ? 'up' : 'down');
          el.textContent = (up ? '▲ ' : '▼ ') + Math.abs(pct) + '%';
        } else {
          el.className  = 'p-num-chg'; el.textContent = val === 0 ? '— no activity' : '— new period';
          el.style.color = 'var(--T3)';
        }
      };

      // "Today" — if no transactions today, fall back to most recent active day
      const todayEl  = document.getElementById('pnl-today');
      const todayLbl = document.querySelector('.p-num-lbl');
      let displayNet = todayNet;
      if (todayNet === 0 && txs.length > 0) {
        const sorted = [...txs].sort((a, b) => new Date(b.date) - new Date(a.date));
        const lastDate = new Date(sorted[0].date).toDateString();
        displayNet = sorted.filter(t => new Date(t.date).toDateString() === lastDate)
                           .reduce((s, t) => s + t.amount, 0);
        if (todayLbl) todayLbl.textContent = new Date(sorted[0].date)
          .toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
      }
      if (todayEl) todayEl.textContent = fmt(displayNet);
      const yesterday = new Date(); yesterday.setDate(yesterday.getDate() - 1);
      const ydNet = txs.filter(t => new Date(t.date).toDateString() === yesterday.toDateString())
                       .reduce((s, t) => s + t.amount, 0);
      setChg('pnl-today-chg', displayNet, ydNet);

      const monthEl = document.getElementById('pnl-month');
      if (monthEl) monthEl.textContent = fmt(monthNet);
      setChg('pnl-month-chg', monthNet, prevMoNet);

      const yearEl = document.getElementById('pnl-year');
      if (yearEl) yearEl.textContent = fmt(yearNet);
      // prevYearNet is the prior rolling-365d window (like-for-like comparison)
      setChg('pnl-year-chg', yearNet, prevYearNet);
    }

    /* ══════════════════════════════════════════════
       7. ASSET ALLOCATION 24h% — from /market/quotes
    ══════════════════════════════════════════════ */
    async function updateAssetAllocation() {
      // Resolve a 24h-change cell to '—' instead of leaving the '…' loading
      // placeholder up forever when a quote fetch fails.
      const markUnavailable = (...ids) => ids.forEach(id => {
        const el = document.getElementById(id);
        const valSpan = el?.querySelector('span:last-child');
        if (valSpan && valSpan.textContent.trim() === '…') valSpan.textContent = '—';
      });

      // BTC + ETH from CoinGecko via existing quotes endpoint
      try {
        const res  = await fetch(`${API}/market/quotes/crypto`);
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();
        const bySymbol = {};
        (data.assets || []).forEach(a => { bySymbol[a.sub] = a; });

        const btcChg = bySymbol['BTC']?.change_pct;
        const ethChg = bySymbol['ETH']?.change_pct;

        const renderChg = (el, pct) => {
          if (el == null || pct == null) return;
          const up  = pct >= 0;
          const valSpan = el.querySelector('span:last-child') || el;
          valSpan.textContent = (up ? '↑ ' : '↓ ') + Math.abs(pct).toFixed(2) + '%';
          valSpan.style.color  = up ? 'var(--A)' : 'var(--RED)';
        };
        renderChg(document.getElementById('alloc-chg-btc'), btcChg);
        renderChg(document.getElementById('alloc-chg-eth'), ethChg);
        markUnavailable('alloc-chg-btc', 'alloc-chg-eth');
      } catch (e) {
        console.warn('updateAssetAllocation crypto failed:', e);
        markUnavailable('alloc-chg-btc', 'alloc-chg-eth');
      }

      // SPY from candles endpoint (compare first open to last close of 1D)
      try {
        const res = await fetch(`${API}/market/candles/spy?tf=1D`);
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();
        const c = data.candles || [];
        if (c.length > 1) {
          const pct = ((c[c.length - 1].c - c[0].o) / c[0].o * 100);
          const el  = document.getElementById('alloc-chg-spy');
          if (el) {
            const valSpan = el.querySelector('span:last-child') || el;
            valSpan.textContent = (pct >= 0 ? '↑ ' : '↓ ') + Math.abs(pct).toFixed(2) + '%';
            valSpan.style.color  = pct >= 0 ? 'var(--A)' : 'var(--RED)';
          }
        }
      } catch (e) {
        console.warn('updateAssetAllocation spy failed:', e);
      } finally {
        markUnavailable('alloc-chg-spy');
      }
    }

    /* ══════════════════════════════════════════════
       8. CHIP + ASSET SWITCHING
    ══════════════════════════════════════════════ */
    document.getElementById('asset-selector').addEventListener('change', async (e) => {
      const key = e.target.value;
      currentAsset = key;
      document.getElementById('asset-name').textContent = ASSETS[key].name;
      document.getElementById('asset-icon').textContent = ASSETS[key].icon;
      document.getElementById('asset-desc').textContent = ASSETS[key].desc;
      await reloadChart(key, currentTimeframe);
    });

    document.querySelectorAll('.chips').forEach(group => {
      group.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', async () => {
          group.querySelectorAll('.chip').forEach(c => {
            c.classList.remove('active');
            c.setAttribute('aria-pressed', 'false');
          });
          chip.classList.add('active');
          chip.setAttribute('aria-pressed', 'true');
          currentTimeframe = chip.textContent.trim();
          await reloadChart(currentAsset, currentTimeframe);
        });
      });
    });

    /* ══════════════════════════════════════════════
       9. NEURAL INTELLIGENCE WATERFALL
       Feeds from /ai/insights (real DB-stored data).
    ══════════════════════════════════════════════ */
    let intelRefresh = null;   // set by initNeuralIntelligence, reused on recovery
    async function initNeuralIntelligence() {
      const stream      = document.getElementById('intel-stream');
      const confEl      = document.getElementById('intel-conf-val');
      const impactEl    = document.getElementById('intel-impact-sum');
      const bullCountEl = document.getElementById('intel-bullish-count');
      const updatedEl   = document.getElementById('intel-updated-val');

      async function fetchAndRender() {
        let insights = [];
        let backendOk = false;
        try {
          const res = await fetch(`${API}/ai/insights?limit=20`);
          if (res.ok) { insights = (await res.json()).insights || []; backendOk = true; }
        } catch (_) {}
        // Update sync status indicator based on backend availability
        const syncDot   = document.getElementById('intel-sync-dot');
        const syncLabel = document.getElementById('intel-sync-label');
        if (syncDot && syncLabel) {
          if (backendOk) {
            syncDot.style.background = 'var(--A)';
            syncLabel.textContent = 'SYNC_STATUS: ACTIVE';
            syncLabel.style.color = '';
          } else {
            syncDot.style.background = 'var(--T3)';
            syncLabel.textContent = 'SYNC_STATUS: OFFLINE';
            syncLabel.style.color = 'var(--T3)';
          }
        }

        if (!insights.length) {
          // Honest empty state — the fetch finished, so don't shimmer forever.
          const msg = backendOk
            ? 'No AI insights generated yet.<br>New signals appear here automatically.'
            : 'Backend unreachable.<br>Retrying automatically every 5 minutes.';
          stream.innerHTML = `
            <div style="padding:48px 24px;display:flex;flex-direction:column;align-items:center;gap:14px;color:var(--T3);font-size:12px;text-align:center;line-height:1.7;">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="opacity:.3">
                <circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>
              </svg>
              <div>${msg}</div>
            </div>`;
          ['intel-conf-val','intel-impact-sum','intel-bullish-count','intel-updated-val'].forEach(id => {
            const el = document.getElementById(id); if (el) el.textContent = '—';
          });
          return;
        }

        // Update AI Confidence metric: average confidence from real insights
        const avgConf = Math.round(insights.reduce((s, r) => s + (r.confidence || 0), 0) / insights.length);
        if (confEl) confEl.textContent = avgConf + '%';

        // Sentiment balance: bullish vs bearish ratio
        const bulls = insights.filter(r => r.sentiment === 'bullish').length;
        const bears = insights.filter(r => r.sentiment === 'bearish').length;
        if (impactEl) {
          if (bulls > bears * 1.5)      { impactEl.textContent = 'Growth Mode';    impactEl.style.color = 'var(--A)'; }
          else if (bears > bulls * 1.5) { impactEl.textContent = 'Risk Elevated';  impactEl.style.color = 'var(--RED)'; }
          else                          { impactEl.textContent = 'Neutral Sync';   impactEl.style.color = 'var(--T2)'; }
        }
        // Bullish signal count
        if (bullCountEl) {
          bullCountEl.textContent = bulls + ' / ' + insights.length;
          bullCountEl.style.color = bulls >= insights.length * 0.6 ? 'var(--A)' : 'var(--T2)';
        }
        // Last updated: most recent insight timestamp
        const latestTs = insights.reduce((max, r) => {
          const t = r.generated_at ? new Date(r.generated_at).getTime() : 0;
          return t > max ? t : max;
        }, 0);
        if (updatedEl) {
          updatedEl.textContent = latestTs
            ? new Date(latestTs).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })
            : '—';
        }

        stream.innerHTML = '';
        insights.forEach(row => {
          const sentiment  = row.sentiment || 'neutral';
          const confidence = row.confidence ?? null;
          const content    = row.content || '';

          // content format: "Signal: BUY | Bullish (75% confidence). <body>. Key level: X. Catalyst: Y."
          const sigMatch = content.match(/Signal:\s*(BUY|SELL|HOLD)/i);
          const signal   = sigMatch ? sigMatch[1].toUpperCase() : '—';

          // Extract key level and catalyst for richer display
          const keyLevelMatch = content.match(/Key level:\s*([^\.\n]+)/i);
          const catalystMatch = content.match(/Catalyst:\s*([^\.\n]+)/i);
          const keyLevel = keyLevelMatch ? keyLevelMatch[1].trim() : null;
          const catalyst = catalystMatch ? catalystMatch[1].trim() : null;

          // Strip "Name (SYM) — Signal: X | Sentiment (N% confidence). " header prefix
          const summary = content
            .replace(/^.+?—\s*Signal:\s*(?:BUY|SELL|HOLD)\s*\|[^\.]+\.\s*/i, '')
            .replace(/\s*Key level:.*?(?=\s*Catalyst:|\.\.|\s*$)/ig, '')
            .replace(/\s*Catalyst:[^\n]*/ig, '')
            .replace(/\.{2,}$/, '.')
            .trim() || content;

          const ts = row.generated_at
            ? new Date(row.generated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : '—';

          const signalColor = signal === 'BUY' ? 'var(--A)' : signal === 'SELL' ? 'var(--RED)' : 'var(--T2)';
          const borderColor = sentiment === 'bullish' ? 'var(--A)' : sentiment === 'bearish' ? 'var(--RED)' : 'var(--T3)';

          const el = document.createElement('div');
          el.className = `intel-item ${sentiment}`;
          el.style.borderLeftColor = borderColor;
          // Expandable card — make it reachable and operable by keyboard.
          el.setAttribute('role', 'button');
          el.setAttribute('tabindex', '0');
          el.setAttribute('aria-expanded', 'false');
          el.innerHTML = `
            <div class="intel-header">
              <span class="intel-tag">${escapeHtml(row.symbol || '—')}</span>
              <span class="intel-time" style="color:var(--T3)">${ts}</span>
            </div>
            <div class="intel-msg" style="color:var(--T1);line-height:1.5;">${escapeHtml(summary)}</div>
            ${(keyLevel || catalyst) ? `<div style="display:flex;gap:12px;flex-wrap:wrap;font-family:var(--M);font-size:10px;color:var(--T3);margin-top:2px;">
              ${keyLevel ? `<span>⊕ <span style="color:var(--T2)">${escapeHtml(keyLevel)}</span></span>` : ''}
              ${catalyst ? `<span>⚡ <span style="color:var(--T2)">${escapeHtml(catalyst)}</span></span>` : ''}
            </div>` : ''}
            <div class="intel-meta">
              <span class="intel-impact" style="color:${signalColor};text-shadow:none">
                ${signal}${confidence !== null ? ' · ' + confidence + '%' : ''}
              </span>
              <span class="intel-source" style="color:var(--T3)">${escapeHtml(row.symbol || '—')}</span>
            </div>
            <div class="intel-full">${escapeHtml(row.content || summary)}</div>`;
          const toggle = () => {
            el.classList.toggle('expanded');
            el.setAttribute('aria-expanded', el.classList.contains('expanded') ? 'true' : 'false');
          };
          el.addEventListener('click', toggle);
          el.addEventListener('keydown', e => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
          });
          stream.appendChild(el);
        });
        // Stamp last sync time
        const syncEl = document.getElementById('intel-last-sync');
        if (syncEl) {
          const now = new Date();
          syncEl.textContent = `LAST_SYNC: ${now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
        }
      }

      intelRefresh = fetchAndRender;
      await fetchAndRender();
      // Skip polls while the tab is hidden; catch up as soon as it's visible.
      setInterval(() => { if (document.visibilityState === 'visible') fetchAndRender(); }, 300000);
    }

    /* ══════════════════════════════════════════════
       9b. ASSET ALLOCATION — dynamic AUM + diversification
       Driven by real holdings from /db/portfolio (qty × avg_price),
       not transaction-category guesswork.
    ══════════════════════════════════════════════ */
    async function updateAssetAllocationMeta() {
      const aumEl  = document.getElementById('alloc-aum');
      const divEl  = document.getElementById('divers-index');
      const b1 = document.getElementById('alloc-bar-1');
      const b2 = document.getElementById('alloc-bar-2');
      const b3 = document.getElementById('alloc-bar-3');
      const p1 = document.getElementById('alloc-pct-spy');
      const p2 = document.getElementById('alloc-pct-btc');
      const p3 = document.getElementById('alloc-pct-eth');

      const fmtInr = v => v >= 1e7 ? '₹' + (v / 1e7).toFixed(2) + 'Cr'
                        : v >= 1e5 ? '₹' + (v / 1e5).toFixed(2) + 'L'
                        : '₹' + Math.round(v).toLocaleString('en-IN');

      // Preferred: live mark-to-market valuation (/portfolio/value — quotes +
      // USDINR FX). Falls back to cost basis when quotes are unavailable.
      let valued = null, total = 0, mtm = false, pnlPct = null;
      try {
        const res = await fetch(`${API}/portfolio/value`);
        if (res.ok) {
          const data = await res.json();
          if ((data.holdings || []).length) {
            valued = data.holdings.map(h => ({ ...h, value: h.value_inr }));
            total  = data.total_value_inr || 1;
            pnlPct = data.pnl_pct;
            mtm    = true;
          }
        }
      } catch (e) { console.warn('portfolio/value fetch failed, falling back to cost basis:', e); }

      if (!valued) {
        try {
          const res = await fetch(`${API}/db/portfolio`);
          if (!res.ok) throw new Error(res.status);
          const holdings = (await res.json()).holdings || [];
          if (!holdings.length) { if (aumEl) aumEl.textContent = 'Invested: —'; return; }
          valued = holdings.map(h => ({ ...h, value: Number(h.quantity) * Number(h.avg_price) }));
          total  = valued.reduce((s, h) => s + h.value, 0) || 1;
        } catch (e) {
          console.warn('updateAssetAllocationMeta portfolio fetch failed:', e);
          if (aumEl) aumEl.textContent = 'Invested: —';
          return;
        }
      }

      // Mark-to-market gets live value + unrealised P&L; the cost-basis
      // fallback keeps the honest "Invested" label.
      if (aumEl) {
        if (mtm) {
          const sign = pnlPct >= 0 ? '▲' : '▼';
          aumEl.textContent = `Value: ${fmtInr(total)} ${sign} ${Math.abs(pnlPct).toFixed(1)}%`;
          aumEl.style.color = pnlPct >= 0 ? 'var(--A)' : 'var(--RED)';
          aumEl.title = 'Live mark-to-market value (incl. USD→INR for crypto) with unrealised P&L vs cost';
        } else {
          aumEl.textContent = 'Invested: ' + fmtInr(total);
          aumEl.style.color = '';
          aumEl.title = 'Total invested capital at cost (qty × avg buy price) — live quotes unavailable';
        }
      }

      // Diversification Index: HHI across all holdings (lower concentration = higher score)
      const hhi = valued.reduce((s, h) => s + (h.value / total) ** 2, 0);
      const divIdx = Math.round((1 - hhi) * 100);
      const label  = divIdx >= 75 ? 'Optimal' : divIdx >= 50 ? 'Moderate' : 'Concentrated';
      if (divEl) divEl.textContent = `${divIdx}/100 — ${label}`;

      // Featured cards: S&P 500 ETF, Bitcoin, Ethereum — weight = value / total portfolio
      const find = sym => valued.find(h => h.symbol === sym);
      const w1 = (find('SPY')?.value || 0) / total;
      const w2 = (find('BTC')?.value || 0) / total;
      const w3 = (find('ETH')?.value || 0) / total;

      if (b1) b1.style.width = (w1 * 100).toFixed(1) + '%';
      if (b2) b2.style.width = (w2 * 100).toFixed(1) + '%';
      if (b3) b3.style.width = (w3 * 100).toFixed(1) + '%';
      if (p1) p1.textContent = Math.round(w1 * 100) + '%';
      if (p2) p2.textContent = Math.round(w2 * 100) + '%';
      if (p3) p3.textContent = Math.round(w3 * 100) + '%';
    }

    // ── Global HTML safety helper (used by Neural Intelligence) ──
    function escapeHtml(str) {
      return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
    }

    /* ══════════════════════════════════════════════
       §9.4  BACKTESTING ENGINE
    ══════════════════════════════════════════════ */
    let btChartState = null;
    let btChartHoverInit = false;

    // Lakh notation only makes sense for INR; use k/M for everything else.
    function fmtAxisMoney(v, ccy) {
      if (ccy === '₹') return v >= 1e5 ? (v / 1e5).toFixed(1) + 'L' : Math.round(v).toLocaleString('en-IN');
      if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
      if (v >= 1e3) return (v / 1e3).toFixed(1) + 'k';
      return Math.round(v).toLocaleString('en-US');
    }

    function setupBtChartHover() {
      if (btChartHoverInit) return;
      btChartHoverInit = true;
      const wrap    = document.getElementById('bt-chart-wrap');
      const tooltip = document.getElementById('bt-chart-tooltip');
      const dateEl  = document.getElementById('bt-tt-date');
      const valEl   = document.getElementById('bt-tt-val');
      const chgEl   = document.getElementById('bt-tt-chg');

      wrap.addEventListener('mousemove', e => {
        if (!btChartState) return;
        const { curve, px, py, pad, W, initialCapital } = btChartState;
        const rect = wrap.getBoundingClientRect();
        const relX = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
        const svgX = relX * W;
        const span = W - pad.l - pad.r;
        let idx = Math.round(((svgX - pad.l) / span) * (curve.length - 1));
        idx = Math.max(0, Math.min(curve.length - 1, idx));

        const point = curve[idx];
        const x = px(idx), y = py(point.v);

        const line = document.getElementById('bt-hover-line');
        const dot  = document.getElementById('bt-hover-dot');
        if (line && dot) {
          line.setAttribute('x1', x); line.setAttribute('x2', x);
          line.style.opacity = 1;
          dot.setAttribute('cx', x); dot.setAttribute('cy', y);
          dot.style.opacity = 1;
        }

        const ccy = document.getElementById('bt-currency-label')?.textContent || '₹';
        const chg = ((point.v - initialCapital) / initialCapital) * 100;
        dateEl.textContent = new Date(point.t).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
        valEl.textContent  = ccy + Math.round(point.v).toLocaleString(ccy === '₹' ? 'en-IN' : 'en-US');
        chgEl.textContent  = `${chg >= 0 ? '+' : ''}${chg.toFixed(2)}% vs initial`;
        chgEl.style.color  = chg >= 0 ? 'var(--A)' : '#ff4d4d';

        const xPx = (x / W) * rect.width;
        tooltip.style.transform = 'none';
        tooltip.style.top  = '8px';
        tooltip.classList.add('show');
        const ttW = tooltip.offsetWidth;
        tooltip.style.left = `${Math.max(4, Math.min(rect.width - ttW - 4, xPx - ttW / 2))}px`;
      });

      wrap.addEventListener('mouseleave', () => {
        const line = document.getElementById('bt-hover-line');
        const dot  = document.getElementById('bt-hover-dot');
        if (line) line.style.opacity = 0;
        if (dot)  dot.style.opacity = 0;
        tooltip.classList.remove('show');
      });
    }

    function renderBtEquityCurve(curve, initialCapital, ccy = '₹') {
      const svg = document.getElementById('bt-svg');
      // Use the wrapper's real pixel size so axis labels aren't stretched.
      const wrapEl = document.getElementById('bt-chart-wrap');
      const W = Math.max(wrapEl?.clientWidth || 800, 320);
      const H = Math.max(wrapEl?.clientHeight || 240, 160);
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

      const vals = curve.map(p => p.v);
      const minV = Math.min(...vals);
      const maxV = Math.max(...vals);
      const range = maxV - minV || 1;
      const pad   = { t: 10, b: 24, l: 4, r: 60 };
      const pw    = W - pad.l - pad.r;
      const ph    = H - pad.t - pad.b;

      const px = i => pad.l + (i / (vals.length - 1)) * pw;
      const py = v => pad.t + ph - ((v - minV) / range) * ph;

      // Gradient fill
      const col = '#00e5a0';

      let d = '', ad = '';
      vals.forEach((v, i) => {
        const x = px(i), y = py(v);
        if (i === 0) { d = `M ${x} ${y}`; ad = `M ${x} ${H - pad.b} L ${x} ${y}`; }
        else          { d += ` L ${x} ${y}`; ad += ` L ${x} ${y}`; }
      });
      ad += ` L ${pw + pad.l} ${H - pad.b} Z`;

      // Baseline (initial capital)
      const baseLine = py(initialCapital);

      svg.innerHTML = `
        <defs>
          <linearGradient id="btGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${col}" stop-opacity="0.2"/>
            <stop offset="100%" stop-color="${col}" stop-opacity="0"/>
          </linearGradient>
        </defs>
        <line x1="${pad.l}" x2="${pw + pad.l}" y1="${baseLine}" y2="${baseLine}"
          stroke="rgba(255,255,255,.12)" stroke-width="1" stroke-dasharray="4 4"/>
        <path d="${ad}" fill="url(#btGrad)"/>
        <path d="${d}" fill="none" stroke="${col}" stroke-width="2"
          style="filter:drop-shadow(0 0 4px ${col}40)"/>
        <text x="${W - pad.r + 6}" y="${py(maxV) + 4}" font-family="var(--M)" font-size="9" fill="rgba(255,255,255,.45)">
          ${fmtAxisMoney(maxV, ccy)}
        </text>
        <text x="${W - pad.r + 6}" y="${py(minV) + 4}" font-family="var(--M)" font-size="9" fill="rgba(255,255,255,.45)">
          ${fmtAxisMoney(minV, ccy)}
        </text>
        <line id="bt-hover-line" x1="0" x2="0" y1="${pad.t}" y2="${H - pad.b}"
          stroke="${col}" stroke-width="1" stroke-dasharray="3 3" opacity="0"/>
        <circle id="bt-hover-dot" r="4" fill="${col}" stroke="#08080a" stroke-width="2" opacity="0"/>`;

      btChartState = { curve, px, py, pad, W, H, initialCapital, ccy };
      setupBtChartHover();
    }

    function renderBtMetrics(m, finalValue, initialCapital, ccy = '₹') {
      const el = document.getElementById('bt-metrics-row');
      const rtn  = m.total_return;
      const rClass = rtn >= 0 ? 'pos' : 'neg';
      const locale = ccy === '₹' ? 'en-IN' : 'en-US';

      el.innerHTML = `
        <div class="bt-metric">
          <div class="bt-m-lbl">Total Return</div>
          <div class="bt-m-val" style="color:${rtn >= 0 ? '#00e5a0' : '#ff4d4d'}">${rtn >= 0 ? '+' : ''}${rtn.toFixed(2)}%</div>
          <div class="bt-m-sub ${rClass}">Final: ${ccy}${Math.round(finalValue).toLocaleString(locale)}</div>
        </div>
        <div class="bt-metric">
          <div class="bt-m-lbl">Sharpe Ratio</div>
          <div class="bt-m-val">${m.sharpe}</div>
          <div class="bt-m-sub ${m.sharpe >= 1 ? 'pos' : m.sharpe >= 0 ? 'neu' : 'neg'}">
            ${m.sharpe >= 1 ? 'Strong' : m.sharpe >= 0 ? 'Moderate' : 'Weak'}
          </div>
        </div>
        <div class="bt-metric">
          <div class="bt-m-lbl">Max Drawdown</div>
          <div class="bt-m-val">-${m.max_drawdown.toFixed(1)}%</div>
          <div class="bt-m-sub ${m.max_drawdown < 10 ? 'pos' : m.max_drawdown < 25 ? 'neu' : 'neg'}">
            ${m.max_drawdown < 10 ? 'Low' : m.max_drawdown < 25 ? 'Moderate' : 'High'} risk
          </div>
        </div>
        <div class="bt-metric">
          <div class="bt-m-lbl">Win Rate</div>
          <div class="bt-m-val">${m.win_rate}%</div>
          <div class="bt-m-sub ${m.win_rate >= 50 ? 'pos' : 'neg'}">${m.win_rate >= 50 ? 'Favorable' : 'Unfavorable'}</div>
        </div>
        <div class="bt-metric">
          <div class="bt-m-lbl">Trades</div>
          <div class="bt-m-val">${m.total_trades}</div>
          <div class="bt-m-sub neu">Executions</div>
        </div>`;
    }

    async function runBacktest() {
      const btn     = document.getElementById('bt-run-btn');
      const symbol  = document.getElementById('bt-symbol').value;
      const start   = document.getElementById('bt-start').value;
      const end     = document.getElementById('bt-end').value;
      const shortV  = parseInt(document.getElementById('bt-short').value);
      const longV   = parseInt(document.getElementById('bt-long').value);
      const capital = parseFloat(document.getElementById('bt-capital').value);

      const errEl = document.getElementById('bt-form-err');
      const showErr = msg => { if (errEl) { errEl.textContent = msg; errEl.style.display = 'block'; } };
      const clearErr = ()  => { if (errEl) { errEl.style.display = 'none'; errEl.textContent = ''; } };

      if (!start || !end || start >= end) {
        showErr('Please set a valid date range where start < end.');
        return;
      }
      if (!Number.isFinite(shortV) || !Number.isFinite(longV) || shortV < 2 || longV < 5) {
        showErr('SMA periods must be numbers (short ≥ 2, long ≥ 5).');
        return;
      }
      if (shortV >= longV) {
        showErr('Short SMA must be less than Long SMA.');
        return;
      }
      // The long SMA needs at least longV data points before it produces a
      // signal. Trading days ≈ 5/7 of calendar days, so require enough range
      // up front instead of surfacing an opaque server error.
      const rangeDays = (new Date(end) - new Date(start)) / 86400000;
      if (rangeDays * 5 / 7 < longV + 10) {
        showErr(`Date range too short for SMA(${longV}) — pick at least ~${Math.ceil((longV + 10) * 7 / 5)} days.`);
        return;
      }
      if (!Number.isFinite(capital) || capital <= 0) {
        showErr('Initial capital must be a positive number.');
        return;
      }
      clearErr();
      btn.disabled = true;
      btn.querySelector('.bt-run-inner').innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation:spin 1s linear infinite"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
        Running…`;

      const titleEl = document.getElementById('bt-results-title');
      const metaEl  = document.getElementById('bt-results-meta');

      try {
        const res = await fetch(`${API}/backtest`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol, start, end, short_sma: shortV, long_sma: longV, initial_capital: capital }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          // FastAPI validation errors (422) carry detail as a list of {loc, msg}.
          const detail = Array.isArray(err.detail)
            ? err.detail.map(d => `${(d.loc || []).slice(-1)[0] || 'input'}: ${d.msg}`).join('; ')
            : err.detail;
          throw new Error(detail || res.status);
        }
        const data = await res.json();

        // Show results
        document.getElementById('bt-empty').style.display = 'none';
        const pop = document.getElementById('bt-populated');
        pop.style.display = 'flex';

        if (titleEl) titleEl.textContent = 'Strategy Results';
        if (metaEl)  metaEl.textContent  = `${data.symbol} · SMA(${shortV}, ${longV}) · ${data.start} → ${data.end}`;

        const btCcy = document.getElementById('bt-currency-label')?.textContent || '₹';
        renderBtMetrics(data.metrics, data.final_value, data.initial_capital, btCcy);
        renderBtEquityCurve(data.equity_curve, data.initial_capital, btCcy);

      } catch (e) {
        // Translate raw failures (HTTP status codes, fetch TypeErrors) into
        // something a user can act on; e.message is escaped before injection.
        let msg = String(e.message || 'Unknown error');
        if (/failed to fetch/i.test(msg))   msg = 'Backend unreachable — is the server running?';
        else if (/^\d{3}$/.test(msg))       msg = `Server error (HTTP ${msg}) — try a different symbol or date range.`;
        document.getElementById('bt-populated').style.display = 'none';
        const emptyEl = document.getElementById('bt-empty');
        emptyEl.style.display = 'flex';
        emptyEl.innerHTML = `
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="opacity:.2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
          <div style="color:#ff4d4d">${escapeHtml(msg)}</div>
          <div style="font-size:11px;opacity:.5">Ensure the backend is running and the date range has sufficient data.</div>`;
      } finally {
        btn.disabled = false;
        btn.querySelector('.bt-run-inner').innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          Run Backtest`;
      }
    }

    /* ══════════════════════════════════════════════
       BOOT
    ══════════════════════════════════════════════ */
    // Re-render data-driven widgets in place when fresh DB data arrives —
    // avoids a full page reload (see js/flux-data.js).
    function refreshLocalDataWidgets() {
      invalidateStorageCache();
      LEDGER_TRADES = null; // refetch trades when fresh DB data lands
      buildHeatmap();
      buildTxGraph();
      buildUnlockDonut();
      updatePortfolioPnl();
      buildTickerFeed();
      updateAssetAllocationMeta();
      updateAssetAllocation();
    }
    window.addEventListener('flux:data-updated', refreshLocalDataWidgets);

    window.addEventListener('DOMContentLoaded', async () => {
      refreshLocalDataWidgets();
      initNeuralIntelligence();

      // Load initial candle chart (async — non-blocking for rest of UI)
      await reloadChart(currentAsset, currentTimeframe);
      // Keep the price genuinely live: silent refresh every 60s. Paused while
      // the tab is hidden; refreshed immediately when it becomes visible again.
      setInterval(() => {
        if (document.visibilityState === 'visible') reloadChart(currentAsset, currentTimeframe, true);
      }, 60000);
      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') reloadChart(currentAsset, currentTimeframe, true);
      });

      // Redraw the charts at the new pixel size when their panels resize.
      let chartResizeTimer = null;
      new ResizeObserver(() => {
        clearTimeout(chartResizeTimer);
        chartResizeTimer = setTimeout(() => { if (allCandles.length > 1) drawChart(); }, 120);
      }).observe(mainSVG.parentElement);

      const btWrap = document.getElementById('bt-chart-wrap');
      if (btWrap) {
        let btResizeTimer = null;
        new ResizeObserver(() => {
          clearTimeout(btResizeTimer);
          btResizeTimer = setTimeout(() => {
            if (btChartState) renderBtEquityCurve(btChartState.curve, btChartState.initialCapital, btChartState.ccy);
          }, 120);
        }).observe(btWrap);
      }

      // §9.4 Backtest wiring
      const today = localDateKey(new Date());
      const btEndInput = document.getElementById('bt-end');
      if (btEndInput) { btEndInput.value = today; btEndInput.max = today; }
      const btStartInput = document.getElementById('bt-start');
      if (btStartInput) btStartInput.max = today;
      document.getElementById('bt-run-btn')?.addEventListener('click', runBacktest);
      // Every backtest symbol is USD-quoted.
      const btCurrLabel = document.getElementById('bt-currency-label');
      if (btCurrLabel) btCurrLabel.textContent = '$';

      // If DB hydration hasn't produced transactions a few seconds in,
      // resolve the P&L skeletons to an explicit empty state.
      setTimeout(() => updatePortfolioPnl(true), 4000);
    });
