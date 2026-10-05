/* FLUX Dashboard — dashboard.js */

/* ────────────────────────────────────────────
   0. TRANSACTION HELPERS
───────────────────────────────────────────── */
function isSameMonth(d1, d2) {
  return d1.getFullYear() === d2.getFullYear() && d1.getMonth() === d2.getMonth();
}

// Same days of last month as have passed this month (1st..today's date), so a
// month-to-date figure is not compared with a whole month and shows -89%.
function isSamePeriodLastMonth(d, now) {
  const prev = new Date(now.getFullYear(), now.getMonth() - 1, 1);
  return isSameMonth(d, prev) && d.getDate() <= now.getDate();
}

// True when the transactions reach back to `start`, i.e. a comparison period
// is fully covered rather than holding the first few weeks of demo data.
function coversFrom(txs, start) {
  return txs.some(t => new Date(t.date) <= start);
}

function getTransactions() {
  return JSON.parse(localStorage.getItem('flux_transactions') || '[]');
}

function computeStatCards() {
  const txs = getTransactions();
  const now  = new Date();

  const thisMo = txs.filter(t => isSameMonth(new Date(t.date), now));
  const lastMo = txs.filter(t => isSamePeriodLastMonth(new Date(t.date), now));

  const sum = (arr) => arr.reduce((s, t) => s + t, 0);
  const income   = sum(thisMo.filter(t => t.amount > 0).map(t => t.amount));
  const spend    = sum(thisMo.filter(t => t.amount < 0 && t.category !== 'investment').map(t => Math.abs(t.amount)));
  const invested = sum(txs.filter(t => t.category === 'investment').map(t => Math.abs(t.amount)));
  const savings  = Math.max(0, income - spend);

  const lastIncome = sum(lastMo.filter(t => t.amount > 0).map(t => t.amount));
  const lastSpend  = sum(lastMo.filter(t => t.amount < 0 && t.category !== 'investment').map(t => Math.abs(t.amount)));

  return { income, spend, invested, savings, lastIncome, lastSpend };
}

function updateStatCards() {
  const { income, spend, invested, savings, lastIncome, lastSpend } = computeStatCards();

  const pct = (cur, prev) => prev > 0 ? ((cur - prev) / prev * 100).toFixed(1) : null;
  const incomePct = pct(income, lastIncome);
  const spendPct  = pct(spend, lastSpend);

  document.querySelectorAll('.stat-card').forEach(card => {
    const label = card.querySelector('.stat-card-label')?.textContent?.trim();
    const valEl = card.querySelector('.stat-val');
    const chgEl = card.querySelector('.stat-change');
    if (!valEl) return;

    let val = 0, chgText = '', chgClass = 'up';

    if (label === 'Monthly Income') {
      val = income;
      if (incomePct !== null) {
        chgClass = incomePct >= 0 ? 'up' : 'down';
        chgText  = `${incomePct >= 0 ? '↑' : '↓'} ${Math.abs(incomePct)}% vs last month`;
      }
    } else if (label === 'Monthly Spend') {
      val = spend;
      if (spendPct !== null) {
        chgClass = spendPct > 0 ? 'down' : 'up';
        chgText  = `${spendPct > 0 ? '↑' : '↓'} ${Math.abs(spendPct)}% vs last month`;
      }
    } else if (label === 'Invested (12 mo)') {
      val = invested;
      chgText = 'What you invested in the past year';
      chgClass = 'up';
    } else if (label === 'Net Savings') {
      val = savings;
      chgText = 'Income minus expenses';
      chgClass = 'up';
    }

    valEl.dataset.count = val;
    animateCounter(valEl, val, 1200, '₹');
    if (chgEl && chgText) {
      chgEl.className = `stat-change ${chgClass}`;
      chgEl.textContent = chgText;
    }
  });
}

/* ────────────────────────────────────────────
   HERO CARD — 24h / Monthly change + Risk Score
   Reads flux_transactions; graceful fallback when
   no data exists (keeps the hardcoded placeholder).
───────────────────────────────────────────── */
function updateHeroMetrics() {
  const txs = getTransactions();
  if (!txs.length) {
    // No transactions (e.g. a brand-new account) → clear the hardcoded HTML
    // placeholders to honest zero/neutral values instead of leaving demo data.
    document.querySelectorAll('.metric-item').forEach(m => {
      const lbl = m.querySelector('.metric-label')?.textContent?.trim();
      const val = m.querySelector('.metric-val');
      if (val && (lbl === '24h Change' || lbl === 'Monthly')) {
        val.textContent = '0.0%';
        val.className = 'metric-val';
      }
    });
    const heroNetEl = document.getElementById('heroNetSavings');
    if (heroNetEl) heroNetEl.textContent = '—';
    return;
  }

  const now  = new Date();

  /* ── 24 h change: today's net txs as % of net worth ── */
  const NET_WORTH = 11845038;         // baseline from the hero card
  const todayNet  = txs
    .filter(t => new Date(t.date).toDateString() === now.toDateString())
    .reduce((s, t) => s + t.amount, 0);
  const change24h = (todayNet / NET_WORTH * 100).toFixed(2);

  /* ── Monthly change: this month net vs last month net ── */
  const thisMonthNet = txs
    .filter(t => { const d = new Date(t.date); return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth(); })
    .reduce((s, t) => s + t.amount, 0);
  const lastMonthNet = txs
    .filter(t => isSamePeriodLastMonth(new Date(t.date), now))
    .reduce((s, t) => s + t.amount, 0);
  const monthlyPct = lastMonthNet !== 0
    ? ((thisMonthNet - lastMonthNet) / Math.abs(lastMonthNet) * 100).toFixed(1)
    : (thisMonthNet >= 0 ? '+0.0' : '-0.0');

  /* ── Update .metric-val elements ── */
  document.querySelectorAll('.metric-item').forEach(m => {
    const lbl = m.querySelector('.metric-label')?.textContent?.trim();
    const val = m.querySelector('.metric-val');
    if (!val) return;

    if (lbl === '24h Change') {
      const v = parseFloat(change24h);
      val.textContent = (v >= 0 ? '+' : '') + change24h + '%';
      val.className   = 'metric-val ' + (v >= 0 ? 'positive' : 'negative');
    } else if (lbl === 'Monthly') {
      const v = parseFloat(monthlyPct);
      val.textContent = (v >= 0 ? '+' : '') + monthlyPct + '%';
      val.className   = 'metric-val ' + (v >= 0 ? 'positive' : 'negative');
    }
  });

  /* ── Risk / Growth score: savings-rate heuristic ── */
  const { income, spend, savings } = computeStatCards();
  if (income > 0) {
    const score = Math.min(100, Math.max(0, Math.round((income - spend) / income * 100)));
    const scoreEl = document.querySelector('.risk-score');
    const ringEl  = document.querySelector('.risk-ring');
    if (scoreEl) scoreEl.textContent = score;
    // Animate conic-gradient ring: set CSS custom property
    if (ringEl) {
      // Animate from current value for a smooth fill-up effect
      const prev = parseInt(ringEl.style.getPropertyValue('--score') || '0', 10);
      let frame = prev;
      const step = () => {
        frame = Math.min(score, frame + 2);
        ringEl.style.setProperty('--score', frame);
        if (frame < score) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }
  }

  /* ── Hero Net Savings ── */
  const heroNetEl = document.getElementById('heroNetSavings');
  if (heroNetEl) {
    heroNetEl.textContent = savings > 0
      ? '₹' + savings.toLocaleString('en-IN')
      : income > 0 ? '₹0' : '—';
  }
}

function refreshRevenueData() {
  const txs = getTransactions();
  const now  = new Date();

  // ── Yearly: income / expense / net per month, last 12 months ──
  const yLabels = [];
  const yInc = [], yExp = [], yNet = [];
  for (let i = 11; i >= 0; i--) {
    const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
    yLabels.push(monthsShort[d.getMonth()]);
    const bucket = txs.filter(t => {
      const td = new Date(t.date);
      return td.getFullYear() === d.getFullYear() && td.getMonth() === d.getMonth();
    });
    const inc = bucket.filter(t => t.amount > 0).reduce((s, t) => s + t.amount, 0);
    const exp = bucket.filter(t => t.amount < 0 && t.category !== 'investment').reduce((s, t) => s + Math.abs(t.amount), 0);
    yInc.push(inc); yExp.push(exp); yNet.push(inc - exp);
  }
  const yTotal     = yInc.reduce((s, v) => s + v, 0);
  const prevYrStart = new Date(now.getFullYear()-1, now.getMonth()-11, 1);
  const prevYrEnd   = new Date(now.getFullYear()-1, now.getMonth()+1, 0);
  const prevYrTot  = !coversFrom(txs, prevYrStart) ? 0 : txs
    .filter(t => { const td = new Date(t.date); return td >= prevYrStart && td <= prevYrEnd && t.amount > 0; })
    .reduce((s, t) => s + t.amount, 0);

  revenueData.yearly.labels         = yLabels;
  revenueData.yearly.values         = yInc;          // primary bars = income
  revenueData.yearly.highlightIndex = 11;
  revenueData.yearly.total          = '₹' + yTotal.toLocaleString('en-IN');
  revenueData.yearly.comparison     = prevYrTot > 0 ? 'vs ₹' + prevYrTot.toLocaleString('en-IN') + ' last period' : '';
  revenueData.yearly.prevTotal      = prevYrTot;
  revenueMetrics.yearly.income  = yInc;
  revenueMetrics.yearly.expense = yExp;
  revenueMetrics.yearly.net     = yNet;

  // ── Monthly: income / expense / net per week of current month ──
  const wkInc = [0,0,0,0,0], wkExp = [0,0,0,0,0], wkNet = [0,0,0,0,0];
  const wkIdx = t => Math.min(4, Math.floor((new Date(t.date).getDate() - 1) / 7));
  txs.filter(t => isSameMonth(new Date(t.date), now)).forEach(t => {
    const i = wkIdx(t);
    if (t.amount > 0) wkInc[i] += t.amount;
    else if (t.category !== 'investment') wkExp[i] += Math.abs(t.amount);
  });
  for (let i = 0; i < 5; i++) wkNet[i] = wkInc[i] - wkExp[i];
  const mTotal    = wkInc.reduce((s, v) => s + v, 0);
  const prevMoTot = txs.filter(t => isSamePeriodLastMonth(new Date(t.date), now) && t.amount > 0).reduce((s,t) => s+t.amount, 0);

  revenueData.monthly.values         = wkInc;
  revenueData.monthly.highlightIndex = Math.min(4, Math.floor((now.getDate() - 1) / 7));
  revenueData.monthly.total          = '₹' + mTotal.toLocaleString('en-IN');
  revenueData.monthly.comparison     = prevMoTot > 0 ? 'vs ₹' + prevMoTot.toLocaleString('en-IN') + ' same days last month' : '';
  revenueData.monthly.prevTotal      = prevMoTot;
  revenueMetrics.monthly.income  = wkInc;
  revenueMetrics.monthly.expense = wkExp;
  revenueMetrics.monthly.net     = wkNet;

  // ── Weekly: income / expense / net per day, last 7 days ──
  const dLabels = [], dInc = [], dExp = [], dNet = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - i);
    dLabels.push(daysShort[d.getDay()]);
    const bucket = txs.filter(t => new Date(t.date).toDateString() === d.toDateString());
    const inc = bucket.filter(t => t.amount > 0).reduce((s,t) => s+t.amount, 0);
    const exp = bucket.filter(t => t.amount < 0 && t.category !== 'investment').reduce((s,t) => s+Math.abs(t.amount), 0);
    dInc.push(inc); dExp.push(exp); dNet.push(inc - exp);
  }
  const wTotal = dInc.reduce((s, v) => s + v, 0);

  revenueData.weekly.labels         = dLabels;
  revenueData.weekly.values         = dInc;
  revenueData.weekly.highlightIndex = 6;
  revenueData.weekly.total          = '₹' + wTotal.toLocaleString('en-IN');
  revenueData.weekly.comparison     = '';
  revenueMetrics.weekly.income  = dInc;
  revenueMetrics.weekly.expense = dExp;
  revenueMetrics.weekly.net     = dNet;
}

function buildDaysChart(numDays) {
  const txs = getTransactions();
  const now = new Date();
  const labels = [], inc = [], exp = [], net = [];
  for (let i = numDays - 1; i >= 0; i--) {
    const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - i);
    const label = numDays <= 14
      ? d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })
      : (i % 5 === 0 ? d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }) : '');
    labels.push(label);
    const bucket  = txs.filter(t => new Date(t.date).toDateString() === d.toDateString());
    const income  = bucket.filter(t => t.amount > 0).reduce((s,t) => s+t.amount, 0);
    const expense = bucket.filter(t => t.amount < 0 && t.category !== 'investment').reduce((s,t) => s+Math.abs(t.amount), 0);
    inc.push(income); exp.push(expense); net.push(income - expense);
  }
  const total = inc.reduce((s,v) => s+v, 0);
  revenueData.custom = { labels, values: inc, highlightIndex: numDays - 1, total: '₹' + total.toLocaleString('en-IN'), comparison: '', prevTotal: 0 };
  revenueMetrics.custom = { income: inc, expense: exp, net };
  buildRevenueChart('custom');
}

function buildYTDChart() {
  const txs = getTransactions();
  const now = new Date();
  const labels = [], inc = [], exp = [], net = [];
  for (let m = 0; m <= now.getMonth(); m++) {
    labels.push(monthsShort[m]);
    const bucket  = txs.filter(t => { const d = new Date(t.date); return d.getFullYear() === now.getFullYear() && d.getMonth() === m; });
    const income  = bucket.filter(t => t.amount > 0).reduce((s,t) => s+t.amount, 0);
    const expense = bucket.filter(t => t.amount < 0 && t.category !== 'investment').reduce((s,t) => s+Math.abs(t.amount), 0);
    inc.push(income); exp.push(expense); net.push(income - expense);
  }
  const total = inc.reduce((s,v) => s+v, 0);
  revenueData.custom = { labels, values: inc, highlightIndex: labels.length - 1, total: '₹' + total.toLocaleString('en-IN'), comparison: '', prevTotal: 0 };
  revenueMetrics.custom = { income: inc, expense: exp, net };
  buildRevenueChart('custom');
}

function buildDateRangeChart(fromDate, toDate) {
  const txs = getTransactions();
  const days = Math.round((toDate - fromDate) / 86400000) + 1;
  const labels = [], inc = [], exp = [], net = [];
  for (let i = 0; i < days; i++) {
    const d = new Date(fromDate.getFullYear(), fromDate.getMonth(), fromDate.getDate() + i);
    const label = (days <= 14 || i % Math.ceil(days / 10) === 0)
      ? d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }) : '';
    labels.push(label);
    const bucket  = txs.filter(t => new Date(t.date).toDateString() === d.toDateString());
    const income  = bucket.filter(t => t.amount > 0).reduce((s,t) => s+t.amount, 0);
    const expense = bucket.filter(t => t.amount < 0 && t.category !== 'investment').reduce((s,t) => s+Math.abs(t.amount), 0);
    inc.push(income); exp.push(expense); net.push(income - expense);
  }
  const total = inc.reduce((s,v) => s+v, 0);
  revenueData.custom = { labels, values: inc, highlightIndex: days - 1, total: '₹' + total.toLocaleString('en-IN'), comparison: '', prevTotal: 0 };
  revenueMetrics.custom = { income: inc, expense: exp, net };
  buildRevenueChart('custom');
}

function renderTransactionList() {
  const list = document.getElementById('liveTransactionList');
  if (!list) return;

  const allTxs = getTransactions();
  const txs = allTxs.slice(0, 6);
  if (txs.length === 0) { checkEmptyStates(); return; }

  list.innerHTML = txs.map((t, i) => {
    const isIn  = t.amount > 0;
    const amt   = Math.abs(t.amount);
    const dStr  = formatTxDate(t.date);
    const color = categoryColor(t.category);
    return `
      <div class="tx-item${t.flagged ? ' flagged' : ''}" data-tx-index="${i}">
        <div class="tx-left">
          <div class="tx-icon">${categoryIcon(t.category)}</div>
          <div class="tx-details">
            <div class="tx-title">${t.title}</div>
            <div class="tx-date">${dStr}</div>
          </div>
        </div>
        <div class="tx-right">
          <div class="tx-amount ${isIn ? 'in' : 'out'}">${isIn ? '+' : '−'}₹${amt.toLocaleString('en-IN')}</div>
          <div class="tx-ai-label" style="background:${color.bg};color:${color.fg};">${capitalize(t.category)}</div>
        </div>
        <div class="tx-actions">
          <button type="button" class="tx-action-btn flag${t.flagged ? ' active' : ''}" data-action="flag" title="${t.flagged ? 'Remove flag' : 'Flag for audit'}" aria-label="${t.flagged ? 'Remove flag' : 'Flag for audit'}">⚑</button>
          <button type="button" class="tx-action-btn" data-action="view" title="View details" aria-label="View transaction details">View</button>
        </div>
      </div>`;
  }).join('');

  // Event delegation — one listener on the list
  list.addEventListener('click', handleTxAction, { once: true });
}

function handleTxAction(e) {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const item = btn.closest('[data-tx-index]');
  if (!item) return;
  const idx = parseInt(item.dataset.txIndex);
  const txs = getTransactions();
  const tx  = txs[idx];
  if (!tx) return;

  if (btn.dataset.action === 'flag') {
    tx.flagged = !tx.flagged;
    localStorage.setItem('flux_transactions', JSON.stringify(txs));
    renderTransactionList();
    showToast(tx.flagged ? '⚑ Transaction flagged for audit' : 'Flag removed', tx.flagged ? 'error' : 'info');
  } else if (btn.dataset.action === 'view') {
    const dStr = new Date(tx.date).toLocaleDateString('en-IN', { weekday:'long', day:'2-digit', month:'long', year:'numeric' });
    const isIn = tx.amount > 0;
    const amt  = Math.abs(tx.amount);
    showToast(
      `<b>${tx.title}</b><br/>${dStr}<br/>${capitalize(tx.category)} · ${isIn ? '+' : '−'}₹${amt.toLocaleString('en-IN')}${tx.account ? '<br/>Account: ' + tx.account : ''}${tx.flagged ? '<br/><span style="color:#ff4d4d">⚑ Flagged for audit</span>' : ''}`,
      'info'
    );
  }
}

const TX_PAGE_SIZE = 10;
let _txPage = 0;

function getFilteredTxs() {
  const search   = (document.getElementById('txSearch')?.value || '').toLowerCase();
  const category = document.getElementById('txCategoryFilter')?.value || '';
  const sort     = document.getElementById('txSort')?.value || 'date-desc';
  let txs = getTransactions();
  if (search)   txs = txs.filter(t => (t.title || '').toLowerCase().includes(search));
  if (category) txs = txs.filter(t => t.category === category);
  txs.sort((a, b) => {
    if (sort === 'date-desc')   return new Date(b.date) - new Date(a.date);
    if (sort === 'date-asc')    return new Date(a.date) - new Date(b.date);
    if (sort === 'amount-desc') return Math.abs(b.amount) - Math.abs(a.amount);
    if (sort === 'amount-asc')  return Math.abs(a.amount) - Math.abs(b.amount);
    return 0;
  });
  return txs;
}

function renderTransactionModal() {
  const fullTxList = document.getElementById('fullTransactionList');
  const pagination = document.getElementById('txPagination');
  if (!fullTxList) return;

  const txs = getFilteredTxs();
  if (txs.length === 0) {
    const hasFilter = document.getElementById('txSearch')?.value || document.getElementById('txCategoryFilter')?.value;
    fullTxList.innerHTML = `
      <div class="tx-empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2" aria-hidden="true">
          <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        <strong>${hasFilter ? 'No results found' : 'No transactions yet'}</strong>
        <span>${hasFilter ? 'Try adjusting your search or filter' : 'Add transactions via the account setup to see them here'}</span>
      </div>`;
    if (pagination) pagination.innerHTML = '';
    return;
  }

  const totalPages = Math.ceil(txs.length / TX_PAGE_SIZE);
  _txPage = Math.min(_txPage, totalPages - 1);
  const pageTxs = txs.slice(_txPage * TX_PAGE_SIZE, (_txPage + 1) * TX_PAGE_SIZE);

  fullTxList.innerHTML = pageTxs.map((t, relIdx) => {
    const absIdx = _txPage * TX_PAGE_SIZE + relIdx;
    const isIn  = t.amount > 0;
    const amt   = Math.abs(t.amount);
    const dStr  = new Date(t.date).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
    const color = categoryColor(t.category);
    return `
      <div class="tx-item${t.flagged ? ' flagged' : ''}" data-tx-index="${absIdx}">
        <div class="tx-left">
          <div class="tx-icon">${categoryIcon(t.category)}</div>
          <div class="tx-details">
            <div class="tx-title">${t.title}</div>
            <div class="tx-date">${dStr}${t.account ? ' · ' + t.account : ''}</div>
          </div>
        </div>
        <div class="tx-right">
          <div class="tx-amount ${isIn ? 'in' : 'out'}">${isIn ? '+' : '−'}₹${amt.toLocaleString('en-IN')}</div>
          <div class="tx-ai-label" style="background:${color.bg};color:${color.fg};">${capitalize(t.category)}</div>
        </div>
        <div class="tx-actions">
          <button type="button" class="tx-action-btn flag${t.flagged ? ' active' : ''}" data-action="flag" title="${t.flagged ? 'Remove flag' : 'Flag for audit'}" aria-label="${t.flagged ? 'Remove flag' : 'Flag for audit'}">⚑</button>
          <button type="button" class="tx-action-btn" data-action="view" title="View details" aria-label="View transaction details">View</button>
        </div>
      </div>`;
  }).join('');

  if (pagination) {
    pagination.innerHTML = `
      <button type="button" class="tx-page-btn" id="txPrevPage" ${_txPage === 0 ? 'disabled' : ''}>‹ Prev</button>
      <span>${_txPage + 1} / ${totalPages}</span>
      <button type="button" class="tx-page-btn" id="txNextPage" ${_txPage >= totalPages - 1 ? 'disabled' : ''}>Next ›</button>`;
    document.getElementById('txPrevPage')?.addEventListener('click', () => { _txPage--; renderTransactionModal(); });
    document.getElementById('txNextPage')?.addEventListener('click', () => { _txPage++; renderTransactionModal(); });
  }
}

function updateCalendarDots(date) {
  const txs = getTransactions();
  const txDays = new Set(
    txs
      .filter(t => isSameMonth(new Date(t.date), date))
      .map(t => new Date(t.date).getDate())
  );
  return txDays;
}

function formatTxDate(iso) {
  const d   = new Date(iso);
  const now = new Date();
  const diff = Math.floor((now - d) / 86400000);
  if (diff === 0) return 'Today';
  if (diff === 1) return 'Yesterday';
  if (diff < 7)  return d.toLocaleDateString('en-IN', { weekday: 'long' });
  return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
}

function categoryColor(cat) {
  const map = {
    income:       { bg: 'rgba(46,204,113,0.15)',  fg: 'var(--accent-teal)' },
    investment:   { bg: 'rgba(59,130,246,0.15)',  fg: 'var(--accent-blue)' },
    essentials:   { bg: 'rgba(243,156,18,0.15)',  fg: '#f39c12' },
    subscription: { bg: 'rgba(224,86,253,0.15)',  fg: 'var(--accent-pink)' },
    lifestyle:    { bg: 'rgba(255,120,50,0.15)',  fg: '#ff7832' },
    business:     { bg: 'rgba(100,200,255,0.15)', fg: '#64c8ff' },
    transfer:     { bg: 'rgba(255,255,255,0.08)', fg: 'var(--text-secondary)' }
  };
  return map[cat] || map.transfer;
}

function categoryIcon(cat) {
  const icons = {
    income:       '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>',
    investment:   '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/></svg>',
    essentials:   '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>',
    subscription: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>',
    lifestyle:    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>',
    business:     '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/></svg>',
    transfer:     '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14"/><path d="M12 5l7 7-7 7"/></svg>'
  };
  return icons[cat] || icons.transfer;
}

function capitalize(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
}

/* ────────────────────────────────────────────
   1. REVENUE DATA (populated from localStorage)
───────────────────────────────────────────── */
const monthsShort = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const daysShort = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const currentDate = new Date();

const revenueData = {
  yearly: {
    labels: Array.from({ length: 12 }, (_, i) => {
      const d = new Date(currentDate.getFullYear(), currentDate.getMonth() - 11 + i, 1);
      return monthsShort[d.getMonth()];
    }),
    values: new Array(12).fill(0),
    highlightIndex: 11,
    total: '₹0',
    comparison: ''
  },
  monthly: {
    labels: ['Wk 1', 'Wk 2', 'Wk 3', 'Wk 4', 'Wk 5'],
    values: [0, 0, 0, 0, 0],
    highlightIndex: Math.floor((currentDate.getDate() - 1) / 7),
    total: '₹0',
    comparison: ''
  },
  weekly: {
    labels: Array.from({ length: 7 }, (_, i) => {
      const d = new Date(currentDate.getFullYear(), currentDate.getMonth(), currentDate.getDate() - 6 + i);
      return daysShort[d.getDay()];
    }),
    values: new Array(7).fill(0),
    highlightIndex: 6,
    total: '₹0',
    comparison: ''
  }
};

// §2.3 — metric labels only; actual values come from revenueMetrics buckets
const metricData = {
  revenue: { label: "Revenue (Income)" },
  profit:  { label: "Expenses" },
  income:  { label: "Net Cashflow" }
};

// Parallel bucket arrays populated by refreshRevenueData()
// income[i]  = gross income for bucket i
// expense[i] = total outflows (excl. investments) for bucket i
// net[i]     = income[i] − all outflows[i]
const revenueMetrics = {
  yearly:  { income: new Array(12).fill(0), expense: new Array(12).fill(0), net: new Array(12).fill(0) },
  monthly: { income: new Array(5).fill(0),  expense: new Array(5).fill(0),  net: new Array(5).fill(0)  },
  weekly:  { income: new Array(7).fill(0),  expense: new Array(7).fill(0),  net: new Array(7).fill(0)  },
};

let currentMetric = 'revenue';
let currentCurrency = 'INR';
let calendarDate = new Date(); // Phase 1: Calendar State

/* ────────────────────────────────────────────
   2. REVENUE BAR CHART
───────────────────────────────────────────── */
let revenueChart;
// §2.3 — live USD rate (open.er-api.com, no key needed)
// Falls back to 0.012 if fetch fails or is offline
let _usdRate = 0.012;
let _usdRateLive = false;
(async () => {
  try {
    const r = await fetch('https://open.er-api.com/v6/latest/INR');
    const d = await r.json();
    if (d.rates && d.rates.USD) { _usdRate = d.rates.USD; _usdRateLive = true; }
  } catch (_) { /* keep fallback — indicator shown in setCurrency */ }
})();

function createBarPattern(ctx) {
  const patternCanvas = document.createElement('canvas');
  const pctx = patternCanvas.getContext('2d');
  const size = 6;
  patternCanvas.width = size;
  patternCanvas.height = size;

  pctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  pctx.lineWidth = 1;
  pctx.beginPath();
  pctx.moveTo(0, size);
  pctx.lineTo(size, 0);
  pctx.stroke();

  return ctx.createPattern(patternCanvas, 'repeat');
}

function buildRevenueChart(key = 'yearly') {
  const chartElement = document.getElementById('revenueBarChart');
  if (!chartElement) return;

  const d = JSON.parse(JSON.stringify(revenueData[key]));           // clone base (labels, highlightIndex, comparison)
  const metric  = metricData[currentMetric];
  const buckets = revenueMetrics[key];                               // real income / expense / net arrays
  const rate    = currentCurrency === 'INR' ? 1 : _usdRate;
  const prefix  = currentCurrency === 'INR' ? '₹' : '$';

  // Pick the right value array based on selected metric
  let values;
  if      (currentMetric === 'profit')  values = buckets.expense;   // Gross Profit = operating expenses (cost view)
  else if (currentMetric === 'income')  values = buckets.net;        // Net Cashflow = income − expenses
  else                                  values = buckets.income;     // Revenue = gross income

  d.values = values;

  // Total for the header label
  const rawTotal    = values.reduce((s, v) => s + Math.max(0, v), 0);
  const rawPrevTotal = d.prevTotal || 0;

  // Update header text
  const revenueValue      = document.querySelector('.revenue-value');
  const revenueComparison = document.querySelector('.revenue-comparison');
  const revenueCurrency   = document.querySelector('.revenue-currency');
  const revenueTitle      = document.querySelector('.revenue-title');
  const revenueBadge      = document.getElementById('revenueBadge');

  if (revenueTitle)      revenueTitle.textContent      = metric.label;
  if (revenueValue)      revenueValue.textContent      = Math.round(rawTotal * rate).toLocaleString('en-IN');
  if (revenueComparison) revenueComparison.textContent  = d.comparison;
  if (revenueCurrency)   revenueCurrency.textContent   = prefix;

  // Keep the canvas text alternative in sync
  if (chartElement) {
    const summary = `${metric.label} chart. Total: ${prefix}${Math.round(rawTotal * rate).toLocaleString('en-IN')}. ${d.labels.length} data points from ${d.labels[0]} to ${d.labels[d.labels.length - 1]}.`;
    chartElement.setAttribute('aria-label', summary);
  }

  if (revenueBadge) {
    if (rawPrevTotal > 0 && rawTotal > 0) {
      const badgePct = ((rawTotal - rawPrevTotal) / rawPrevTotal * 100).toFixed(1);
      const isUp = parseFloat(badgePct) >= 0;
      const arrow = isUp ? '18 15 12 9 6 15' : '6 9 12 15 18 9';
      revenueBadge.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="${arrow}"/></svg>${Math.abs(badgePct)}%`;
      revenueBadge.style.display = '';
      revenueBadge.className = isUp ? 'revenue-badge' : 'revenue-badge down';
    } else {
      revenueBadge.style.display = 'none';
    }
  }

  if (revenueChart) revenueChart.destroy();

  const ctx     = chartElement.getContext('2d');
  const pattern = createBarPattern(ctx);

  // Highlight colour: teal for income, amber for expense, blue for net
  const hlColor = currentMetric === 'profit'  ? '#f39c12'
                : currentMetric === 'income'   ? '#3b82f6'
                : 'var(--accent-teal)';

  revenueChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: d.labels,
      datasets: [{
        data: d.values.map(v => Math.max(0, v)),     // negative net → 0 (handled by colour)
        backgroundColor: d.values.map((_, i) => i === d.highlightIndex ? hlColor : 'rgba(255,255,255,0.08)'),
        borderRadius: 12,
        borderSkipped: false,
        barPercentage: 0.6,
        categoryPercentage: 0.8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: true,
          backgroundColor: '#111',
          titleColor: '#fff',
          bodyColor: hlColor,
          padding: 12,
          displayColors: false,
          callbacks: {
            label: (ctx) => {
              const val = Math.round(ctx.raw * rate);
              return `  ${prefix}${val.toLocaleString('en-IN')}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#666', font: { size: 11, family: 'Inter' } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.03)', drawTicks: false },
          border: { display: false },
          ticks: {
            color: '#666',
            font: { size: 10 },
            callback: (v) => v >= 1e5 ? (v/1e5).toFixed(1)+'L' : v >= 1000 ? (v/1000)+'k' : v
          }
        }
      },
      onHover: (event, elements) => {
        event.native.target.style.cursor = elements.length ? 'pointer' : 'default';
      }
    },
    plugins: [{
      id: 'barPattern',
      beforeDatasetsDraw(chart) {
        const { ctx: c } = chart;
        const meta = chart.getDatasetMeta(0);
        c.save();
        meta.data.forEach((bar, i) => {
          if (i !== d.highlightIndex) {
            c.fillStyle = pattern;
            c.fillRect(bar.x - bar.width / 2, bar.y, bar.width, bar.base - bar.y);
          } else {
            c.fillStyle = '#fff';
            c.textAlign = 'center';
            c.font = 'bold 11px Inter';
            const val = Math.round(Math.max(0, d.values[i]) * rate);
            c.fillText(`${prefix}${val.toLocaleString('en-IN')}`, bar.x, bar.y - 10);
          }
        });
        c.restore();
      }
    }]
  });
}

function syncMetricToChart(label, value) {
  const period = document.getElementById('insightPeriod');
  const item = document.getElementById('chartInsightItem');
  const text = document.getElementById('chartInsightText');
  const nwValue = document.getElementById('netWorthValue');

  if (!period || !item || !text) return;

  period.textContent = label.toUpperCase();
  item.style.display = 'block';

  const rate   = currentCurrency === 'INR' ? 1 : _usdRate;
  const prefix = currentCurrency === 'INR' ? '₹' : '$';
  const valFmt = prefix + Math.round(Math.max(0, value) * rate).toLocaleString('en-IN');
  text.textContent = `${metricData[currentMetric].label} for ${label}: ${valFmt}`;
  
  // Visual pulse on Net Worth
  if (nwValue) {
    nwValue.style.color = 'var(--accent-teal)';
    nwValue.style.textShadow = '0 0 15px var(--accent-teal)';
    setTimeout(() => {
      nwValue.style.color = '';
      nwValue.style.textShadow = '';
    }, 800);
  }
}

function resetMetricSync() {
  const period = document.getElementById('insightPeriod');
  const item = document.getElementById('chartInsightItem');
  if (period) period.textContent = 'OVERALL';
  if (item) item.style.display = 'none';
}

/* ────────────────────────────────────────────
   3. INITIALIZATION & CORE LOGIC
───────────────────────────────────────────── */
// ── Priority 1: Core Systems ──

function updateGreeting() {
  const header = document.getElementById('greetingHeader');
  if (!header) return;

  const hr = new Date().getHours();
  let greet = "Overview";

  if (hr < 12) greet = "Good Morning, Nishanth";
  else if (hr < 17) greet = "Good Afternoon, Nishanth";
  else greet = "Good Evening, Nishanth";

  header.textContent = greet;
}

// ── Priority 2: Phase 2 Intelligence & Interactivity ──

function initTooltipEngine() {
  const tooltip = document.createElement('div');
  tooltip.className = 'flux-tooltip';
  document.body.appendChild(tooltip);

  const triggers = [
    {
      sel: '.risk-ring',
      fn: () => {
        const { income, spend } = computeStatCards();
        const savingsRate = income > 0 ? Math.round((income - spend) / income * 100) : 0;
        const p = JSON.parse(localStorage.getItem('flux_portfolio') || '{}');
        const eq = p.equity ?? 24, cr = p.crypto ?? 14, ca = p.cash ?? 62;
        const diversification = Math.round(100 - Math.max(eq, cr, ca));
        return `<b>SAVINGS SCORE</b><br/>Savings rate: ${savingsRate}%<br/>Equity ${eq}% · Crypto ${cr}% · Cash ${ca}%<br/>Diversification spread: ${diversification}%`;
      }
    },
    {
      sel: '.metric-item:nth-child(1)',
      fn: () => {
        const txs = getTransactions();
        const todayStr = new Date().toDateString();
        const todayTxs = txs.filter(t => new Date(t.date).toDateString() === todayStr);
        const todayNet = todayTxs.reduce((s, t) => s + t.amount, 0);
        const todayIn  = todayTxs.filter(t => t.amount > 0).reduce((s, t) => s + t.amount, 0);
        const sign = todayNet >= 0 ? '+' : '−';
        return `<b>24h Activity</b><br/>Net today: ${sign}₹${Math.abs(Math.round(todayNet)).toLocaleString('en-IN')}<br/>Income today: ₹${Math.round(todayIn).toLocaleString('en-IN')}`;
      }
    },
    {
      sel: '.security-status',
      fn: () => `<b>Vault Protection</b><br/>Card details stored locally in your browser.<br/>Use the Vault button to reveal or mask card numbers.`
    }
  ];

  triggers.forEach(t => {
    const el = document.querySelector(t.sel);
    if (!el) return;

    el.addEventListener('mouseenter', (e) => {
      tooltip.innerHTML = t.fn();
      tooltip.classList.add('show');
    });

    el.addEventListener('mousemove', (e) => {
      tooltip.style.left = (e.pageX + 15) + 'px';
      tooltip.style.top = (e.pageY + 15) + 'px';
    });

    el.addEventListener('mouseleave', () => {
      tooltip.classList.remove('show');
    });
  });
}

function initAssetInteractions() {
  const legendItems = document.querySelectorAll('.asset-legend span');
  const bars = document.querySelectorAll('.asset-bar');
  
  legendItems.forEach((item, idx) => {
    item.addEventListener('mouseenter', () => {
      if (bars[idx]) bars[idx].classList.add('highlight-active');
    });
    item.addEventListener('mouseleave', () => {
      if (bars[idx]) bars[idx].classList.remove('highlight-active');
    });
  });
}

function initOptimizationSummary() {
  const aiLabels = document.querySelectorAll('.portfolio-card .tx-ai-label');
  aiLabels.forEach(label => {
    label.classList.add('clickable');
    label.addEventListener('click', (e) => {
      e.stopPropagation();
      const p = JSON.parse(localStorage.getItem('flux_portfolio') || '{}');
      const eq = p.equity ?? 24, cr = p.crypto ?? 14, ca = p.cash ?? 62;
      const { income, spend } = computeStatCards();
      const savingsRate = income > 0 ? Math.round((income - spend) / income * 100) : 0;
      showToast(`<b>Portfolio Snapshot</b><br/>Equity ${eq}% · Crypto ${cr}% · Cash ${ca}%<br/>Savings rate this month: ${savingsRate}%`, "info");
    });
  });
}


function setMetric(metric) {
  currentMetric = metric;
  buildRevenueChart(document.querySelector('[data-revenue-filter].active')?.dataset.revenueFilter || 'yearly');
  showToast(`${metricData[metric].label} View Active`, "success");
}


let _calTxDays = new Set();

function renderCalendar(date) {
  _calTxDays = updateCalendarDots(date);
  const grid = document.getElementById('calendarGrid');
  const monthTitle = document.getElementById('calMonth');
  if (!grid || !monthTitle) return;

  grid.innerHTML = '';
  
  // Set Title
  const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  monthTitle.textContent = `${monthNames[date.getMonth()]}, ${date.getFullYear()}`;

  // Days Labels
  ["M", "T", "W", "T", "F", "S", "S"].forEach(day => {
    const el = document.createElement('div');
    el.className = 'cal-day-label';
    el.textContent = day;
    grid.appendChild(el);
  });

  const year = date.getFullYear();
  const month = date.getMonth();
  const firstDay = new Date(year, month, 1).getDay(); // 0 is Sunday
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const today = new Date();

  // Shift firstDay to handle Monday as start of week (1-7, where 0=Sunday)
  let startOffset = firstDay === 0 ? 6 : firstDay - 1;

  // Empty Slots
  for (let i = 0; i < startOffset; i++) {
    const el = document.createElement('div');
    el.className = 'cal-day empty';
    grid.appendChild(el);
  }

  // Days
  for (let d = 1; d <= daysInMonth; d++) {
    const el = document.createElement('div');
    el.className = 'cal-day';
    el.textContent = d;

    // Check if it's "Today"
    if (d === today.getDate() && month === today.getMonth() && year === today.getFullYear()) {
      el.classList.add('active');
      const dot = document.createElement('span');
      dot.className = 'cal-dot';
      el.appendChild(dot);
    }

    // Activity dots from real transaction dates
    if (d !== today.getDate() && _calTxDays.has(d)) {
      const dot = document.createElement('span');
      dot.className = 'cal-dot';
      dot.style.opacity = '0.3';
      el.appendChild(dot);
    }

    // Click to view that day's transactions
    if (_calTxDays.has(d)) {
      el.classList.add('has-activity');
      el.setAttribute('tabindex', '0');
      el.setAttribute('role', 'button');
      el.setAttribute('aria-label', `${d} ${monthsShort[date.getMonth()]} — view transactions`);
      el.addEventListener('click', () => showCalDayDetail(date, d));
      keyActivatable(el);
    }

    grid.appendChild(el);
  }

  // Fill remaining slots for 6 rows
  const totalSlots = grid.children.length - 7; // subtract labels
  const remaining = 42 - totalSlots;
  for (let i = 0; i < remaining; i++) {
    const el = document.createElement('div');
    el.className = 'cal-day empty stripe';
    grid.appendChild(el);
  }
}

function showCalDayDetail(calDate, day) {
  const detail    = document.getElementById('calDayDetail');
  const titleEl   = document.getElementById('calDayDetailTitle');
  const listEl    = document.getElementById('calDayDetailList');
  if (!detail || !titleEl || !listEl) return;

  const d = new Date(calDate.getFullYear(), calDate.getMonth(), day);
  const dayStr = d.toLocaleDateString('en-IN', { weekday: 'long', day: '2-digit', month: 'long' });
  const txs = getTransactions().filter(t => new Date(t.date).toDateString() === d.toDateString());

  titleEl.textContent = dayStr.toUpperCase();
  listEl.innerHTML = txs.length === 0
    ? '<div style="font-size:12px;color:var(--text-secondary);">No transactions on this day.</div>'
    : txs.map(t => {
        const isIn = t.amount > 0;
        const amt  = Math.abs(t.amount);
        return `<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:12px;">
          <span style="color:rgba(255,255,255,0.8)">${t.title}</span>
          <span style="color:${isIn ? '#00e5a0' : '#ff4d4d'};font-weight:600">${isIn ? '+' : '−'}₹${amt.toLocaleString('en-IN')}</span>
        </div>`;
      }).join('');

  detail.style.display = 'block';
}

function setCurrency(currency) {
  currentCurrency = currency;
  const rates = { INR: 1, USD: _usdRate };  // live rate (fetched on load)
  const prefix = currency === 'INR' ? '₹' : '$';

  // 1. Update counter elements
  document.querySelectorAll('[data-count]').forEach(el => {
    const originalValue = parseInt(el.dataset.count);
    const converted = Math.round(originalValue * rates[currency]);
    animateCounter(el, converted, 800, prefix);
  });

  // 2. Update Net Worth
  const nw = document.getElementById('netWorthValue');
  if (nw) {
     const savedAccounts = JSON.parse(localStorage.getItem('flux_accounts')) || [];
     const totalBase = calculateTotalBalance(savedAccounts);
     animateCounter(nw, totalBase * rates[currency], 1000, prefix);
  }

  // 3. Update Revenue Chart Metrics
  buildRevenueChart(document.querySelector('[data-revenue-filter].active')?.dataset.revenueFilter || 'yearly');

  if (currency === 'USD' && !_usdRateLive) {
    showToast(`Switched to USD — using estimated rate (₹1 ≈ $${_usdRate.toFixed(4)}). Live rate unavailable.`, "info");
  } else {
    showToast(`Currency Switched to ${currency}`, "info");
  }
}

document.addEventListener('DOMContentLoaded', () => {
  // --- A. Highlight Active Navigation ---
  const currentPath = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-item').forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPath) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  // --- B. Load Persisted State ---
  // No hardcoded fallback — an authenticated user with no accounts must see an
  // empty state, never demo accounts. Backend hydration fills this in place.
  const savedAccounts = JSON.parse(localStorage.getItem('flux_accounts') || '[]');

  // Apply card color based on active account
  const card = document.getElementById('mainCreditCard');
  if (card) {
    card.className = 'flux-credit-card'; // reset classes
    const activeAcc = savedAccounts.find(a => a.active);
    if (activeAcc && activeAcc.name.includes("HDFC")) {
      card.classList.add('teal');
    } else {
      card.classList.add('stealth');
    }
  }

  // Render accounts
  renderAccounts(savedAccounts);

  // --- C. Initialize Core ---
  // Pre-seed conic ring at default 84 so it's visible before data loads
  const ringEl = document.querySelector('.risk-ring');
  if (ringEl) ringEl.style.setProperty('--score', '84');

  updateGreeting();
  refreshRevenueData();   // populate chart data from localStorage
  renderCalendar(calendarDate);
  renderTransactionList(); // replace hardcoded tx items
  updateStatCards();       // replace hardcoded stat card values

  // Update portfolio allocation from localStorage config
  const portfolio = JSON.parse(localStorage.getItem('flux_portfolio') || '{}');
  if (portfolio.equity !== undefined) {
    const bars = document.querySelectorAll('.asset-bar');
    if (bars[0]) { bars[0].style.width = `${portfolio.equity}%`; bars[0].title = `Equity: ${portfolio.equity}%`; }
    if (bars[1]) { bars[1].style.width = `${portfolio.crypto}%`; bars[1].title = `Crypto: ${portfolio.crypto}%`; }
    if (bars[2]) { bars[2].style.width = `${portfolio.cash}%`;   bars[2].title = `Cash: ${portfolio.cash}%`; }
  }

  // Update calendar footer with real monthly total + MoM badge
  const calFooterVal   = document.querySelector('.cal-footer-val');
  const calFooterBadge = document.querySelector('.cal-footer-badge');
  if (calFooterVal) {
    const txs  = getTransactions();
    const now  = new Date();

    const monthTotal = txs
      .filter(t => isSameMonth(new Date(t.date), now) && t.amount < 0)
      .reduce((s, t) => s + Math.abs(t.amount), 0);
    const prevTotal = txs
      .filter(t => isSamePeriodLastMonth(new Date(t.date), now) && t.amount < 0)
      .reduce((s, t) => s + Math.abs(t.amount), 0);

    calFooterVal.textContent = '₹' + monthTotal.toLocaleString('en-IN');

    if (calFooterBadge) {
      if (prevTotal > 0) {
        const momPct = ((monthTotal - prevTotal) / prevTotal * 100).toFixed(1);
        const isUp   = parseFloat(momPct) >= 0;
        calFooterBadge.textContent = (isUp ? '↑ ' : '↓ ') + Math.abs(momPct) + '%';
        calFooterBadge.style.color = isUp ? '#ff4d6d' : '#00e5a0'; // spend up = red, down = teal
      } else {
        // No prior-month baseline → clear the hardcoded placeholder badge.
        calFooterBadge.textContent = '';
      }
    }
  }

  // --- C1b. Hero card live metrics ---
  updateHeroMetrics();

  // --- C1c. Next Milestone — §2.8 live from flux_portfolio.goal ---
  const milestoneEl = document.getElementById('milestoneText');
  if (milestoneEl) {
    const pCfg    = JSON.parse(localStorage.getItem('flux_portfolio') || '{}');
    const goal    = pCfg.goal || 15000000;       // default ₹1.5Cr
    // Use cumulative income as a proxy for progress (all-time inflows)
    const txs2    = getTransactions();
    const allIncome = txs2.filter(t => t.amount > 0).reduce((s, t) => s + t.amount, 0);
    const balBase   = calculateTotalBalance(savedAccounts);
    const progress  = Math.max(allIncome, balBase);
    const pct       = Math.min(100, Math.round(progress / goal * 100));
    const goalFmt   = goal >= 1e7
      ? '₹' + (goal / 1e7).toFixed(1) + 'Cr'
      : goal >= 1e5
      ? '₹' + (goal / 1e5).toFixed(1) + 'L'
      : '₹' + goal.toLocaleString('en-IN');
    milestoneEl.textContent = `${goalFmt} (${pct}% Goal Reached)`;
  }

  // --- C2. Intelligence & Interactivity ---
  initTooltipEngine();
  initAssetInteractions();

  // --- C3. Polishing ---
  initOptimizationSummary();

  const prevBtn = document.getElementById('prevMonth');
  const nextBtn = document.getElementById('nextMonth');
  if (prevBtn) prevBtn.addEventListener('click', () => {
    calendarDate.setMonth(calendarDate.getMonth() - 1);
    renderCalendar(calendarDate);
  });
  if (nextBtn) nextBtn.addEventListener('click', () => {
    calendarDate.setMonth(calendarDate.getMonth() + 1);
    renderCalendar(calendarDate);
  });

  // --- D. Initialize Chart ---
  buildRevenueChart('yearly');

  // --- D. Revenue Filters ---
  document.querySelectorAll('[data-revenue-filter]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-revenue-filter]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      buildRevenueChart(btn.dataset.revenueFilter);
    });
  });

  // --- E. Counters ---
  const nw = document.getElementById('netWorthValue');
  const baseNetWorth = calculateTotalBalance(savedAccounts);
  if (nw) animateCounter(nw, baseNetWorth);

  document.querySelectorAll('.stat-val[data-count]').forEach(el => {
    animateCounter(el, parseInt(el.dataset.count), 1200, el.dataset.prefix || '₹');
  });

  // --- F. Account Addition Logic ---
  // Opens the Account Setup modal with a fresh, empty account row so the
  // user enters real details (name / balance / card / expiry) instead of
  // injecting a hardcoded account.
  const addAccountBtn = document.getElementById('addAccountBtn');
  if (addAccountBtn) {
    addAccountBtn.addEventListener('click', () => {
      if (typeof window.openAccountSetup === 'function') {
        window.openAccountSetup({ addRow: true });
      }
    });
  }


  // --- H. Popovers ---
  const popoverTriggers = [
    { btn: 'rangeBtn', pop: 'rangePopover' },
    { btn: 'filterBtn', pop: 'filterPopover' }
  ];

  popoverTriggers.forEach(t => {
    const btn = document.getElementById(t.btn);
    const pop = document.getElementById(t.pop);
    const dashBody = document.querySelector('.dash-body');

    if (btn && pop) {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        
        const isAlreadyVisible = pop.classList.contains('show');

        // Close others
        popoverTriggers.forEach(other => {
          const op = document.getElementById(other.pop);
          if (op) op.classList.remove('show');
          const parentCard = op ? op.closest('.glass-card') : null;
          if (parentCard) parentCard.classList.remove('focused-card');
        });

        if (!isAlreadyVisible) {
          pop.classList.add('show');
          if (dashBody) dashBody.classList.add('is-focusing');
          const card = pop.closest('.glass-card');
          if (card) card.classList.add('focused-card');
        } else {
          pop.classList.remove('show');
          if (dashBody) dashBody.classList.remove('is-focusing');
        }
      });
    }
  });

  // Make all popover items keyboard-activatable
  document.querySelectorAll('.popover-item[tabindex="0"]').forEach(keyActivatable);

  // Range popover presets
  document.querySelectorAll('#rangePopover [data-range]').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('#rangePopover .popover-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      const range = item.dataset.range;
      if (range === '7d') {
        document.querySelectorAll('[data-revenue-filter]').forEach(b => b.classList.remove('active'));
        document.querySelector('[data-revenue-filter="weekly"]')?.classList.add('active');
        buildRevenueChart('weekly');
      } else if (range === '30d') {
        buildDaysChart(30);
      } else if (range === 'ytd') {
        buildYTDChart();
      }
      document.getElementById('rangePopover')?.classList.remove('show');
    });
  });

  // Custom range toggle
  const customRangeToggle = document.getElementById('customRangeToggle');
  const customRangeInputs = document.getElementById('customRangeInputs');
  if (customRangeToggle && customRangeInputs) {
    customRangeToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      customRangeInputs.style.display = customRangeInputs.style.display === 'none' ? 'flex' : 'none';
    });
  }
  document.getElementById('applyCustomRange')?.addEventListener('click', (e) => {
    e.stopPropagation();
    const from = document.getElementById('customRangeFrom')?.value;
    const to   = document.getElementById('customRangeTo')?.value;
    if (!from || !to) { showToast("Select both a start and end date", "error"); return; }
    buildDateRangeChart(new Date(from), new Date(to));
    document.getElementById('rangePopover')?.classList.remove('show');
  });

  // Currency Switching Logic
  document.querySelectorAll('[data-currency]').forEach(item => {
    item.addEventListener('click', () => {
      const currency = item.dataset.currency;
      if (currency === currentCurrency) return;
      
      document.querySelectorAll('[data-currency]').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      setCurrency(currency);
    });
  });

  // Metric Switching Logic
  const metricMap = {
     "Expenses": "profit",
     "Net Income": "income",
     "Reset Filters": "revenue"
  };

  document.querySelectorAll('#filterPopover .popover-item').forEach(item => {
    item.addEventListener('click', () => {
      const text = item.textContent.trim();
      if (metricMap[text]) {
        document.querySelectorAll('#filterPopover .popover-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        setMetric(metricMap[text]);
      }
    });
  });

  document.addEventListener('click', () => {
    const dashBody = document.querySelector('.dash-body');
    popoverTriggers.forEach(t => {
      const p = document.getElementById(t.pop);
      if (p) {
        p.classList.remove('show');
        const card = p.closest('.glass-card');
        if (card) card.classList.remove('focused-card');
      }
    });
    if (dashBody) dashBody.classList.remove('is-focusing');
  });

  // --- I. Card Tilt ---
  if (card) {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      card.style.transform = `rotateY(${x * 20}deg) rotateX(${-y * 20}deg) translateY(-5px)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = 'rotateY(0) rotateX(0) translateY(0)';
    });
  }
});

/* ────────────────────────────────────────────
   4. HELPER FUNCTIONS
───────────────────────────────────────────── */

function renderAccounts(accounts) {
  const accountList = document.getElementById('accountList');
  const portfolioBanks = null; // removed from hero — account names shown only in Card Vault
  if (!accountList) return;

  accountList.innerHTML = '';
  if (!accounts.length) {
    // New account with no linked banks/cards — prompt, never demo accounts.
    accountList.innerHTML = `
      <div class="acc-mini-empty" style="padding:16px;text-align:center;color:var(--text-secondary,rgba(255,255,255,.5));font-size:13px;line-height:1.5;">
        No accounts linked yet.<br><span style="font-size:12px;opacity:.7;">Add one to see balances here.</span>
      </div>`;
    updateAssetAllocation();
    return;
  }
  accounts.forEach((acc, idx) => {
    const item = document.createElement('div');
    item.className = `acc-mini-item ${acc.active ? 'active' : ''}`;
    item.innerHTML = `
      <div class="acc-mini-info">
        <div class="acc-mini-name">${acc.name}</div>
        <div class="acc-mini-val">${acc.val}</div>
      </div>
      ${acc.active ? '<div class="acc-mini-dot"></div>' : ''}
    `;

    item.setAttribute('role', 'button');
    item.setAttribute('tabindex', acc.active ? '-1' : '0');
    item.setAttribute('aria-label', `${acc.name} — ${acc.val}${acc.active ? ', active' : ', set as default'}`);
    item.addEventListener('click', () => {
      if (acc.active) return;
      swapActiveAccount(idx);
    });
    if (!acc.active) keyActivatable(item);

    accountList.appendChild(item);
  });

  if (portfolioBanks) {
    const names = accounts.map(a => a.name);
    portfolioBanks.textContent = names.length <= 2 ? names.join(", ") : `${names[0]}, ${names[1]}, +${names.length - 2}`;
  }

  updateAssetAllocation();
}

function swapActiveAccount(index) {
  const accounts = JSON.parse(localStorage.getItem('flux_accounts') || '[]');
  if (!accounts.length) return;
  const perspective = document.querySelector('.card-perspective');
  const cardNum = document.getElementById('cardNumber');
  const cardHolder = document.getElementById('cardHolder');
  
  if (!perspective || !cardNum) return;

  // 1. Mark active in data
  accounts.forEach((a, i) => a.active = (i === index));
  localStorage.setItem('flux_accounts', JSON.stringify(accounts));

  // 2. Trigger Swoosh Animation
  perspective.classList.remove('enter');
  perspective.classList.add('exit');

  setTimeout(() => {
    // Update visuals while invisible
    const active = accounts[index];

    // Prefer the card data stored on the account (set during Account Setup);
    // only fall back to a masked placeholder when it's missing.
    const num  = active.cardNum || "•••• •••• •••• 0000";
    const real = active.realNum || num;
    const cardExpiry = document.getElementById('cardExpiry');

    // Holder is the account owner (the profile), not a per-bank mock.
    const profile = JSON.parse(localStorage.getItem('flux_profile') || '{}');
    const holder = (profile.name ? profile.name : (cardHolder?.textContent || 'CARD HOLDER')).toUpperCase();

    // Set card theme
    const cardEl = document.getElementById('mainCreditCard');
    if (cardEl) {
      cardEl.className = 'flux-credit-card';
      cardEl.classList.add(active.name.includes("HDFC") ? 'teal' : 'stealth');
    }

    cardNum.textContent = num;
    cardNum.dataset.hidden = num;
    cardNum.dataset.real = real;
    if (cardHolder) cardHolder.textContent = holder;

    // Reflect the account's expiry (shown by default; masked value kept for the vault toggle)
    if (cardExpiry) {
      const realExp = active.realExpiry || '••/••';
      cardExpiry.textContent = realExp;
      cardExpiry.dataset.real = realExp;
      cardExpiry.dataset.hidden = '••/••';
    }

    renderAccounts(accounts);
    showToast(`${active.name} Set as Default`, "info");

    // Re-enter
    perspective.classList.remove('exit');
    perspective.classList.add('enter');
  }, 400);
}

function updateAssetAllocation() {
  const bars = document.querySelectorAll('.asset-bar');
  if (bars.length < 3) return;

  // Single source of truth: the allocation the user set in Account Setup.
  // Falls back to a sensible default only when nothing is configured.
  const p = JSON.parse(localStorage.getItem('flux_portfolio') || '{}');
  const equityPct = p.equity ?? 24;
  const cryptoPct = p.crypto ?? 14;
  const cashPct   = p.cash   ?? 25;

  bars[0].style.width = `${equityPct}%`;
  bars[0].title = `Equity: ${equityPct}%`;
  bars[1].style.width = `${cryptoPct}%`;
  bars[1].title = `Crypto: ${cryptoPct}%`;
  bars[2].style.width = `${cashPct}%`;
  bars[2].title = `Cash: ${cashPct}%`;
}

function calculateTotalBalance(accounts) {
  const total = accounts.reduce((sum, acc) => {
    const num = parseInt(acc.val.replace(/[₹,]/g, ''));
    return sum + num;
  }, 0);
  return Math.max(0, total);
}


// ── Priority 3: UX & Polish (Toasts, Skeletons, Empty States) ──

function showToast(message, type = 'success') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.setAttribute('role', 'alert');

  let icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>';
  if (type === 'error') icon = '<span aria-hidden="true">!</span>';
  if (type === 'info')  icon = '<span aria-hidden="true">i</span>';

  toast.innerHTML = `
    <div class="toast-icon">${icon}</div>
    <div class="toast-msg">${message}</div>
    <button type="button" class="toast-close" aria-label="Dismiss notification">×</button>
  `;

  // Newest at top (closest to the anchor corner)
  container.prepend(toast);

  // Wire dismiss button
  toast.querySelector('.toast-close').addEventListener('click', () => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 500);
  });

  // Trigger reveal
  requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.add('show')));

  // Auto-dismiss after 5 s
  const timer = setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 500);
  }, 5000);

  // Cancel auto-dismiss while hovered (user is reading)
  toast.addEventListener('mouseenter', () => clearTimeout(timer));
  toast.addEventListener('mouseleave', () => {
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 500);
    }, 2000);
  });
}

function initLoadingSequence() {
  const splash = document.getElementById('splashScreen');

  // Set ARIA current for active nav (always)
  const currentPath = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-item').forEach(link => {
    if (link.getAttribute('href') === currentPath) link.setAttribute('aria-current', 'page');
  });

  if (sessionStorage.getItem('flux_splash_shown')) {
    // Already shown this session — skip immediately
    if (splash) splash.style.display = 'none';
    document.body.classList.remove('loading-state');
    return;
  }

  sessionStorage.setItem('flux_splash_shown', '1');
  document.body.classList.add('loading-state');

  setTimeout(() => { if (splash) splash.classList.add('fade-out'); }, 1800);
  setTimeout(() => {
    document.body.classList.remove('loading-state');
    showToast("FLUX Intelligence Activated", "info");
  }, 2600);
}

function checkEmptyStates() {
  const list = document.getElementById('liveTransactionList');
  if (list && list.children.length === 0) {
    list.innerHTML = `
      <div class="tx-empty-state">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2" aria-hidden="true">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <strong>No recent transactions</strong>
        <span>Set up your account to see live transaction data</span>
      </div>`;
  }
}

/* ────────────────────────────────────────────
   §9.2  DAILY BRIEFING CARD  — near-realtime
   Backend-pushed market data drives the movers, the
   portfolio change, net worth and the hero 24h metric.
   Polls live quotes on an interval; falls back to
   plausible simulated values when the backend is offline.
───────────────────────────────────────────── */
// Centralised config — js/flux-data.js (loaded first) resolves the API base
// (FLUX_CONFIG override → :8000 on localhost → same-origin in production).
const FLUX_CONFIG = Object.assign(
  { apiBase: (typeof window !== 'undefined' && window.FLUX_API) || 'http://localhost:8000' },
  (typeof window !== 'undefined' && window.FLUX_CONFIG) || {}
);
const API_BASE = FLUX_CONFIG.apiBase;   // keep alias so rest of code is unchanged
const BRIEFING_POLL_MS = 30000;         // refresh cadence (30s)
let _briefingTimer = null;
let _briefingAbort = null;              // AbortController for in-flight fetch

// No demo accounts — a user with no accounts has ₹0 net worth, not a fake one.
const _defaultAccounts = [];

function _avgChange(assets) {
  const v = (assets || []).filter(a => a && a.change_pct != null).map(a => a.change_pct);
  return v.length ? v.reduce((s, x) => s + x, 0) / v.length : 0;
}

// Blend the live market move into a portfolio daily change %,
// weighted by the user's equity / crypto allocation (cash stays flat).
function blendedPortfolioPct(stockAvg, cryptoAvg) {
  const p  = JSON.parse(localStorage.getItem('flux_portfolio') || '{}');
  const eq = (p.equity ?? 24) / 100;
  const cr = (p.crypto ?? 14) / 100;
  return eq * stockAvg + cr * cryptoAvg;
}

async function fetchLiveQuotes(signal) {
  const opts = signal ? { signal } : {};
  const [cryptoRes, stocksRes] = await Promise.all([
    fetch(`${API_BASE}/market/quotes/crypto`, opts).then(r => r.json()),
    fetch(`${API_BASE}/market/quotes/stocks`, opts).then(r => r.json()),
  ]);
  return { crypto: cryptoRes.assets || [], stocks: stocksRes.assets || [] };
}

// Market quotes are USD. Sub-dollar coins (ADA, SHIB) need more decimals or they read as $0.
function fmtUsdQuote(v) {
  const digits = v >= 100 ? 0 : v >= 1 ? 2 : v >= 0.01 ? 4 : 8;
  return '$' + v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

// Plausible, gently-drifting values so the card is never empty when offline.
function simulatedQuotes() {
  const wave = (seed) => +(Math.sin(Date.now() / 11000 + seed) * 2.3 + (Math.random() - 0.5) * 0.7).toFixed(2);
  return {
    crypto: [
      { sub: 'BTC', name: 'Bitcoin',  change_pct: wave(1) },
      { sub: 'ETH', name: 'Ethereum', change_pct: wave(2) },
      { sub: 'SOL', name: 'Solana',   change_pct: wave(3) },
    ],
    stocks: [
      { sub: 'NVDA', name: 'NVIDIA', change_pct: wave(4) },
      { sub: 'AAPL', name: 'Apple',  change_pct: wave(5) },
      { sub: 'TSLA', name: 'Tesla',  change_pct: wave(6) },
    ],
  };
}

// Push the live blended market move into the hero "24h Change" metric.
// (Net worth stays the settled account balance; the briefing card carries
//  the live market-adjusted figure, so the headline number doesn't jump.)
function reflectLiveHero(chgPct) {
  document.querySelectorAll('.metric-item').forEach(m => {
    const lbl = m.querySelector('.metric-label')?.textContent?.trim();
    const val = m.querySelector('.metric-val');
    if (val && lbl === '24h Change') {
      const up = chgPct >= 0;
      val.textContent = (up ? '+' : '') + chgPct.toFixed(2) + '%';
      val.className   = 'metric-val ' + (up ? 'positive' : 'negative');
    }
  });
}

async function refreshBriefing() {
  const headEl   = document.getElementById('briefingAction');
  const moversEl = document.getElementById('briefingMovers');
  const portVal  = document.getElementById('briefingPortVal');
  const portChg  = document.getElementById('briefingPortChg');
  const dotEl    = document.getElementById('briefingLiveDot');
  const cardEl   = document.getElementById('briefingCard');
  if (!headEl) return;

  // Cancel any in-flight request from a previous call
  if (_briefingAbort) { _briefingAbort.abort(); }
  _briefingAbort = new AbortController();

  if (cardEl) cardEl.classList.add('loading');

  let data, live = true;
  try {
    data = await fetchLiveQuotes(_briefingAbort.signal);
    if (!data.crypto.length && !data.stocks.length) throw new Error('empty');
  } catch (err) {
    if (err && err.name === 'AbortError') return;   // superseded by newer call
    data = simulatedQuotes();
    live = false;
  } finally {
    if (cardEl) cardEl.classList.remove('loading');
  }

  // Live indicator
  if (dotEl) {
    dotEl.classList.toggle('offline', !live);
    dotEl.parentElement && (dotEl.parentElement.title = live ? 'Live market data' : 'Simulated — backend offline');
    const lbl = document.getElementById('briefingLiveLabel');
    if (lbl) lbl.textContent = live ? 'LIVE' : 'SIM';
  }

  // Top 3 movers by absolute move
  const all = [...data.crypto, ...data.stocks].filter(a => a.change_pct != null);
  all.sort((a, b) => Math.abs(b.change_pct) - Math.abs(a.change_pct));
  const top3 = all.slice(0, 3);

  const CRYPTO_SUBS = new Set(['BTC','ETH','SOL','BNB','XRP','DOGE','ADA','AVAX','MATIC','DOT']);
  const p = JSON.parse(localStorage.getItem('flux_portfolio') || '{}');
  const holdsCrypto = (p.crypto ?? 14) > 0;
  const holdsEquity = (p.equity ?? 24) > 0;

  if (moversEl) {
    moversEl.innerHTML = top3.map(a => {
      const up  = a.change_pct >= 0;
      const pct = (up ? '+' : '') + a.change_pct.toFixed(2) + '%';
      const priceStr = a.price != null ? `<div class="mover-price">${fmtUsdQuote(a.price)}</div>` : '';
      return `<div class="mover-chip">
        <div class="mover-sym">${a.sub}</div>
        ${priceStr}
        <div class="mover-pct ${up ? 'up' : 'down'}">${pct}</div>
      </div>`;
    }).join('');
  }

  if (top3.length) {
    const leader = top3[0];
    const up = leader.change_pct >= 0;
    const isCrypto = CRYPTO_SUBS.has(leader.sub);
    const userHolds = isCrypto ? holdsCrypto : holdsEquity;
    const assetClass = isCrypto ? 'crypto' : 'equity';
    const allocation = isCrypto ? (p.crypto ?? 14) : (p.equity ?? 24);
    const offlineNote = live ? '' : ' <span style="opacity:.5">(simulated)</span>';

    let tail = '';
    if (userHolds) {
      tail = up
        ? ` You have ${allocation}% in ${assetClass} — consider reviewing your position.`
        : ` You have ${allocation}% in ${assetClass} — monitor your exposure.`;
    }

    headEl.innerHTML = (up
      ? `${leader.name} is today's top mover at <strong class="teal">+${leader.change_pct.toFixed(2)}%</strong>.${tail}`
      : `${leader.name} is under pressure at <strong class="red">${leader.change_pct.toFixed(2)}%</strong>.${tail}`) + offlineNote;
  }

  // Near-realtime portfolio value & change, blended from live market
  const stockAvg  = _avgChange(data.stocks);
  const cryptoAvg = _avgChange(data.crypto);
  const chgPct    = blendedPortfolioPct(stockAvg, cryptoAvg);

  const accounts = JSON.parse(localStorage.getItem('flux_accounts') || '[]');
  const base     = calculateTotalBalance(accounts.length ? accounts : _defaultAccounts);
  const liveVal  = Math.round(base * (1 + chgPct / 100));

  // No accounts → no portfolio value to blend; show a clean zero, not a
  // market-drifted fake number.
  if (portVal) portVal.textContent = accounts.length ? '₹' + liveVal.toLocaleString('en-IN') : '₹0';
  if (portChg) {
    const up = chgPct >= 0;
    portChg.textContent = (up ? '↑ ' : '↓ ') + Math.abs(chgPct).toFixed(2) + '%';
    portChg.className   = 'briefing-port-chg ' + (up ? 'up' : 'down');
  }

  reflectLiveHero(chgPct);
}

async function initDailyBriefing() {
  const dateEl = document.getElementById('briefingDate');
  if (dateEl) {
    const now = new Date();
    dateEl.textContent = now.toLocaleDateString('en-IN', { weekday:'short', day:'numeric', month:'short' });
    dateEl.setAttribute('datetime', now.toISOString().slice(0, 10));
  }

  await refreshBriefing();

  // Poll for near-realtime updates; avoid stacking timers across re-inits.
  if (_briefingTimer) clearInterval(_briefingTimer);
  _briefingTimer = setInterval(refreshBriefing, BRIEFING_POLL_MS);

  // Pause polling when the tab is hidden, resume (with an immediate refresh) on focus.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (_briefingTimer) { clearInterval(_briefingTimer); _briefingTimer = null; }
    } else if (!_briefingTimer) {
      refreshBriefing();
      _briefingTimer = setInterval(refreshBriefing, BRIEFING_POLL_MS);
    }
  });
}

// Start simulation and init sequence
document.addEventListener('DOMContentLoaded', () => {
  initLoadingSequence();
  initDailyBriefing();

  // Hook into existing buttons for feedback
  document.querySelectorAll('.btn-action').forEach(btn => {
    btn.addEventListener('click', () => {
      // Emergency Check
      const card = document.getElementById('mainCreditCard');
      if (card && card.classList.contains('is-frozen')) {
        showToast("Access Denied: Card is Frozen", "error");
        return;
      }

      const text = btn.textContent.trim();
      
      // Add loading state
      btn.classList.add('btn-loading');

      setTimeout(() => {
        btn.classList.remove('btn-loading');

        if (text === "Deposit") {
          window.location.href = '../pages/payments.html?action=deposit';
        } else if (text === "Send") {
          window.location.href = '../pages/payments.html?action=send';
        } else if (text === "Add Card") showToast("HDFC Platinum Account Connected", "success");
        else showToast("Request Processed Successfully", "success");
      }, 1200);
    });
  });

  // --- K. Freeze Card Logic ---
  const freezeBtn = document.getElementById('freezeBtn');
  if (freezeBtn) {
    freezeBtn.addEventListener('click', () => {
      const card = document.getElementById('mainCreditCard');
      if (!card) return;

      const isFrozen = card.classList.toggle('is-frozen');
      freezeBtn.classList.toggle('frozen-active');

      if (isFrozen) {
        showToast("EMERGENCY PROTOCOL: Card Frozen", "error");
      } else {
        showToast("Protocol Cleared: Card Active", "success");
      }
    });
  }

  // --- J. History button → open full transaction modal ---
  const historyBtn = document.getElementById('historyBtn');
  if (historyBtn) {
    historyBtn.addEventListener('click', () => {
      _txPage = 0;
      const txModal = document.getElementById('txModal');
      renderTransactionModal();
      if (txModal) {
        txModal.classList.add('show');
        document.body.style.overflow = 'hidden';
      }
    });
  }

  // Transaction list is fully rendered by renderTransactionList() above.

  // --- Auto-mark remaining decorative SVGs for screen readers ---
  document.querySelectorAll('svg:not([aria-label]):not([role])').forEach(svg => {
    svg.setAttribute('aria-hidden', 'true');
  });

  // --- Debounced chart resize (avoid janky redraws on every pixel during resize) ---
  let _resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(() => {
      if (typeof Chart !== 'undefined') {
        Chart.instances && Object.values(Chart.instances).forEach(c => {
          try { c.resize(); } catch (_) {}
        });
      }
    }, 150);
  });

  // --- localStorage schema versioning ---
  const SCHEMA_VERSION = 1;
  const storedVer = parseInt(localStorage.getItem('flux_schema_version') || '0', 10);
  if (storedVer < SCHEMA_VERSION) {
    // Future migrations go here — for now just stamp the current version.
    localStorage.setItem('flux_schema_version', String(SCHEMA_VERSION));
  }
});

function revealVault() {
  const num = document.getElementById('cardNumber');
  const exp = document.getElementById('cardExpiry');
  if (!num || !exp) return;

  const isHidden = num.textContent.includes('•');
  const targetNum = isHidden ? num.dataset.real : num.dataset.hidden;
  const targetExp = isHidden ? exp.dataset.real : exp.dataset.hidden;

  shuffleText(num, targetNum);
  shuffleText(exp, targetExp);

  if (isHidden) {
    showToast("Decryption Complete. Masking in 10s.", "success");
    setTimeout(() => {
      if (!num.textContent.includes('•')) revealVault(); // Auto-hide
    }, 10000);
  }
}

function shuffleText(el, finalValue) {
  const chars = "0123456789•/ ";
  let iteration = 0;
  const interval = setInterval(() => {
    el.innerText = el.innerText.split("")
      .map((_, index) => {
        if (index < iteration) return finalValue[index];
        return chars[Math.floor(Math.random() * chars.length)];
      })
      .join("");

    if (iteration >= finalValue.length) clearInterval(interval);
    iteration += 1 / 3;
  }, 30); // Faster for a professional terminal feel
}

function disableAddButton(btn) {
  btn.disabled = true;
  btn.style.opacity = '0.5';
  btn.style.cursor = 'not-allowed';
}

function animateCounter(el, target, duration = 1400, prefix = '₹') {
  const start = performance.now();
  const safeTarget = Math.max(0, target);
  const update = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(eased * safeTarget);
    el.textContent = prefix + current.toLocaleString('en-IN');
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
  // Guarantee landing on target even if RAF is throttled (tab inactive, mid-nav, etc.)
  setTimeout(() => {
    el.textContent = prefix + safeTarget.toLocaleString('en-IN');
  }, duration + 50);
}

/* ────────────────────────────────────────────
   6b. ACCESSIBILITY UTILITIES
───────────────────────────────────────────── */

// Focusable selector used for focus-trapping
const FOCUSABLE = 'a[href],button:not([disabled]),input,select,textarea,[tabindex]:not([tabindex="-1"])';

function trapFocus(modalEl) {
  const focusable = () => [...modalEl.querySelectorAll(FOCUSABLE)].filter(el => !el.closest('[hidden]'));
  const handler = (e) => {
    if (e.key !== 'Tab') return;
    const els = focusable();
    if (!els.length) { e.preventDefault(); return; }
    const first = els[0], last = els[els.length - 1];
    if (e.shiftKey) {
      if (document.activeElement === first) { e.preventDefault(); last.focus(); }
    } else {
      if (document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  };
  modalEl._trapHandler = handler;
  modalEl.addEventListener('keydown', handler);
  // Focus first focusable element when opened
  requestAnimationFrame(() => { focusable()[0]?.focus(); });
}

function releaseFocus(modalEl) {
  if (modalEl._trapHandler) {
    modalEl.removeEventListener('keydown', modalEl._trapHandler);
    delete modalEl._trapHandler;
  }
}

// Make any element keyboard-activatable (Enter/Space fires click)
function keyActivatable(el) {
  el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.click(); }
  });
}

/* ────────────────────────────────────────────
   7. TRANSACTION MODAL LOGIC
───────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const txModal = document.getElementById('txModal');
  const viewAllBtn = document.getElementById('viewAllTransactions');
  const closeModalBtn = document.getElementById('closeTxModalBtn');
  const fullTxList = document.getElementById('fullTransactionList');
  const exportCsv = document.getElementById('exportCsvBtn');
  const exportPdf = document.getElementById('exportPdfBtn');

  const openTxModal = () => {
    renderTransactionModal();
    txModal.classList.add('show');
    document.body.style.overflow = 'hidden';
    trapFocus(txModal);
  };

  const closeTxModal = () => {
    txModal.classList.remove('show');
    document.body.style.overflow = '';
    releaseFocus(txModal);
  };

  // Esc closes the modal
  txModal.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeTxModal(); });

  if (viewAllBtn) viewAllBtn.addEventListener('click', (e) => { e.preventDefault(); _txPage = 0; openTxModal(); });
  if (closeModalBtn) closeModalBtn.addEventListener('click', closeTxModal);

  // Close on outside click
  txModal.addEventListener('click', (e) => {
    if (e.target === txModal) closeTxModal();
  });

  // Search / filter / sort — re-render on change
  ['txSearch', 'txCategoryFilter', 'txSort'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', () => { _txPage = 0; renderTransactionModal(); });
    document.getElementById(id)?.addEventListener('change', () => { _txPage = 0; renderTransactionModal(); });
  });

  // Event delegation for flag/view inside the full modal list
  fullTxList?.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const item = btn.closest('[data-tx-index]');
    if (!item) return;
    const pageIdx = parseInt(item.dataset.txIndex);
    const allTxs = getFilteredTxs();
    const tx = allTxs[pageIdx];
    if (!tx) return;

    // Find the tx in the raw storage by identity to mutate it
    const rawTxs = getTransactions();
    const rawIdx = rawTxs.findIndex(r => r.date === tx.date && r.title === tx.title && r.amount === tx.amount);
    if (rawIdx === -1) return;

    if (btn.dataset.action === 'flag') {
      rawTxs[rawIdx].flagged = !rawTxs[rawIdx].flagged;
      localStorage.setItem('flux_transactions', JSON.stringify(rawTxs));
      renderTransactionModal();
      showToast(rawTxs[rawIdx].flagged ? '⚑ Flagged for audit' : 'Flag removed', rawTxs[rawIdx].flagged ? 'error' : 'info');
    } else if (btn.dataset.action === 'view') {
      const t = rawTxs[rawIdx];
      const isIn = t.amount > 0;
      const amt  = Math.abs(t.amount);
      const dStr = new Date(t.date).toLocaleDateString('en-IN', { weekday:'long', day:'2-digit', month:'long', year:'numeric' });
      showToast(`<b>${t.title}</b><br/>${dStr}<br/>${capitalize(t.category)} · ${isIn ? '+' : '−'}₹${amt.toLocaleString('en-IN')}${t.account ? '<br/>Account: ' + t.account : ''}${t.flagged ? '<br/><span style="color:#ff4d4d">⚑ Flagged</span>' : ''}`, 'info');
    }
  });

  // §2.6 — Export: download real transaction data as CSV
  if (exportCsv) exportCsv.addEventListener('click', () => {
    const txs = getTransactions();
    if (!txs.length) { showToast("No transactions to export", "error"); return; }

    exportCsv.classList.add('btn-loading');

    const header = ['Date', 'Title', 'Category', 'Amount (INR)', 'Account'];
    const rows   = txs.map(t => [
      new Date(t.date).toLocaleDateString('en-IN', { day:'2-digit', month:'short', year:'numeric' }),
      `"${(t.title || t.description || '').replace(/"/g, '""')}"`,
      t.category || '—',
      t.amount.toFixed(2),
      t.account || '—'
    ]);

    const csv  = [header, ...rows].map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `flux-ledger-${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    setTimeout(() => {
      exportCsv.classList.remove('btn-loading');
      showToast(`Exported ${txs.length} transactions to CSV`, "success");
    }, 400);
  });

  if (exportPdf) exportPdf.addEventListener('click', () => {
    const txs = getTransactions();
    if (!txs.length) { showToast("No transactions to export", "error"); return; }
    exportPdf.classList.add('btn-loading');

    const rows = txs.map(t => {
      const isIn = t.amount > 0;
      const amt  = Math.abs(t.amount);
      const dStr = new Date(t.date).toLocaleDateString('en-IN', { day:'2-digit', month:'short', year:'numeric' });
      return `<tr>
        <td>${dStr}</td>
        <td>${t.title || '—'}</td>
        <td>${capitalize(t.category || '—')}</td>
        <td style="color:${isIn ? '#16a34a' : '#dc2626'};font-weight:600">${isIn ? '+' : '−'}₹${amt.toLocaleString('en-IN')}</td>
        <td>${t.account || '—'}</td>
        ${t.flagged ? '<td style="color:#dc2626">⚑ Flagged</td>' : '<td>—</td>'}
      </tr>`;
    }).join('');

    const totalIncome  = txs.filter(t => t.amount > 0).reduce((s,t) => s+t.amount, 0);
    const totalExpense = txs.filter(t => t.amount < 0).reduce((s,t) => s+Math.abs(t.amount), 0);

    const win = window.open('', '_blank');
    win.document.write(`<!DOCTYPE html><html><head><title>FLUX Transaction Ledger</title>
<style>
  body { font-family: Arial, sans-serif; padding: 32px; color: #111; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .sub { font-size: 13px; color: #666; margin-bottom: 24px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #f5f5f5; text-align: left; padding: 8px 12px; border-bottom: 2px solid #ddd; }
  td { padding: 7px 12px; border-bottom: 1px solid #eee; }
  tr:hover td { background: #fafafa; }
  .summary { margin-top: 24px; display: flex; gap: 32px; }
  .sum-box { background: #f9f9f9; border-radius: 8px; padding: 12px 20px; }
  .sum-label { font-size: 11px; color: #888; text-transform: uppercase; }
  .sum-val { font-size: 18px; font-weight: 700; margin-top: 2px; }
  @media print { button { display: none; } }
</style></head><body>
<h1>FLUX — Transaction Ledger</h1>
<div class="sub">Generated ${new Date().toLocaleDateString('en-IN', { day:'2-digit', month:'long', year:'numeric' })} · ${txs.length} transactions</div>
<table>
  <thead><tr><th>Date</th><th>Description</th><th>Category</th><th>Amount</th><th>Account</th><th>Note</th></tr></thead>
  <tbody>${rows}</tbody>
</table>
<div class="summary">
  <div class="sum-box"><div class="sum-label">Total Income</div><div class="sum-val" style="color:#16a34a">+₹${totalIncome.toLocaleString('en-IN')}</div></div>
  <div class="sum-box"><div class="sum-label">Total Expenses</div><div class="sum-val" style="color:#dc2626">−₹${totalExpense.toLocaleString('en-IN')}</div></div>
  <div class="sum-box"><div class="sum-label">Net</div><div class="sum-val">₹${(totalIncome - totalExpense).toLocaleString('en-IN')}</div></div>
</div>
<br><button onclick="window.print()" style="padding:8px 20px;font-size:13px;cursor:pointer;">Print / Save as PDF</button>
</body></html>`);
    win.document.close();

    setTimeout(() => exportPdf.classList.remove('btn-loading'), 300);
  });
});

/* ────────────────────────────────────────────
   LIVE RE-HYDRATION
   js/flux-data.js dispatches "flux:data-updated" whenever fresh DB data
   lands in localStorage (initial hydration finishes AFTER DOMContentLoaded,
   so without this the dashboard renders one cycle of stale data).
───────────────────────────────────────────── */
window.addEventListener('flux:data-updated', () => {
  try {
    const accounts = JSON.parse(localStorage.getItem('flux_accounts') || '[]');
    if (accounts.length) {
      renderAccounts(accounts);
      const nw = document.getElementById('netWorthValue');
      if (nw) animateCounter(nw, calculateTotalBalance(accounts));
    }
    renderTransactionList();
    updateStatCards();
    updateHeroMetrics();
    renderCalendar(calendarDate);
    buildRevenueChart(document.querySelector('[data-revenue-filter].active')?.dataset.revenueFilter || 'yearly');
  } catch (e) {
    console.warn('[FLUX] dashboard re-hydration failed:', e);
  }
});
