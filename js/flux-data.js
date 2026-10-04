/* FLUX — DB Hydration Layer
 * ==========================
 * Single source of truth = MySQL (served by the backend at /db/*).
 *
 * Strategy (zero rewrites of page render code):
 *   1. On load, fetch the seeded dataset from /db/* for the demo user.
 *   2. Map each response into the exact localStorage shape the pages already
 *      read (flux_transactions, flux_accounts, flux_portfolio, …).
 *   3. If anything changed vs the cached copy, write it and dispatch
 *      "flux:data-updated" so pages can refresh their on-screen values
 *      in place — no full page reload.
 *   4. If the backend is unreachable, leave existing localStorage / seed.js
 *      defaults in place — the app still works offline.
 *
 * Load this BEFORE seed.js and the page scripts. It also stamps
 * flux_seeded so seed.js skips its hardcoded fallback once DB data is present.
 *
 * Override the backend with:  window.FLUX_CONFIG = { apiBase: '...' }
 */
(function fluxHydrate() {
  // API base resolution: explicit FLUX_CONFIG override → dev default (:8000 on
  // localhost) → same-origin (production behind a reverse proxy serving /…).
  function defaultApiBase() {
    const h = window.location.hostname;
    if (h === 'localhost' || h === '127.0.0.1' || h === '') {
      return 'http://' + (h || 'localhost') + ':8000';
    }
    return window.location.origin;
  }
  const API = ((typeof window !== 'undefined' && window.FLUX_CONFIG && window.FLUX_CONFIG.apiBase) ||
               defaultApiBase()).replace(/\/$/, '');
  const USER = (typeof window !== 'undefined' && window.FLUX_CONFIG && window.FLUX_CONFIG.userId) || 1;
  // Single source of truth for pages — read these instead of re-deriving.
  window.FLUX_API = API;
  window.FLUX_USER_ID = USER;

  /* ── Auth interceptor ─────────────────────────────────────────────────────
   * Wraps window.fetch so every request to the FLUX API automatically carries
   * the session token, and any 401 on personal data routes the user to the
   * login page. One hook here covers all pages — no per-call-site changes.
   */
  const TOKEN_KEY = 'flux_token';
  window.FluxAuth = {
    get token() { return localStorage.getItem(TOKEN_KEY); },
    set token(t) { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); },
    loginUrl() {
      // Pages live in /pages/, the marketing index at the root.
      return location.pathname.includes('/pages/') ? 'login.html' : 'pages/login.html';
    },
    logout() {
      localStorage.removeItem(TOKEN_KEY);
      // Wipe the hydrated personal dataset on logout — it must not survive
      // into the next (possibly different) user's session.
      ['flux_user', 'flux_transactions', 'flux_accounts', 'flux_portfolio', 'flux_recurring',
       'flux_contacts', 'flux_protocols',
       'flux_reward_states', 'flux_seeded', 'flux_sample_data'].forEach(k => localStorage.removeItem(k));
    },
  };

  /* ── Cold-start notice ────────────────────────────────────────────────────
   * The API is hosted on a free tier that spins down after ~15 minutes idle,
   * so the first request of a visit can take 30-60 s while the container wakes.
   * The static site renders instantly either way, which makes the wait look
   * like a hang rather than a cold start. This says which it is.
   *
   * Only shown after a request has been slow for 2.5 s — a warm backend
   * answers in well under that, so day-to-day use never sees it.
   */
  const FluxWaking = (function () {
    let inFlight = 0, timer = null, el = null;

    function node() {
      if (el) return el;
      el = document.createElement('div');
      el.id = 'flux-waking';
      el.setAttribute('role', 'status');
      el.setAttribute('aria-live', 'polite');
      el.innerHTML =
        '<span class="flux-waking__dot"></span>' +
        '<span>Waking the server — first load takes up to a minute.</span>';
      // Inlined so the notice works on every page without a CSS import, and
      // cannot be broken by a page stylesheet that loads later.
      el.style.cssText = [
        'position:fixed', 'left:50%', 'bottom:24px', 'transform:translateX(-50%)',
        'z-index:9999', 'display:flex', 'gap:10px', 'align-items:center',
        'padding:10px 16px', 'border-radius:999px',
        'background:rgba(17,17,20,.92)', 'color:#fff',
        'font:500 13px/1.4 system-ui,-apple-system,sans-serif',
        'box-shadow:0 6px 24px rgba(0,0,0,.28)',
        'backdrop-filter:blur(6px)',
      ].join(';');
      const dot = el.querySelector('.flux-waking__dot');
      dot.style.cssText =
        'width:8px;height:8px;border-radius:50%;background:#4ade80;' +
        'animation:flux-waking-pulse 1.2s ease-in-out infinite';
      const style = document.createElement('style');
      style.textContent =
        '@keyframes flux-waking-pulse{0%,100%{opacity:1}50%{opacity:.25}}' +
        '@media (prefers-reduced-motion:reduce){.flux-waking__dot{animation:none}}';
      document.head.appendChild(style);
      return el;
    }

    function show() {
      if (document.body && !document.getElementById('flux-waking')) {
        document.body.appendChild(node());
      }
    }
    function hide() {
      const n = document.getElementById('flux-waking');
      if (n) n.remove();
    }

    return {
      start() {
        inFlight += 1;
        if (timer === null) timer = setTimeout(show, 2500);
      },
      end() {
        inFlight = Math.max(0, inFlight - 1);
        if (inFlight === 0) {
          clearTimeout(timer);
          timer = null;
          hide();
        }
      },
    };
  })();

  /* ── Sample-data badge ────────────────────────────────────────────────────
   * js/seed.js fills localStorage with a hard-coded demo dataset when there is
   * no session (anonymous / offline). On a cold start that dataset is on screen
   * for up to a minute before the API answers, and it looks real. While it is
   * the data being shown, say so. Removed on the first successful hydration.
   */
  const SampleBadge = {
    show() {
      if (localStorage.getItem('flux_sample_data') !== '1') return;
      if (!document.body || document.getElementById('flux-sample-badge')) return;
      const el = document.createElement('div');
      el.id = 'flux-sample-badge';
      el.setAttribute('role', 'status');
      el.textContent = 'Showing sample data — API is waking up';
      el.style.cssText = [
        'position:fixed', 'left:50%', 'top:14px', 'transform:translateX(-50%)',
        'z-index:9999', 'padding:6px 14px', 'border-radius:999px',
        'background:#f59e0b', 'color:#111',
        'font:600 12px/1.4 system-ui,-apple-system,sans-serif',
        'box-shadow:0 4px 16px rgba(0,0,0,.25)', 'pointer-events:none',
      ].join(';');
      document.body.appendChild(el);
    },
    clear() {
      localStorage.removeItem('flux_sample_data');
      const n = document.getElementById('flux-sample-badge');
      if (n) n.remove();
    },
  };
  // seed.js loads after this file, so check once the page has parsed.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { if (!hydrated) SampleBadge.show(); });
  } else {
    setTimeout(() => { if (!hydrated) SampleBadge.show(); }, 0);
  }

  const _origFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if (!url.startsWith(API)) return _origFetch(input, init);

    const opts = Object.assign({}, init);
    const token = window.FluxAuth.token;
    if (token) {
      opts.headers = Object.assign({}, opts.headers, { 'Authorization': 'Bearer ' + token });
    }
    FluxWaking.start();
    return _origFetch(input, opts).then((res) => {
      // Expired/missing session on a protected route → go log in. /auth/*
      // is excluded so a failed login attempt doesn't redirect-loop.
      if (res.status === 401 && !url.includes('/auth/')) {
        window.FluxAuth.token = null;
        if (!location.pathname.endsWith('login.html')) {
          location.href = window.FluxAuth.loginUrl();
        }
      }
      return res;
    }).finally(() => FluxWaking.end());
  };

  // Global sign-out: any .logout-btn click ends the session before navigating.
  document.addEventListener('click', (e) => {
    if (e.target.closest && e.target.closest('.logout-btn')) window.FluxAuth.logout();
  });
  const SEED_VERSION = '2';            // must match js/seed.js SEED_VERSION

  const num = (v) => (v == null ? 0 : Number(v));
  const inr = (v) => '₹' + num(v).toLocaleString('en-IN');
  const get = (path) => fetch(`${API}${path}`).then((r) => {
    if (!r.ok) throw new Error(`${path} → ${r.status}`);
    return r.json();
  });

  // ── API → legacy localStorage shape mappers ──────────────────────────────
  const mapTransactions = (d) => (d.transactions || []).map((t) => ({
    id: t.ext_id || ('tx_' + t.id),
    title: t.title,
    date: new Date(t.tx_date).toISOString(),
    amount: num(t.amount),
    category: t.category,
    account: t.account,
  }));

  // Only masked card data is mapped — unmasked PAN/expiry must never be
  // persisted client-side (the API no longer returns them either).
  const mapAccounts = (d) => (d.accounts || []).map((a) => ({
    name: a.name,
    val: inr(a.balance),
    balance: num(a.balance),
    creditLimit: num(a.credit_limit),
    acctType: a.acct_type,
    active: !!a.active,
    cardNum: a.card_masked,
    expiry: a.expiry_masked,
  }));

  const mapPortfolio = (d) => {
    const p = d.allocation || {};
    return { equity: p.equity_pct ?? 45, crypto: p.crypto_pct ?? 30, cash: p.cash_pct ?? 25, goal: num(p.goal) || 15000000 };
  };

  const mapRecurring = (d) => (d.recurring || []).map((r) => ({
    title: r.title, amount: num(r.amount), dueDay: r.due_day, category: r.category,
  }));

  const mapContacts = (d) => (d.contacts || []).map((c) => ({
    name: c.name, fluxId: c.flux_id, initial: c.initial,
  }));

  const mapProtocols = (sec) => (sec.settings || []).slice(0, 3).map((s) => !!s.enabled);

  const mapRewardStates = (d) => {
    const out = {};
    (d.rewards || []).forEach((r) => { out[r.reward_key] = !!r.claimed; });
    return out;
  };

  // Write only if changed; track whether any key actually changed.
  let changed = false;
  function put(key, value) {
    const next = JSON.stringify(value);
    if (localStorage.getItem(key) !== next) { localStorage.setItem(key, next); changed = true; }
  }

  let retryDelay = 15000;          // grows to a 2-minute ceiling
  let retryTimer = null;
  let hydrated   = false;

  // One /db/bootstrap call; the 7 separate routes stay as a fallback for an
  // API that predates it (Pages and Render don't deploy at the same instant).
  function fetchAll() {
    return get(`/db/bootstrap`).catch((err) => {
      if (!/→ 404$/.test(err.message)) throw err;
      return Promise.all([
        get(`/db/transactions?limit=2000`),
        get(`/db/accounts`),
        get(`/db/portfolio`),
        get(`/db/recurring`),
        get(`/db/contacts`),
        get(`/db/security`),
        get(`/db/rewards`),
      ]).then(([transactions, accounts, portfolio, recurring, contacts, security, rewards]) =>
        ({ transactions, accounts, portfolio, recurring, contacts, security, rewards }));
    });
  }

  // The login page fetches /db/bootstrap before redirecting and leaves it in
  // sessionStorage, so the first page after login renders real data at once.
  function takePrefetched() {
    try {
      const raw = sessionStorage.getItem('flux_bootstrap');
      if (!raw) return null;
      sessionStorage.removeItem('flux_bootstrap');
      const { at, data } = JSON.parse(raw);
      return Date.now() - at < 60000 ? data : null;
    } catch (_) { return null; }
  }

  function apply(b) {
    const tx = b.transactions, acc = b.accounts, pf = b.portfolio, rec = b.recurring,
          con = b.contacts, sec = b.security, rew = b.rewards;
    // Expose raw payloads for any page that wants to read directly.
    window.FluxData = { tx, acc, pf, rec, con, sec, rew };

    changed = false;
    put('flux_transactions', mapTransactions(tx));
    put('flux_accounts', mapAccounts(acc));
    put('flux_portfolio', mapPortfolio(pf));
    put('flux_recurring', mapRecurring(rec));
    put('flux_contacts', mapContacts(con));
    put('flux_protocols', mapProtocols(sec));
    put('flux_reward_states', mapRewardStates(rew));

    // Stop seed.js from injecting its hardcoded fallback.
    if (localStorage.getItem('flux_seeded') !== SEED_VERSION) {
      localStorage.setItem('flux_seeded', SEED_VERSION);
      changed = true;
    }

    const firstSuccess = !hydrated;
    hydrated = true;
    if (localStorage.getItem('flux_sample_data') === '1') { SampleBadge.clear(); changed = true; }
    retryDelay = 15000;
    // Refresh pages on data change OR on offline→online recovery, so widgets
    // that rendered an empty state while the backend was down repopulate.
    if (changed || firstSuccess) {
      window.dispatchEvent(new CustomEvent('flux:data-updated', { detail: window.FluxData }));
    }
  }

  function hydrate() {
    return fetchAll().then(apply).catch((err) => {
      // Backend offline → keep existing localStorage / seed.js defaults and
      // retry with backoff so the page self-heals when the backend comes up.
      console.warn('[FLUX] DB hydration failed (backend offline?), retrying in ' +
                   Math.round(retryDelay / 1000) + 's:', err.message);
      clearTimeout(retryTimer);
      retryTimer = setTimeout(hydrate, retryDelay);
      retryDelay = Math.min(retryDelay * 2, 120000);
    });
  }

  // Pages can force a re-hydration (e.g. when a direct fetch succeeds again
  // after being offline): window.FluxHydrate.refresh()
  window.FluxHydrate = {
    refresh() { clearTimeout(retryTimer); return hydrate(); },
    get hydrated() { return hydrated; },
  };

  const prefetched = takePrefetched();
  if (prefetched) apply(prefetched);
  else hydrate();
})();
