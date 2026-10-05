/**
 * FLUX Market Data Engine — Sweep & Render Cycle v4
 * 
 * Logic:
 * 1. Store all market data in a central 'marketPool' object.
 * 2. On Load: Fetch all categories once (staggered) to fill everything.
 * 3. Loop: Continuously sweep through all asset classes, updating the pool.
 * 4. Render: Update the UI tables and cards at the end of each category fetch.
 * 5. Global: The ticker and summary strip update after a full cycle is complete.
 */

// Same API resolution as js/flux-data.js (marketplace doesn't load the
// hydration layer): FLUX_CONFIG override → :8000 on localhost → same-origin.
const API_BASE = ((window.FLUX_CONFIG && window.FLUX_CONFIG.apiBase) ||
  (['localhost', '127.0.0.1', ''].includes(location.hostname)
    ? 'http://' + (location.hostname || 'localhost') + ':8000'
    : location.origin)).replace(/\/$/, '');
window.FLUX_API = window.FLUX_API || API_BASE;
const CATEGORIES = ["crypto", "stocks"];
const SWEEP_INTERVAL_MS = 30000; // Time between full cycles
const STAGGER_MS = 4000;        // Time between category fetches within a cycle

// Debug logging gate — set window.FLUX_DEBUG = true (or ?debug in URL) to enable.
const DEBUG = (typeof window !== "undefined") &&
  (window.FLUX_DEBUG === true || /[?&]debug\b/.test(location.search));
function dlog(...args) { if (DEBUG) console.log(...args); }

// ── XSS-safe rendering helpers ────────────────────────────────────────────────
// All external strings (news titles, sources, asset names from APIs) MUST pass
// through esc() before being placed in innerHTML. Exposed for the inline page
// script and any other module.
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}
// Only allow http(s) links through — blocks javascript:, data:, etc.
function safeUrl(u) {
  try {
    const url = new URL(u, location.href);
    return (url.protocol === "http:" || url.protocol === "https:") ? url.href : "#";
  } catch (_) {
    return "#";
  }
}
window.fluxEsc = esc;
window.fluxSafeUrl = safeUrl;

// ── Live connection state ─────────────────────────────────────────────────────
// Tracks whether the last sweep reached the backend so the UI can show honest
// "live / stale / offline" status instead of a perpetually-green dot.
const marketState = {
  online: null,          // null=unknown, true=reachable, false=offline
  lastUpdated: null,     // Date of last successful category fetch
  consecutiveFailures: 0,
};
window.fluxMarketState = marketState;

function setConnState(ok) {
  if (ok) {
    marketState.online = true;
    marketState.lastUpdated = new Date();
    marketState.consecutiveFailures = 0;
  } else {
    marketState.consecutiveFailures += 1;
    // Only flip to "offline" after 2 straight misses to avoid flicker on a single blip.
    if (marketState.consecutiveFailures >= 2) marketState.online = false;
  }
  if (window.renderConnState) window.renderConnState();
}

let marketPool = {
  crypto: [],
  stocks: []
};
// Expose to inline scripts in marketplace.html (grid icon lookup, drawer AI prompt, etc.)
window.marketPool = marketPool;

const prevPrices = {};

// Maps for Top Glass Cards per category
const CARD_MAPS = {
  crypto: { "BINANCE:BTCUSDT": "btc", "BINANCE:ETHUSDT": "eth", "BINANCE:SOLUSDT": "sol" },
  stocks: { "NVDA": "nvda", "AAPL": "aapl", "MSFT": "msft" }
};

function fmtPrice(val, category) {
  if (!val || isNaN(val)) return "—";
  const n = parseFloat(val);
  if (category === "forex") return n.toFixed(4);
  if (n >= 1000) return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (n >= 1)    return "$" + n.toFixed(2);
  return "$" + String(Number(n.toPrecision(4)));
}

function renderSparkline(prices) {
  if (!prices || prices.length < 2) return `<svg width="64" height="24"></svg>`;
  const w = 64, h = 24, pad = 2;
  const min = Math.min(...prices), max = Math.max(...prices);
  const range = max - min || 1;
  const step = (w - pad * 2) / (prices.length - 1);
  const pts = prices.map((p, i) => {
    const x = pad + i * step;
    const y = h - pad - ((p - min) / range) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const isUp = prices[prices.length - 1] >= prices[0];
  const color = isUp ? "#00ff9d" : "#ff4d4d";
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="display:block"><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" opacity="0.85"/></svg>`;
}

function fmtChange(pct) {
  const n = parseFloat(pct || 0);
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

function changeClass(pct) {
  return parseFloat(pct || 0) >= 0 ? "positive" : "negative";
}

function fmtAbrv(val) {
  if (!val || isNaN(val)) return "—";
  const n = parseFloat(val);
  if (n >= 1e12) return "$" + (n / 1e12).toFixed(2) + "T";
  if (n >= 1e9)  return "$" + (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6)  return "$" + (n / 1e6).toFixed(1) + "M";
  return "$" + n.toLocaleString();
}

function flashCell(td, isUp) {
  if(!td) return;
  const color = isUp ? "rgba(0, 255, 157, 0.2)" : "rgba(255, 77, 77, 0.2)";
  td.animate([
    { background: color, color: isUp ? "#00ff9d" : "#ff4d4d", offset: 0 },
    { background: "transparent", color: "inherit", offset: 1 }
  ], {
    duration: 800,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)'
  });
}

// ── UI Renderers ─────────────────────────────────────────────────────────────

function updateGlassCards(category, assets) {
  const map = CARD_MAPS[category];
  if (!map) return;
  assets.forEach(asset => {
    const key =  map[asset.symbol] || map[asset.name] || map[asset.short];
    if (key) {
      const cardEl = document.getElementById(`card-${key}`);
      const priceEl = document.getElementById(`${key}-price`);
      const changeEl = document.getElementById(`${key}-change`);

      // Update Icon if available
      if (cardEl && asset.icon) {
        const header = cardEl.querySelector('.asset-header');
        if (header) {
          const oldIcon = header.querySelector('svg, img');
          if (oldIcon && oldIcon.tagName !== 'IMG') {
            const img = document.createElement('img');
            img.src = asset.icon;
            img.width = 32;
            img.height = 32;
            img.style.borderRadius = "8px";
            header.replaceChild(img, oldIcon);
          }
        }
      }

      if (priceEl) {
        const newPrice = parseFloat(asset.price || 0);
        const oldPrice = prevPrices[`card-${key}`] || newPrice;
        priceEl.textContent = fmtPrice(newPrice, category);
        if (newPrice !== oldPrice && newPrice !== 0) flashCell(priceEl.parentElement, newPrice >= oldPrice);
        prevPrices[`card-${key}`] = newPrice;
      }
      if (changeEl) {
        changeEl.textContent = fmtChange(asset.change_pct);
        changeEl.className = `asset-change ${changeClass(asset.change_pct)}`;
      }
    }
  });
}

function renderTicker() {
  // Price ticker removed — element no longer exists in DOM.
  return;
}

// ── Row Builders ─────────────────────────────────────────────────────────────

function buildRow(asset, category, index) {
  const cc = changeClass(asset.change_pct);
  const tr = document.createElement("tr");
  tr.dataset.symbol = asset.symbol;
  tr.dataset.name = asset.name || asset.symbol;
  tr.dataset.sub = asset.sub || asset.symbol;
  tr.dataset.category = category;
  // Numeric fields stashed for client-side sorting / filtering.
  tr.dataset.price = String(parseFloat(asset.price || 0));
  tr.dataset.change = String(parseFloat(asset.change_pct || 0));
  tr.dataset.mcap = String(parseFloat(asset.market_cap || 0));
  tr.dataset.sector = asset.sector || "";
  tr.setAttribute("tabindex", "0");
  tr.setAttribute("role", "button");
  tr.setAttribute("aria-label", `${asset.name || asset.symbol} details`);

  const price = parseFloat(asset.price || 0);
  const priceDisplay = price > 0 ? fmtPrice(price, category) : "—";
  const nm = esc(asset.name);
  const sub = esc(asset.sub || asset.symbol);

  let cols = "";
  if (category === "crypto") {
    const cap = asset.market_cap ? fmtAbrv(asset.market_cap) : "—";
    const icon = asset.icon ? `<img src="${safeUrl(asset.icon)}" width="24" height="24" alt="" style="border-radius:4px; margin-right:12px;">` : "";
    cols = `
    <td style="color:var(--text-secondary)">${index + 1}</td>
    <td><div style="display:flex;align-items:center;">${icon}<div><div class="tbl-name">${nm}</div><div class="tbl-sub">${esc(asset.sub)}</div></div></div></td>
    <td class="price-cell" data-sym="${esc(asset.symbol)}">${priceDisplay}</td>
    <td class="asset-change ${cc}">${fmtChange(asset.change_pct)}</td>
    <td style="font-size:11px;">${cap}</td>
    <td>${renderSparkline(asset.sparkline_7d)}</td>`;
  } else { // stocks
    const icon = asset.icon ? `<img src="${safeUrl(asset.icon)}" width="24" height="24" alt="" style="border-radius:4px; margin-right:12px;">` : "";
    cols = `
    <td style="color:var(--text-secondary)">${index + 1}</td>
    <td><div style="display:flex;align-items:center;">${icon}<div><div class="tbl-name">${nm}</div><div class="tbl-sub">${esc(asset.symbol)}</div></div></div></td>
    <td class="price-cell" data-sym="${esc(asset.symbol)}">${priceDisplay}</td>
    <td class="asset-change ${cc}">${fmtChange(asset.change_pct)}</td>
    <td>${esc(asset.sector || "—")}</td>
    <td><span style="font-size:11px;color:var(--text-secondary)">${sub}</span></td>`;
  }

  tr.innerHTML = cols + `
    <td><div style="display:flex;gap:8px;align-items:center;">
        <button class="watchlist-btn" aria-label="Toggle watchlist" title="Add to watchlist" style="padding:8px;display:flex;align-items:center;justify-content:center;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></button>
        <button class="explore-btn" onclick="openIntel(this)" aria-label="Open ${nm} intel" style="background:none;border:1px solid rgba(255,255,255,0.1);border-radius:6px;padding:5px 10px;color:var(--text-secondary);font-size:11px;cursor:pointer;">Explore</button>
      </div></td>`;
  return tr;
}

// ── Core Sweep Engine ────────────────────────────────────────────────────────

async function fetchCategory(category) {
  try {
    const res = await fetch(`${API_BASE}/market/quotes/${category}`);
    if (!res.ok) { setConnState(false); return; }
    const data = await res.json();
    setConnState(true);
    const assets = (data.assets || []).map(a => ({...a, category}));
    
    // Update Global Pool
    marketPool[category] = assets;

    // Refresh the UI for this category specifically
    const tbody = document.querySelector(`#cat-${category} tbody`);
    if (tbody) {
      if (!tbody.dataset.live || tbody.children.length === 0) {
        tbody.innerHTML = "";
        assets.forEach((a, i) => tbody.appendChild(buildRow(a, category, i)));
        tbody.dataset.live = "1";
      } else {
        // Cascade animation: update rows slowly one by one
        assets.forEach((asset, idx) => {
          setTimeout(() => {
            const sym = asset.symbol;
            const cell = tbody.querySelector(`[data-sym="${sym}"]`);
            if (!cell) return;
            const newPrice = parseFloat(asset.price || 0);
            const oldPrice = prevPrices[sym];
            const priceStr = fmtPrice(newPrice, category);
            if (priceStr !== cell.textContent) {
              cell.textContent = priceStr;
              flashCell(cell, oldPrice !== undefined ? newPrice >= oldPrice : true);
              const changeTd = cell.nextElementSibling;
              if (changeTd) {
                changeTd.textContent = fmtChange(asset.change_pct);
                changeTd.className = `asset-change ${changeClass(asset.change_pct)}`;
              }
            }
            prevPrices[sym] = newPrice;
          }, idx * 100); // 100ms delay per row for 'slow changes' effect
        });
      }
      
      // Global reactive hooks — always update grid for all panels
      setTimeout(() => {
        if (window.injectAlertIcons) window.injectAlertIcons();
        if (window.updateWatchlistGrid) {
           const panel = tbody.closest('.cat-panel');
           if (panel) window.updateWatchlistGrid(panel);
        }
        // Re-apply screener search/sort/filter after rows are (re)built.
        if (window.fluxApplyScreener) window.fluxApplyScreener(category);
      }, assets.length * 100 + 50);
    }
    
    updateGlassCards(category, assets);
  } catch (err) {
    setConnState(false);
    dlog(`[FLUX] Sweep failed for ${category}:`, err.message);
  }
}

async function updateMarketOverview() {
  const cards = document.querySelectorAll(".market-strip .strip-card");
  if (cards.length < 4) return;

  // Source of truth: backend /market/summary returns 4 deterministic highlights
  // (top-2 crypto by mcap + top-2 stock movers). No random shuffling.
  let highlights = [];
  try {
    const res = await fetch(`${API_BASE}/market/summary`);
    if (res.ok) {
      const data = await res.json();
      highlights = data.highlights || [];
    }
  } catch (err) {
    console.warn("[FLUX] /market/summary failed, using pool fallback:", err.message);
  }

  // Fallback: derive locally if the endpoint is unavailable (still deterministic)
  if (highlights.length < 4) {
    const stocks = marketPool.stocks.slice().sort((a,b) => Math.abs(b.change_pct||0) - Math.abs(a.change_pct||0)).slice(0, 2);
    const crypto = marketPool.crypto.slice().sort((a,b) => (b.market_cap||0) - (a.market_cap||0)).slice(0, 4 - stocks.length);
    const local = [
      ...crypto.map(a => ({...a, category: 'crypto'})),
      ...stocks.map(a => ({...a, category: 'stocks'})),
    ];
    if (local.length > highlights.length) highlights = local;
  }

  // Cards with no asset to show are hidden rather than left on "Loading...".
  cards.forEach((card, i) => { card.style.display = highlights[i] ? '' : 'none'; });

  highlights.forEach((asset, i) => {
    const card = cards[i];
    if (!card || !asset) return;
    const pct = parseFloat(asset.change_pct || 0);
    card.querySelector(".strip-label").textContent = `${asset.name || asset.symbol}`;
    card.querySelector(".strip-value").textContent = fmtPrice(asset.price, asset.category || (i < 2 ? 'crypto' : 'stocks'));
    const chEl = card.querySelector(".strip-change");
    chEl.textContent = `${pct >= 0 ? "↑" : "↓"} ${Math.abs(pct).toFixed(2)}%`;
    chEl.className = `strip-change ${pct >= 0 ? "up" : "dn"}`;
  });
}

async function loadInitialLayout() {
  dlog("[FLUX] Injecting initial real values...");
  // Fetch everything once at start to fill the tables with real data
  await Promise.all(CATEGORIES.map(cat => fetchCategory(cat)));
  renderTicker();
  await updateMarketOverview();
}

let _sweepTimer = null;

async function runUpdateCycle() {
  // Skip the network sweep entirely while the tab is hidden — saves API quota
  // and CPU. The visibilitychange handler reschedules a sweep on return.
  if (document.hidden) {
    _sweepTimer = setTimeout(runUpdateCycle, SWEEP_INTERVAL_MS);
    return;
  }
  dlog("[FLUX] Starting Market Sweep...");

  // Sweep through all categories with a stagger
  for (let i = 0; i < CATEGORIES.length; i++) {
    await fetchCategory(CATEGORIES[i]);
    if (i < CATEGORIES.length - 1) await new Promise(r => setTimeout(r, STAGGER_MS));
  }

  // End of cycle renders
  renderTicker();
  await updateMarketOverview();
  if (window.checkPriceAlerts) window.checkPriceAlerts();

  dlog("[FLUX] Sweep Complete. Waiting for next cycle...");
  _sweepTimer = setTimeout(runUpdateCycle, SWEEP_INTERVAL_MS);
}

// On returning to the tab, refresh immediately if data is stale (>1 sweep old).
document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  const stale = !marketState.lastUpdated ||
    (Date.now() - marketState.lastUpdated.getTime() > SWEEP_INTERVAL_MS);
  if (stale) {
    clearTimeout(_sweepTimer);
    runUpdateCycle();
  }
});

function setupInteractions() {
  // Use event delegation to support both static HTML assets and dynamically loaded ones
  document.addEventListener("click", (e) => {
    // 1. Grid Explore Button
    const exploreBtn = e.target.closest(".btn-explore");
    if (exploreBtn) {
      e.stopPropagation();
      if (window.openIntel) window.openIntel(exploreBtn);
      return;
    }

    // 2. Table Explore Button
    const tableExploreBtn = e.target.closest(".explore-btn");
    if (tableExploreBtn) {
      e.stopPropagation();
      if (window.openIntel) window.openIntel(tableExploreBtn);
      return;
    }

    // 3. Grid Eye Button (Watch)
    const watchIcon = e.target.closest(".watch-icon-btn");
    if (watchIcon) {
      e.stopPropagation();
      watchIcon.classList.toggle("active");
      const card = watchIcon.closest(".asset-card");
      if (card) {
        const symbol = card.dataset.symbol;
        const isWatching = watchIcon.classList.contains("active");
        dlog(`[FLUX] ${isWatching ? "Watching" : "Unwatched"} ${symbol}`);
      }
      return;
    }

    // 4. Asset Card Click (Drawer)
    const card = e.target.closest(".asset-card");
    if (card && !e.target.closest(".asset-card-footer") && !e.target.closest(".market-tabs")) {
      if (window.openIntel) window.openIntel(card.dataset.symbol);
      return;
    }
  });

  // Keyboard activation for focusable table rows (Enter / Space → open intel).
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const row = e.target.closest && e.target.closest(".market-table tbody tr");
    if (row && row.dataset.symbol && !e.target.closest("button")) {
      e.preventDefault();
      if (window.openIntel) window.openIntel(row.dataset.symbol);
    }
  });
}

// ── News Engine ──────────────────────────────────────────────────────────────

/**
 * Fetch headlines from the backend and populate the scrolling ticker.
 * The content string is doubled so the CSS `translateX(-50%)` loop is seamless.
 */
async function fetchAndPopulateTicker() {
  const el = document.getElementById("tickerContent");
  if (!el) return;

  try {
    const res  = await fetch(`${API_BASE}/market/news`);
    if (!res.ok) return;
    const data = await res.json();
    const articles = data.articles || [];
    if (!articles.length) return;

    // Join titles with a teal dot separator. External strings are escaped to
    // prevent HTML/script injection from the news feed.
    const single = articles
      .map(a => `${a.source ? `<span style="color:var(--accent-teal);opacity:.7;">[${esc(a.source)}]</span> ` : ""}${esc(a.title)}`)
      .join("&nbsp;&nbsp;·&nbsp;&nbsp;");

    // Double the content — CSS translates -50% so the loop is invisible
    el.innerHTML = single + "&nbsp;&nbsp;·&nbsp;&nbsp;" + single;
  } catch (err) {
    dlog("[FLUX] News ticker fetch failed:", err.message);
  }
}

/**
 * Fetch per-asset headlines and render them in the intel drawer.
 * Called by openIntel whenever the drawer opens.
 * @param {string} assetName  e.g. "Bitcoin", "Apple Inc."
 */
window.fetchDrawerNews = async function (assetName) {
  const listEl = document.getElementById("drawerNewsList");
  if (!listEl) return;

  listEl.innerHTML = `<div style="font-size:11px;color:rgba(255,255,255,.3);padding:8px 0;font-family:var(--font-mono);">Fetching headlines…</div>`;

  try {
    const q   = encodeURIComponent(assetName);
    const res = await fetch(`${API_BASE}/market/news?q=${q}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data     = await res.json();
    const articles = data.articles || [];

    if (!articles.length) {
      listEl.innerHTML = `<div style="font-size:11px;color:rgba(255,255,255,.25);padding:8px 0;">No recent headlines found.</div>`;
      return;
    }

    listEl.innerHTML = articles.slice(0, 6).map(a => {
      const pub = a.published_at
        ? new Date(a.published_at).toLocaleString("en-IN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
        : "";
      return `
        <a class="drawer-news-item" href="${safeUrl(a.url)}" target="_blank" rel="noopener noreferrer">
          <div class="drawer-news-title">${esc(a.title)}</div>
          <div class="drawer-news-meta">
            <span class="drawer-news-source">${esc(a.source || "—")}</span>
            ${pub ? `<span>${esc(pub)}</span>` : ""}
          </div>
        </a>`;
    }).join("");
  } catch (err) {
    listEl.innerHTML = `<div style="font-size:11px;color:rgba(255,255,255,.25);padding:8px 0;">Headlines unavailable.</div>`;
    dlog("[FLUX] Drawer news fetch failed:", err.message);
  }
};

async function initMarket() {
  // Inject pulsing live dot
    const dot = document.getElementById("live-dot");
    if (dot) {
      dot.style.display = "flex";
    }

  setupInteractions();

  // 1. Instantly populate the grid with 'recent' starting values
  await loadInitialLayout();

  // 2. Populate news ticker (fire-and-forget, don't block grid load)
  fetchAndPopulateTicker();
  setInterval(fetchAndPopulateTicker, 5 * 60 * 1000); // refresh every 5 min

  // 3. Start the sweep loop after one full interval to avoid a double-sweep on load
  setTimeout(runUpdateCycle, SWEEP_INTERVAL_MS);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initMarket);
} else {
  initMarket();
}
