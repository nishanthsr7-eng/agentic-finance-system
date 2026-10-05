/* FLUX — Marketplace feature module
 * ==================================
 * Real, backend-wired logic for the marketplace page. Loads after market-live.js.
 * Owns:
 *   • Identity (header name/avatar from the session)
 *   • Live connection-state indicator (live / stale / offline)
 *   • Paper-trading wallet state + spot buy/sell order ticket (POST /db/trades)
 *   • Persistent watchlist (GET/POST/DELETE /db/watchlist)
 *   • Persistent price alerts with above/below direction + management UI
 *   • Screener search, column sort, sector/gainers/losers filters
 *   • Modal accessibility: Esc to close, backdrop click, focus handling
 *
 * Anything fake (leverage, fabricated AI, hardcoded fills) was removed from the
 * page; this module is the source of truth for those behaviours now.
 */
(function fluxMarketplace() {
  "use strict";

  const API = () => window.FLUX_API || "";
  const esc = (s) => (window.fluxEsc ? window.fluxEsc(s) : String(s == null ? "" : s));
  const toast = (t, s, type) => window.showToast && window.showToast(t, s, type);
  const $ = (id) => document.getElementById(id);

  // ── Formatting ──────────────────────────────────────────────────────────────
  const fmtUSD = (n) => "$" + Number(n || 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtQty = (n) => Number(n || 0).toLocaleString("en-US", { maximumFractionDigits: 6 });
  function fmtPrice(n) {
    n = Number(n || 0);
    if (n >= 1000) return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (n >= 1) return "$" + n.toFixed(2);
    return "$" + n.toFixed(4);
  }

  // ── Market pool lookup ────────────────────────────────────────────────────────
  function allAssets() {
    const p = window.marketPool || {};
    return [...(p.crypto || []), ...(p.stocks || [])];
  }
  function findAsset(symbolOrName) {
    if (!symbolOrName) return null;
    const key = String(symbolOrName).trim().toLowerCase();
    return allAssets().find((a) =>
      (a.symbol || "").toLowerCase() === key ||
      (a.sub || "").toLowerCase() === key ||
      (a.name || "").toLowerCase() === key
    ) || null;
  }

  // ── Authenticated JSON helpers (token attached by flux-data.js fetch wrapper) ──
  async function apiGet(path) {
    const r = await fetch(`${API()}${path}`);
    if (!r.ok) throw new Error(`${path} → ${r.status}`);
    return r.json();
  }
  async function apiSend(method, path, body) {
    const r = await fetch(`${API()}${path}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `${path} → ${r.status}`);
    return data;
  }

  // ── Identity ──────────────────────────────────────────────────────────────────
  async function loadIdentity() {
    const nameEl = $("hdrUserName"), avEl = $("hdrAvatar"), statusEl = $("hdrUserStatus");
    try {
      const { user } = await apiGet("/auth/me");
      if (!user) return;
      const name = user.name || "Trader";
      if (nameEl) nameEl.textContent = name;
      if (statusEl) statusEl.textContent = user.plan || "";
      if (avEl) {
        const initials = name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
        avEl.textContent = initials || "U";
      }
    } catch (_) {
      // Unauthenticated → flux-data.js will route to login on the /db/* calls.
      if (nameEl) nameEl.textContent = "Guest";
    }
  }

  // ── Connection-state indicator ──────────────────────────────────────────────
  window.renderConnState = function renderConnState() {
    const st = window.fluxMarketState || {};
    const dot = $("live-dot"), label = $("live-dot-label");
    const banner = $("connBanner"), bannerText = $("connBannerText");
    if (!dot || !label) return;

    let cls = "", text = "LIVE", showBanner = false, bannerMsg = "";
    if (st.online === false) {
      cls = "state-offline"; text = "OFFLINE"; showBanner = true;
      bannerMsg = "Live market data is unavailable — showing the last known values. Retrying automatically.";
    } else if (st.online === true) {
      const ageMs = st.lastUpdated ? Date.now() - st.lastUpdated.getTime() : 0;
      if (ageMs > 90000) {
        cls = "state-stale"; text = "STALE"; showBanner = true;
        bannerMsg = "Market data hasn't refreshed recently — values may be delayed.";
      } else {
        text = "LIVE";
      }
    } else {
      text = "CONNECTING";
    }
    dot.className = cls;
    label.textContent = st.lastUpdated && st.online !== false
      ? `${text} · ${st.lastUpdated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
      : text;
    if (banner) {
      banner.style.display = showBanner ? "flex" : "none";
      if (bannerText && showBanner) bannerText.textContent = bannerMsg;
    }
  };
  // Re-evaluate staleness on a slow tick (lastUpdated ages even without events).
  setInterval(() => window.renderConnState && window.renderConnState(), 30000);

  // ════════════════════════════════════════════════════════════════════════════
  //  WALLET  +  ORDER TICKET
  // ════════════════════════════════════════════════════════════════════════════
  let wallet = null;          // last /db/wallet payload
  let order = null;           // current order context {symbol,name,asset_type,price,side,type}
  let confirmArmed = false;   // two-step confirm state

  async function loadWallet() {
    try {
      wallet = await apiGet("/db/wallet");
      renderWallet();
    } catch (e) {
      // 401 handled by flux-data redirect; other errors leave the bar as "—".
    }
    return wallet;
  }

  function renderWallet() {
    if (!wallet) return;
    const set = (id, v, cls) => { const el = $(id); if (el) { el.textContent = v; if (cls !== undefined) el.className = "wc-value " + cls; } };
    set("walletCash", fmtUSD(wallet.cash_usd));
    set("walletHoldings", fmtUSD(wallet.holdings_value_usd));
    set("walletEquity", fmtUSD(wallet.equity_usd));
    const pnl = Number(wallet.unrealized_pnl_usd || 0);
    set("walletPnl", (pnl >= 0 ? "+" : "−") + fmtUSD(Math.abs(pnl)).slice(1), pnl >= 0 ? "up" : "dn");
  }

  function positionFor(symbol) {
    if (!wallet || !symbol) return null;
    const k = symbol.toLowerCase();
    return (wallet.positions || []).find((p) => p.symbol.toLowerCase() === k) || null;
  }

  // Prep the order ticket when the modal opens (called by inline openOrderModal).
  window.fluxPrepOrder = async function (assetName, side) {
    const asset = findAsset(assetName);
    order = {
      symbol: asset ? asset.symbol : (assetName || ""),
      name: asset ? asset.name : (assetName || ""),
      asset_type: asset ? (asset.category === "crypto" ? "crypto" : "stocks") : "stocks",
      price: asset ? Number(asset.price || 0) : 0,
      side: side === "sell" ? "sell" : "buy",
      type: "market",
    };
    confirmArmed = false;
    // Reset form controls
    document.querySelectorAll(".exec-tab-refined").forEach((t) => {
      const isSide = t.dataset.side === order.side;
      t.classList.toggle("active", isSide);
      t.setAttribute("aria-selected", String(isSide));
    });
    document.querySelectorAll('#execForm .percent-btn[data-type]').forEach((b) =>
      b.classList.toggle("active", b.dataset.type === "market"));
    const lf = $("limitPriceField"); if (lf) lf.style.display = "none";
    const err = $("execError"); if (err) err.style.display = "none";
    const sizeInput = $("execSizeInput");
    if (sizeInput) sizeInput.value = "1000";
    await loadWallet();
    updateOrderSummary();
  };

  window.switchExecSide = function (side) {
    if (!order) return;
    order.side = side;
    confirmArmed = false;
    document.querySelectorAll(".exec-tab-refined").forEach((t) => {
      const on = t.dataset.side === side;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", String(on));
    });
    updateOrderSummary();
  };

  window.switchOrderType = function (btn, type) {
    if (!order) return;
    order.type = type;
    btn.parentElement.querySelectorAll(".percent-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const lf = $("limitPriceField");
    if (lf) lf.style.display = type === "limit" ? "block" : "none";
    confirmArmed = false;
    updateOrderSummary();
  };

  // Percent of buying power (BUY) or of current position value (SELL).
  window.setExecPercent = function (p, btn) {
    if (!order) return;
    if (btn) {
      btn.parentElement.querySelectorAll(".percent-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
    }
    const px = orderPrice();
    let usd = 0;
    if (order.side === "buy") {
      usd = (wallet ? wallet.cash_usd : 0) * p;
    } else {
      const pos = positionFor(order.symbol);
      usd = pos ? pos.value_usd * p : 0;
    }
    const sizeInput = $("execSizeInput");
    if (sizeInput) sizeInput.value = usd > 0 ? usd.toFixed(2) : "0";
    updateOrderSummary();
  };

  function orderPrice() {
    if (order && order.type === "limit") {
      const v = parseFloat(($("execLimitPrice") || {}).value);
      if (v > 0) return v;
    }
    // Always re-read the freshest live price from the pool.
    const a = findAsset(order && order.symbol);
    return a ? Number(a.price || 0) : (order ? order.price : 0);
  }

  // Resolve {px, usd, qty} for the current form. For SELL we clamp the quantity
  // to the held position so "MAX" (or any amount ≥ holding value) sells exactly
  // what's held — live-price drift can otherwise push usd/price slightly over.
  function computeOrder() {
    const px = orderPrice();
    const usd = parseFloat(($("execSizeInput") || {}).value) || 0;
    let qty = px > 0 ? usd / px : 0;
    let clamped = false;
    if (order && order.side === "sell") {
      const pos = positionFor(order.symbol);
      const held = pos ? pos.quantity : 0;
      if (qty > held) { qty = held; clamped = true; }
    }
    return { px, usd, qty, clamped };
  }

  function updateOrderSummary() {
    if (!order) return;
    const { px, qty, clamped } = computeOrder();
    const usd = px > 0 ? qty * px : (parseFloat(($("execSizeInput") || {}).value) || 0);
    // SELL capped at the holding: show the capped amount in the box too, unless
    // the user is typing in it right now.
    const sizeBox = $("execSizeInput");
    if (clamped && sizeBox && document.activeElement !== sizeBox) sizeBox.value = usd.toFixed(2);

    const setT = (id, v) => { const el = $(id); if (el) el.textContent = v; };
    setT("execPrice", px > 0 ? fmtPrice(px) : "—");
    setT("execQty", px > 0 ? `${fmtQty(qty)} ${esc(order.symbol.replace(/^BINANCE:/, "").replace(/USDT$/, ""))}` : "—");
    setT("execTotal", fmtUSD(usd));

    // Available context line
    const availEl = $("execAvailLabel");
    if (availEl) {
      if (order.side === "buy") {
        availEl.textContent = wallet ? `Buying power: ${fmtUSD(wallet.cash_usd)}` : "";
      } else {
        const pos = positionFor(order.symbol);
        availEl.textContent = pos ? `Holding: ${fmtQty(pos.quantity)} (${fmtUSD(pos.value_usd)})` : "No position held";
      }
    }

    // Button state
    const btn = $("execConfirmBtn");
    if (btn) {
      btn.className = `confirm-btn ${order.side}`;
      const word = order.side.toUpperCase();
      btn.textContent = confirmArmed ? `CONFIRM ${word} · ${fmtUSD(usd)}` : `REVIEW ${word} ORDER`;
    }
    $("execTotalLabel").textContent = "Estimated Total";
  }

  function validateOrder() {
    const { px, usd, qty } = computeOrder();
    if (!order || !order.symbol) return "No asset selected.";
    if (order.type === "limit" && !(parseFloat(($("execLimitPrice") || {}).value) > 0)) return "Enter a valid limit price.";
    if (!(px > 0)) return "Live price unavailable — try again in a moment.";
    if (!(qty > 0)) return "Enter an amount greater than zero.";
    if (order.side === "buy" && wallet && usd > wallet.cash_usd + 0.005) {
      return `Amount exceeds buying power (${fmtUSD(wallet.cash_usd)}).`;
    }
    if (order.side === "sell") {
      const pos = positionFor(order.symbol);
      if (!pos || pos.quantity <= 0) return "You don't hold this asset.";
    }
    return null;
  }

  function showOrderError(msg) {
    const err = $("execError");
    if (err) { err.textContent = msg; err.style.display = msg ? "block" : "none"; }
  }

  async function submitOrder() {
    const { px, qty } = computeOrder();
    const btn = $("execConfirmBtn");
    if (btn) { btn.disabled = true; btn.textContent = "PLACING…"; btn.style.opacity = "0.7"; }
    try {
      const res = await apiSend("POST", "/db/trades", {
        symbol: order.symbol,
        name: order.name,
        side: order.side.toUpperCase(),
        price: px,
        quantity: qty,
        asset_type: order.asset_type,
      });
      wallet = res.wallet;
      renderWallet();
      // Success receipt
      $("execForm").style.display = "none";
      $("execSuccess").style.display = "block";
      $("successHeading").textContent = `${order.side.toUpperCase()} FILLED`;
      $("successRef").textContent = "FLX-" + String(res.trade.id).padStart(6, "0");
      $("successAsset").textContent = order.name;
      $("successFilled").textContent = `${fmtQty(res.trade.quantity)} @ ${fmtPrice(res.trade.price)} (server quote)`;
      $("successValue").textContent = fmtUSD(res.trade.amount);
      $("successCash").textContent = fmtUSD(res.wallet.cash_usd);
      toast("Order Filled", `${order.side.toUpperCase()} ${fmtQty(res.trade.quantity)} ${order.name} for ${fmtUSD(res.trade.amount)}.`, "price");
    } catch (e) {
      showOrderError(e.message || "Order failed.");
      confirmArmed = false;
      updateOrderSummary();
    } finally {
      if (btn) { btn.disabled = false; btn.style.opacity = "1"; }
    }
  }

  function wireOrderTicket() {
    const sizeInput = $("execSizeInput");
    if (sizeInput) sizeInput.addEventListener("input", () => { confirmArmed = false; showOrderError(""); updateOrderSummary(); });
    if (sizeInput) sizeInput.addEventListener("blur", updateOrderSummary);
    const limitInput = $("execLimitPrice");
    if (limitInput) limitInput.addEventListener("input", () => { confirmArmed = false; updateOrderSummary(); });
    const btn = $("execConfirmBtn");
    if (btn) btn.addEventListener("click", () => {
      const err = validateOrder();
      if (err) { showOrderError(err); confirmArmed = false; updateOrderSummary(); return; }
      showOrderError("");
      if (!confirmArmed) {           // first click: arm confirmation
        confirmArmed = true;
        updateOrderSummary();
        setTimeout(() => { if (confirmArmed) { confirmArmed = false; updateOrderSummary(); } }, 5000);
        return;
      }
      submitOrder();                 // second click: execute
    });
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  WATCHLIST  (persisted)
  // ════════════════════════════════════════════════════════════════════════════
  let backendWatch = new Set();   // symbols known to be persisted

  async function loadWatchlist() {
    try {
      const { watchlist } = await apiGet("/db/watchlist");
      backendWatch = new Set((watchlist || []).map((w) => w.symbol));
      // Make the backend the authoritative source for the in-memory grid state
      // used by the inline updateWatchlistGrid — but only if the user has saved
      // selections; an empty backend leaves the top-3 default seed untouched.
      if (watchlist && watchlist.length) {
        window.userWatchedByCategory = { crypto: new Set(), stocks: new Set() };
        window.userWatchedSymbols = new Set();
        watchlist.forEach((w) => {
          const cat = w.category === "stocks" ? "stocks" : "crypto";
          window.userWatchedByCategory[cat].add(w.symbol);
          window.userWatchedSymbols.add(w.symbol);
        });
        document.querySelectorAll(".cat-panel").forEach((p) => window.updateWatchlistGrid && window.updateWatchlistGrid(p));
      }
    } catch (_) { /* unauthenticated → handled elsewhere */ }
  }

  // After any watchlist-btn click, reconcile the full in-memory set with the backend.
  function wireWatchlistSync() {
    document.addEventListener("click", (e) => {
      if (!e.target.closest(".market-table .watchlist-btn")) return;
      setTimeout(syncWatchlist, 0);   // run after the inline handler mutates state
    });
  }
  async function syncWatchlist() {
    const want = new Map();   // symbol → {category, name}
    ["crypto", "stocks"].forEach((cat) => {
      (window.userWatchedByCategory?.[cat] || new Set()).forEach((sym) => {
        const a = findAsset(sym);
        want.set(sym, { category: cat, name: a ? a.name : sym });
      });
    });
    // Additions
    for (const [sym, meta] of want) {
      if (!backendWatch.has(sym)) {
        try { await apiSend("POST", "/db/watchlist", { symbol: sym, category: meta.category, name: meta.name }); backendWatch.add(sym); } catch (_) {}
      }
    }
    // Removals
    for (const sym of [...backendWatch]) {
      if (!want.has(sym)) {
        try { await apiSend("DELETE", `/db/watchlist/${encodeURIComponent(sym)}`); backendWatch.delete(sym); } catch (_) {}
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  ALERTS  (persisted, above/below)
  // ════════════════════════════════════════════════════════════════════════════
  let alerts = [];                 // active alerts from backend
  let alertDir = "above";          // current form direction
  let alertCtx = { name: "", symbol: "" };

  async function loadAlerts() {
    try {
      const { alerts: rows } = await apiGet("/db/alerts");
      alerts = (rows || []).filter((a) => a.active);
      renderAlertList();
      updateAlertBadge();
    } catch (_) {}
  }

  function updateAlertBadge() {
    const badge = $("alertCountBadge");
    if (!badge) return;
    const n = alerts.length;
    badge.textContent = String(n);
    badge.style.display = n ? "flex" : "none";
  }

  window.fluxOpenAlertModal = function (show, assetName, symbol) {
    const ov = $("alertOverlay");
    if (!ov) return;
    if (!show) { closeModal(ov); return; }
    alertCtx = { name: assetName || "", symbol: symbol || (findAsset(assetName)?.symbol) || "" };
    const form = $("alertCreateForm");
    const hasAsset = !!alertCtx.symbol;
    if (form) form.style.display = hasAsset ? "block" : "none";   // header bell w/o asset → manage only
    if (hasAsset) {
      $("alertAssetName").textContent = alertCtx.name || alertCtx.symbol;
      const a = findAsset(alertCtx.symbol);
      $("alertAssetPrice").textContent = a ? `Live: ${fmtPrice(a.price)}` : "";
      $("alertTargetPrice").value = a ? Number(a.price || 0).toFixed(a.price >= 1 ? 2 : 4) : "";
      alertDir = "above";
      document.querySelectorAll("#alertCreateForm .alert-type-card").forEach((c) =>
        c.classList.toggle("selected", c.dataset.dir === "above"));
      const ae = $("alertError"); if (ae) ae.style.display = "none";
    }
    loadAlerts();
    openModal(ov);
  };

  window.selectAlertDir = function (el, dir) {
    alertDir = dir;
    document.querySelectorAll("#alertCreateForm .alert-type-card").forEach((c) => c.classList.remove("selected"));
    el.classList.add("selected");
  };

  window.saveAlert = async function () {
    const target = parseFloat(($("alertTargetPrice") || {}).value);
    const ae = $("alertError");
    if (!(target > 0)) { if (ae) { ae.textContent = "Enter a valid target price."; ae.style.display = "block"; } return; }
    if (!alertCtx.symbol) { if (ae) { ae.textContent = "No asset selected."; ae.style.display = "block"; } return; }
    try {
      const { alerts: rows } = await apiSend("POST", "/db/alerts", {
        symbol: alertCtx.symbol, name: alertCtx.name || alertCtx.symbol,
        direction: alertDir, target_price: target,
      });
      alerts = (rows || []).filter((a) => a.active);
      renderAlertList();
      updateAlertBadge();
      if (ae) ae.style.display = "none";
      toast("Alert Set", `Notify when ${alertCtx.name || alertCtx.symbol} ${alertDir === "above" ? "rises above" : "falls below"} ${fmtPrice(target)}.`);
    } catch (e) {
      if (ae) { ae.textContent = e.message || "Could not save alert."; ae.style.display = "block"; }
    }
  };

  window.fluxDeleteAlert = async function (id) {
    try {
      const { alerts: rows } = await apiSend("DELETE", `/db/alerts/${id}`);
      alerts = (rows || []).filter((a) => a.active);
      renderAlertList();
      updateAlertBadge();
    } catch (_) {}
  };

  function renderAlertList() {
    const list = $("alertList"), count = $("alertListCount");
    if (count) count.textContent = String(alerts.length);
    if (!list) return;
    if (!alerts.length) {
      list.innerHTML = `<div style="font-size:11px; color:rgba(255,255,255,0.3); padding:8px 0;">No active alerts.</div>`;
      return;
    }
    list.innerHTML = alerts.map((a) => {
      const tp = Number(a.target_price);
      const arrow = a.direction === "above" ? "▲" : "▼";
      const col = a.direction === "above" ? "#00e5a0" : "#ff4d6d";
      return `
        <div style="display:flex; align-items:center; justify-content:space-between; gap:10px; padding:10px 12px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:10px;">
          <div>
            <div style="font-size:12px; font-weight:600; color:#fff;">${esc(a.name || a.symbol)}</div>
            <div style="font-size:10px; color:var(--text-secondary); font-family:var(--font-mono);">
              <span style="color:${col};">${arrow} ${a.direction}</span> ${fmtPrice(tp)}
            </div>
          </div>
          <button type="button" onclick="fluxDeleteAlert(${a.id})" aria-label="Delete alert"
            style="background:rgba(255,77,109,0.08); border:1px solid rgba(255,77,109,0.2); color:#ff8da3; border-radius:8px; padding:6px 10px; font-size:10px; cursor:pointer;">Remove</button>
        </div>`;
    }).join("");
  }

  // Evaluate alerts against the live pool after every sweep (called by market-live).
  const firedThisSession = new Set();
  window.checkPriceAlerts = function () {
    if (!alerts.length) return;
    alerts.forEach((a) => {
      if (firedThisSession.has(a.id)) return;
      const asset = findAsset(a.symbol);
      if (!asset) return;
      const price = Number(asset.price || 0);
      if (!price) return;
      const tp = Number(a.target_price);
      const crossed = a.direction === "above" ? price >= tp : price <= tp;
      if (crossed) {
        firedThisSession.add(a.id);
        toast(`${a.name || a.symbol} Alert`, `Price ${fmtPrice(price)} ${a.direction === "above" ? "rose above" : "fell below"} your ${fmtPrice(tp)} target.`, "price");
        apiSend("POST", `/db/alerts/${a.id}/triggered`).then(() => loadAlerts()).catch(() => {});
      }
    });
  };

  // ════════════════════════════════════════════════════════════════════════════
  //  SCREENER: search / sort / filter
  // ════════════════════════════════════════════════════════════════════════════
  const screenerState = {
    crypto: { q: "", filter: "all", sort: null, dir: -1 },
    stocks: { q: "", filter: "all", sort: null, dir: -1 },
  };

  function panelFor(cat) { return document.getElementById(`cat-${cat}`); }

  window.fluxApplyScreener = function (cat) {
    const panel = panelFor(cat);
    if (!panel) return;
    const st = screenerState[cat];
    const tbody = panel.querySelector(".market-table tbody");
    if (!tbody) return;
    let rows = Array.from(tbody.querySelectorAll("tr"));

    // Filter (search + chip)
    let visible = 0;
    rows.forEach((r) => {
      const name = (r.dataset.name || "").toLowerCase();
      const sym = (r.dataset.symbol || "").toLowerCase();
      const sub = (r.dataset.sub || "").toLowerCase();
      const change = parseFloat(r.dataset.change || "0");
      const sector = (r.dataset.sector || "").toLowerCase();
      let ok = true;
      if (st.q) ok = name.includes(st.q) || sym.includes(st.q) || sub.includes(st.q);
      if (ok && st.filter !== "all") {
        if (st.filter === "gainers") ok = change >= 0;
        else if (st.filter === "losers") ok = change < 0;
        else ok = sector === st.filter;   // sector slug
      }
      r.style.display = ok ? "" : "none";
      if (ok) visible++;
    });

    // Sort (only the visible set, reorder DOM)
    if (st.sort) {
      const key = st.sort;
      const sorted = rows.filter((r) => r.style.display !== "none").sort((a, b) => {
        let av, bv;
        if (key === "name" || key === "sector") {
          av = (a.dataset[key === "name" ? "name" : "sector"] || "").toLowerCase();
          bv = (b.dataset[key === "name" ? "name" : "sector"] || "").toLowerCase();
          return av < bv ? st.dir : av > bv ? -st.dir : 0;
        }
        av = parseFloat(a.dataset[key] || "0");
        bv = parseFloat(b.dataset[key] || "0");
        return (av - bv) * st.dir;
      });
      sorted.forEach((r) => tbody.appendChild(r));
    }

    const empty = panel.querySelector(".screener-empty");
    if (empty) empty.style.display = visible ? "none" : "block";
  };

  function buildSectorChips() {
    // Add real sector filter chips for stocks based on live data.
    const group = document.querySelector('.screener-filters[data-cat="stocks"]');
    if (!group || group.dataset.sectorsBuilt) return;
    const sectors = [...new Set((window.marketPool?.stocks || []).map((a) => a.sector).filter(Boolean))].sort();
    if (!sectors.length) return;
    sectors.forEach((sec) => {
      const b = document.createElement("button");
      b.className = "filter-chip";
      b.dataset.filter = sec.toLowerCase();
      b.textContent = sec;
      group.appendChild(b);
    });
    group.dataset.sectorsBuilt = "1";
  }

  function wireScreener() {
    // Search inputs
    document.querySelectorAll(".screener-search").forEach((inp) => {
      inp.addEventListener("input", () => {
        const cat = inp.dataset.cat;
        screenerState[cat].q = inp.value.trim().toLowerCase();
        window.fluxApplyScreener(cat);
      });
    });
    // Filter chips (delegated — sector chips are added later)
    document.querySelectorAll(".screener-filters").forEach((group) => {
      const cat = group.dataset.cat;
      group.addEventListener("click", (e) => {
        const chip = e.target.closest(".filter-chip");
        if (!chip) return;
        group.querySelectorAll(".filter-chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        screenerState[cat].filter = chip.dataset.filter;
        window.fluxApplyScreener(cat);
      });
    });
    // Sortable headers (delegated per table)
    document.querySelectorAll(".cat-panel").forEach((panel) => {
      const cat = panel.id.replace("cat-", "");
      panel.querySelectorAll("th.sortable").forEach((th) => {
        const act = () => {
          const st = screenerState[cat];
          const key = th.dataset.sort;
          if (st.sort === key) st.dir = -st.dir; else { st.sort = key; st.dir = (key === "name" || key === "sector") ? 1 : -1; }
          panel.querySelectorAll("th.sortable").forEach((h) => {
            h.removeAttribute("data-active");
            const a = h.querySelector(".sort-arrow"); if (a) a.remove();
          });
          th.setAttribute("data-active", "1");
          const arrow = document.createElement("span");
          arrow.className = "sort-arrow";
          arrow.textContent = st.dir === 1 ? "▲" : "▼";
          th.appendChild(arrow);
          window.fluxApplyScreener(cat);
        };
        th.addEventListener("click", act);
        th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } });
      });
    });
  }

  // ════════════════════════════════════════════════════════════════════════════
  //  MODAL ACCESSIBILITY
  // ════════════════════════════════════════════════════════════════════════════
  let lastFocus = null;
  function openModal(ov) {
    lastFocus = document.activeElement;
    ov.style.display = "flex";
    const focusable = ov.querySelector('input, button, [tabindex]:not([tabindex="-1"])');
    if (focusable) setTimeout(() => focusable.focus(), 50);
  }
  function closeModal(ov) {
    ov.style.display = "none";
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  function topMostOpen() {
    // Returns the currently-visible overlay/drawer to close on Esc.
    const drawer = $("intelDrawer");
    if (drawer && drawer.classList.contains("active")) return { kind: "drawer" };
    const overlays = ["execOverlay", "chartOverlay", "alertOverlay"]
      .map((id) => $(id)).filter((o) => o && getComputedStyle(o).display !== "none");
    return overlays.length ? { kind: "overlay", el: overlays[overlays.length - 1] } : null;
  }

  function wireModalA11y() {
    // Esc closes the top-most modal/drawer.
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      const top = topMostOpen();
      if (!top) return;
      if (top.kind === "drawer") { if (window.closeIntel) window.closeIntel(); }
      else {
        if (top.el.id === "execOverlay") window.toggleExec(false);
        else if (top.el.id === "chartOverlay") window.toggleChartModal(false);
        else if (top.el.id === "alertOverlay") window.toggleAlertModal(false);
      }
    });
    // Backdrop click closes (click directly on the overlay, not its modal child).
    [["execOverlay", () => window.toggleExec(false)],
     ["chartOverlay", () => window.toggleChartModal(false)],
     ["alertOverlay", () => window.toggleAlertModal(false)]].forEach(([id, close]) => {
      const ov = $(id);
      if (ov) ov.addEventListener("mousedown", (e) => { if (e.target === ov) close(); });
    });
  }

  // ── Header bell + Compare rename ──────────────────────────────────────────────
  function wireHeaderBell() {
    const bell = $("alertBellBtn");
    if (!bell) return;
    const open = () => window.toggleAlertModal(true, "", "");
    bell.addEventListener("click", open);
    bell.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
  }

  // ── Boot ──────────────────────────────────────────────────────────────────────
  function boot() {
    window.renderConnState && window.renderConnState();
    wireOrderTicket();
    wireWatchlistSync();
    wireScreener();
    wireModalA11y();
    wireHeaderBell();

    loadIdentity();
    loadWallet();
    loadWatchlist();
    loadAlerts();

    // The wallet's holdings value / P&L are marked-to-market server-side; the
    // first load can land before the backend price cache is warm (P&L shows 0).
    // Refresh shortly after boot and then periodically so the bar stays live.
    setTimeout(loadWallet, 5000);
    setInterval(() => { if (!document.hidden) loadWallet(); }, 45000);

    // Build sector chips + apply screener once live data arrives.
    let tries = 0;
    const poll = setInterval(() => {
      tries++;
      if ((window.marketPool?.stocks || []).length) {
        buildSectorChips();
        window.fluxApplyScreener("crypto");
        window.fluxApplyScreener("stocks");
        clearInterval(poll);
      }
      if (tries > 40) clearInterval(poll);
    }, 500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
