/* FLUX — Account Setup / Onboarding
 * Lets the user configure the data required when setting up the account:
 *   • Profile      → name, preferred currency        (key: flux_profile)
 *   • Accounts     → bank name, balance, card, expiry (key: flux_accounts)
 *   • Portfolio    → equity/crypto/cash split + goal  (key: flux_portfolio)
 * These are the SAME localStorage keys the dashboard already renders from,
 * so saving + reloading makes every container reflect the user's real data.
 *
 * Loaded AFTER dashboard.js, so its DOMContentLoaded runs last and can
 * override the hardcoded placeholders (name, avatar, card holder, greeting).
 */
(function () {
  const PROFILE_KEY = 'flux_profile';
  const DEMO_EMAIL = 'nishanth@flux.app'; // public demo login (pages/login.html)

  /* ───────────── helpers ───────────── */
  const $ = (id) => document.getElementById(id);
  const parseAmount = (s) => parseInt(String(s).replace(/[^0-9]/g, ''), 10) || 0;
  const fmtINR = (n) => '₹' + Math.max(0, Math.round(n)).toLocaleString('en-IN');

  function initialsFrom(name) {
    const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
    if (!parts.length) return 'FX';
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }

  const readJSON = (key, fallback) => {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; }
    catch (_) { return fallback; }
  };
  const getProfile   = () => readJSON(PROFILE_KEY, null);
  const getAccounts  = () => readJSON('flux_accounts', []);
  const getPortfolio = () => readJSON('flux_portfolio', {});

  /* ───────────── apply persisted data to the live page ───────────── */
  function applyProfile() {
    const profile = getProfile();
    if (!profile || !profile.name) return;
    const initials = profile.initials || initialsFrom(profile.name);

    document.querySelectorAll('.user-info-name').forEach(el => { el.textContent = profile.name; });
    document.querySelectorAll('.avatar').forEach(el => { el.textContent = initials; });

    const holder = $('cardHolder');
    if (holder) holder.textContent = profile.name.toUpperCase();

    const greet = $('greetingHeader');
    if (greet) {
      const first = profile.name.trim().split(/\s+/)[0];
      const hr = new Date().getHours();
      const part = hr < 12 ? 'Good Morning' : hr < 17 ? 'Good Afternoon' : 'Good Evening';
      greet.textContent = `${part}, ${first}`;
    }

    // Apply currency preference quietly (default is INR)
    if (profile.currency === 'USD' && typeof setCurrency === 'function') {
      try {
        document.querySelectorAll('[data-currency]').forEach(i =>
          i.classList.toggle('active', i.dataset.currency === 'USD'));
        setCurrency('USD');
      } catch (_) { /* non-fatal */ }
    }
  }

  // Reflect the ACTIVE account on the credit card (dashboard.js only did this on swap).
  function applyActiveCard() {
    const accounts = getAccounts();
    if (!accounts.length) return;
    const acc = accounts.find(a => a.active) || accounts[0];

    const numEl = $('cardNumber');
    if (numEl && acc.cardNum) {
      numEl.textContent = acc.cardNum;
      numEl.dataset.hidden = acc.cardNum;
      // realNum is intentionally not persisted; show masked form only
      numEl.dataset.real = acc.cardNum;
    }
    const expEl = $('cardExpiry');
    if (expEl) {
      // expiry is stored as masked only; show masked form
      expEl.textContent = '••/••';
      expEl.dataset.real = '••/••';
      expEl.dataset.hidden = '••/••';
    }
  }

  /* ───────────── modal: account rows ───────────── */
  function accountRowHTML(acc = {}) {
    const last4 = (acc.cardNum || '').replace(/[^0-9]/g, '').slice(-4);
    const exp   = acc.realExpiry || '';
    const bal   = acc.val ? parseAmount(acc.val).toLocaleString('en-IN') : '';
    return `
      <div class="setup-acc-row">
        <input class="setup-input setup-acc-name" placeholder="Account name" value="${(acc.name || '').replace(/"/g, '&quot;')}">
        <input class="setup-input setup-acc-bal" placeholder="Balance ₹" inputmode="numeric" value="${bal}">
        <input class="setup-input setup-acc-card" placeholder="Card last 4" inputmode="numeric" maxlength="4" value="${last4}">
        <input class="setup-input setup-acc-exp" placeholder="MM/YY" maxlength="5" value="${exp}">
        <button type="button" class="setup-acc-remove" title="Remove account">&times;</button>
      </div>`;
  }

  function addAccountRow(acc) {
    const rows = $('setupAccountRows');
    if (!rows) return;
    rows.insertAdjacentHTML('beforeend', accountRowHTML(acc));
    rows.lastElementChild.querySelector('.setup-acc-remove')
      .addEventListener('click', (e) => {
        if (rows.children.length <= 1) { flashError('At least one account is required.'); return; }
        e.target.closest('.setup-acc-row').remove();
      });
  }

  function updateAllocSum() {
    const sum = ['setupEquity', 'setupCrypto', 'setupCash']
      .reduce((s, id) => s + (parseInt($(id).value, 10) || 0), 0);
    const el = $('setupAllocSum');
    if (el) {
      el.textContent = sum + '%';
      el.classList.toggle('bad', sum !== 100);
    }
    return sum;
  }

  let _errTimer;
  function flashError(msg) {
    const el = $('setupError');
    if (!el) return;
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(_errTimer);
    _errTimer = setTimeout(() => el.classList.remove('show'), 4000);
  }

  /* ───────────── modal open / prefill ───────────── */
  function openSetup(isOnboarding, opts = {}) {
    const modal = $('setupModal');
    if (!modal) return;

    const profile = getProfile() || {};
    const accounts = getAccounts();
    const portfolio = getPortfolio();

    $('setupName').value = profile.name || '';
    $('setupCurrency').value = profile.currency || 'INR';
    $('setupEquity').value = portfolio.equity ?? 45;
    $('setupCrypto').value = portfolio.crypto ?? 30;
    $('setupCash').value   = portfolio.cash ?? 25;
    $('setupGoal').value   = portfolio.goal ? portfolio.goal.toLocaleString('en-IN') : '';

    const rows = $('setupAccountRows');
    rows.innerHTML = '';
    if (accounts.length) accounts.forEach(addAccountRow);
    else addAccountRow();

    // "Add Account" entry point → append a fresh empty row and focus it
    if (opts.addRow) {
      addAccountRow();
      const last = rows.lastElementChild;
      if (last) {
        last.scrollIntoView({ block: 'nearest' });
        last.querySelector('.setup-acc-name')?.focus();
      }
    }

    updateAllocSum();

    $('setupModalTitle').textContent = isOnboarding ? 'Welcome to FLUX — Set Up Your Account' : 'Edit Account Details';
    $('setupCloseBtn').style.display = isOnboarding ? 'none' : '';
    const skipBtn = $('setupSkipBtn');
    if (skipBtn) skipBtn.style.display = isOnboarding ? '' : 'none';
    modal.dataset.onboarding = isOnboarding ? '1' : '0';

    modal.classList.add('show');
    document.body.style.overflow = 'hidden';
    // Focus trap — use the shared utility if loaded, else fallback
    if (typeof trapFocus === 'function') trapFocus(modal);
  }

  function closeSetup(force = false) {
    const modal = $('setupModal');
    if (!modal) return;
    if (modal.dataset.onboarding === '1' && !force) return;
    modal.classList.remove('show');
    document.body.style.overflow = '';
    if (typeof releaseFocus === 'function') releaseFocus(modal);
  }

  /* ───────────── save ───────────── */
  function saveSetup() {
    const name = $('setupName').value.trim();
    if (!name) { flashError('Please enter your full name.'); return; }

    const sum = updateAllocSum();
    if (sum !== 100) { flashError(`Portfolio allocation must total 100% (currently ${sum}%).`); return; }

    // Gather accounts
    const rows = [...document.querySelectorAll('.setup-acc-row')];
    const accounts = [];
    for (const row of rows) {
      const aName = row.querySelector('.setup-acc-name').value.trim();
      const aBal  = parseAmount(row.querySelector('.setup-acc-bal').value);
      const last4 = row.querySelector('.setup-acc-card').value.replace(/[^0-9]/g, '').slice(-4);
      const exp   = row.querySelector('.setup-acc-exp').value.trim();
      if (!aName) continue;                       // skip blank rows
      const masked = '•••• •••• •••• ' + (last4 || '0000');
      accounts.push({
        name: aName,
        val: fmtINR(aBal),
        active: false,
        cardNum: masked,   // only the masked form is persisted — never a real card number
        expiry: '••/••'
      });
    }
    if (!accounts.length) { flashError('Add at least one account with a name.'); return; }
    accounts[0].active = true;                    // first account is the default/active one

    const profile = {
      name,
      initials: initialsFrom(name),
      currency: $('setupCurrency').value || 'INR'
    };
    const portfolio = {
      equity: parseInt($('setupEquity').value, 10) || 0,
      crypto: parseInt($('setupCrypto').value, 10) || 0,
      cash:   parseInt($('setupCash').value, 10) || 0,
      goal:   parseAmount($('setupGoal').value) || 15000000
    };

    localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
    localStorage.setItem('flux_accounts', JSON.stringify(accounts));
    localStorage.setItem('flux_portfolio', JSON.stringify(portfolio));

    const modal = $('setupModal');
    if (modal) modal.dataset.onboarding = '0';     // allow the reload to proceed cleanly

    // Reload so every container re-renders from the freshly-saved data.
    location.reload();
  }

  /* ───────────── wire up ───────────── */
  document.addEventListener('DOMContentLoaded', () => {
    applyProfile();
    applyActiveCard();

    const addBtn  = $('setupAddAccount');
    const saveBtn = $('setupSaveBtn');
    const closeBtn = $('setupCloseBtn');
    const skipBtn = $('setupSkipBtn');
    const modal = $('setupModal');

    if (addBtn)   addBtn.addEventListener('click', () => addAccountRow());
    if (saveBtn)  saveBtn.addEventListener('click', saveSetup);
    if (closeBtn) closeBtn.addEventListener('click', () => closeSetup());
    if (skipBtn)  skipBtn.addEventListener('click', () => closeSetup(true));
    if (modal) {
      modal.addEventListener('click', (e) => { if (e.target === modal) closeSetup(); });
      modal.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSetup(); });
    }

    ['setupEquity', 'setupCrypto', 'setupCash'].forEach(id => {
      const el = $(id);
      if (el) el.addEventListener('input', updateAllocSum);
    });

    // Entry points for editing later
    const profileBtn = document.querySelector('.user-profile');
    if (profileBtn) {
      profileBtn.style.cursor = 'pointer';
      profileBtn.title = 'Edit account details';
      profileBtn.addEventListener('click', () => openSetup(false));
      // Keyboard activation for the now-focusable role="button" element
      profileBtn.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openSetup(false); }
      });
    }
    const settingsNav = $('settingsNav');
    if (settingsNav) settingsNav.addEventListener('click', (e) => { e.preventDefault(); openSetup(false); });

    // The shared demo account is already seeded: give it a profile instead of
    // asking every visitor to fill in the onboarding form.
    const user = readJSON('flux_user', null);
    if (!getProfile() && user && user.email === DEMO_EMAIL) {
      localStorage.setItem(PROFILE_KEY, JSON.stringify({ name: user.name || 'Demo', currency: 'INR' }));
      applyProfile();
    }

    // First visit → mandatory onboarding (after the splash clears)
    if (!getProfile()) {
      setTimeout(() => openSetup(true), 2700);
    }
  });

  // Expose for any other UI that wants to trigger it
  // opts.addRow → open with a fresh empty account row ("Add Account" flow)
  window.openAccountSetup = (opts = {}) => openSetup(false, opts);
})();
