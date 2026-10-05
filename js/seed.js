/* FLUX — Initial Data Seed
 * Runs once on first load. Guards with flux_seeded flag.
 * Generates ~12 months of realistic transactions ANCHORED TO TODAY,
 * so the dashboard is never empty regardless of the current date.
 */
(function seedFluxData() {
  const SEED_VERSION = '2';
  // Never inject the hardcoded demo dataset for a signed-in user. Their data
  // is authoritative from the backend (/db/* scoped to their user_id) — a new
  // account legitimately has none, and must see empty states, not this seed.
  // The seed survives only as an offline convenience for anonymous/dev mode
  // (AUTH_REQUIRED=false, no token), where the dashboard is otherwise empty.
  if (localStorage.getItem('flux_token')) return;
  if (localStorage.getItem('flux_seeded') === SEED_VERSION) return;
  // Clear old seed data so updated amounts take effect
  ['flux_transactions','flux_accounts','flux_portfolio','flux_recurring','flux_contacts','flux_protocols','flux_reward_states'].forEach(k => localStorage.removeItem(k));

  let _id = 1;
  function tx(y, m, d, title, amount, category, account) {
    return {
      id: 'tx_' + String(_id++).padStart(4, '0'),
      title,
      date: new Date(y, m, d, 10, 0, 0).toISOString(),
      amount,
      category,
      account: account || 'HDFC Platinum'
    };
  }

  // Variant-based amounts so months differ realistically
  const salaryBase  = [182000, 184500, 188000];
  const groceryAmt  = [4800, 5600, 6200];
  const restAmt     = [2200, 3100, 4000];
  const zomatoAmt   = [1200, 1900, 2600];
  const uberAmt     = [600, 950, 1400];
  const investExtra = [0, 0, 25000]; // occasional extra investment

  const txs = [];

  // Standard month template. maxDay clamps the current (partial) month.
  function genMonth(y, m, v, maxDay) {
    const push = (d, title, amount, category, account) => {
      if (d <= maxDay) txs.push(tx(y, m, d, title, amount, category, account));
    };
    push(1,  'Salary — Acme Corp',       salaryBase[v],  'income');
    push(2,  'Vanguard S&P 500 SIP',     -45000,         'investment');
    push(3,  'BigBasket / Groceries',    -groceryAmt[v], 'essentials');
    push(4,  'ChatGPT Plus',             -1800,          'subscription');
    push(5,  'AWS Cloud Services',       -8500,          'business');
    push(7,  'BESCOM Electricity',       -2800,          'essentials');
    push(10, 'Netflix India',            -649,           'subscription');
    push(12, 'Restaurant — Taj Bistro',  -restAmt[v],    'lifestyle');
    push(14, 'Adobe Creative Cloud',       -1450,          'subscription');
    push(15, 'Zomato / Blinkit',         -zomatoAmt[v],  'lifestyle');
    if (investExtra[v]) push(18, 'Binance — BTC Purchase', -investExtra[v], 'investment');
    push(20, 'Uber / Rapido',            -uberAmt[v],    'lifestyle');
    if (v === 2) push(20, 'Freelance Settlement', 18000, 'income');
    push(22, 'Rent — Koramangala',       -22000,         'essentials');
    push(25, 'Zerodha — ETF SIP',        -10000,         'investment');
    push(28, 'HDFC Platinum — CC Settle', -12000,        'transfer', 'ICICI Wealth');
  }

  // Anchor everything to "today" at seed time.
  const today = new Date();
  const curY  = today.getFullYear();
  const curM  = today.getMonth();
  const curD  = today.getDate();
  const lastDayOf = (y, m) => new Date(y, m + 1, 0).getDate();

  // 11 complete prior months (oldest → newest), variant cycles 0,1,2
  for (let i = 11; i >= 1; i--) {
    const d = new Date(curY, curM - i, 1);
    const y = d.getFullYear();
    const m = d.getMonth();
    const v = (11 - i) % 3;
    genMonth(y, m, v, lastDayOf(y, m));
  }

  // Current (partial) month — only entries up to today
  genMonth(curY, curM, (11 % 3), curD);

  // Newest first
  txs.sort((a, b) => new Date(b.date) - new Date(a.date));

  localStorage.setItem('flux_transactions', JSON.stringify(txs));

  // Masked card data only — unmasked PAN / expiry must never be persisted
  // client-side (mirrors the backend contract in js/flux-data.js mapAccounts).
  // balance/creditLimit are kept numeric so the Payments source card and
  // headroom gauge can render real figures offline.
  localStorage.setItem('flux_accounts', JSON.stringify([
    { name: 'HDFC Platinum', val: '₹84,500',   balance: 84500,  creditLimit: 420000, acctType: 'credit',  active: true,  cardNum: '•••• •••• •••• 8842', expiry: '••/••' },
    { name: 'ICICI Wealth',  val: '₹2,45,000', balance: 245000, creditLimit: 0,      acctType: 'savings', active: false, cardNum: '•••• •••• •••• 1290', expiry: '••/••' }
  ]));

  localStorage.setItem('flux_portfolio', JSON.stringify({
    equity: 24, crypto: 14, cash: 62, goal: 15000000
  }));

  localStorage.setItem('flux_recurring', JSON.stringify([
    { title: 'AWS Infrastructure', amount: 8500,  dueDay: 5,  category: 'business'      },
    { title: 'ChatGPT Plus',       amount: 1800,  dueDay: 4,  category: 'subscription'  },
    { title: 'Netflix India',      amount: 649,   dueDay: 10, category: 'subscription'  },
    { title: 'Rent — Koramangala', amount: 22000, dueDay: 22, category: 'essentials'    }
  ]));

  localStorage.setItem('flux_contacts', JSON.stringify([
    { name: 'Alex',   fluxId: 'alex@flux',   initial: 'A' },
    { name: 'Sanjay', fluxId: 'sanjay@flux', initial: 'S' },
    { name: 'Meera',  fluxId: 'meera@flux',  initial: 'M' }
  ]));

  // Mirrors the first three security_settings (2FA, Biometric, Txn Alerts) the
  // Payments page exposes — all enabled, matching the MySQL seed.
  localStorage.setItem('flux_protocols',    JSON.stringify([true, true, true]));
  localStorage.setItem('flux_reward_states', JSON.stringify({}));
  localStorage.setItem('flux_seeded', SEED_VERSION);
  // Marks everything above as sample data; js/flux-data.js shows a badge while
  // it is set and clears it once real data has loaded from the API.
  localStorage.setItem('flux_sample_data', '1');
})();
