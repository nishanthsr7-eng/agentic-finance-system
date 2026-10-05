# Datasets

How to download every dataset the prediction agent was trained or tested on.
Files are staged locally under `Dataset/` (git-ignored, not in the repo). The
download scripts live in `scripts/` and are run from the project root.

> **What the served model needs:** only §1, the daily OHLCV history. Every other
> dataset feeds a feature block that was tested and switched off (see
> [docs/AGENT_TRAINING.md](AGENT_TRAINING.md) §6); they are needed only to
> re-run those experiments.
>
> Most sources are keyless and public. Kaggle requires `KAGGLE_USERNAME` /
> `KAGGLE_KEY` (see [docs/API_KEYS.md](API_KEYS.md)).

---

## Quick Reference

| Dataset | Script | Keyless | Destination |
|---|---|---|---|
| Daily OHLCV history (**served model**) | `scripts/backfill_history.py` | yes (yfinance) | `ohlcv_history` table |
| Binance funding + 1d klines | `scripts/dl_binance.py` | yes | `Dataset/binance/` |
| Binance open interest | `scripts/dl_binance_oi_parallel.py` | yes | `Dataset/binance/` |
| Deribit DVOL (implied vol) | `scripts/dl_deribit.py` | yes | `Dataset/deribit/` |
| Coin Metrics on-chain | `scripts/dl_coinmetrics.py` | yes | `Dataset/coinmetrics/` |
| Blockchain.com on-chain | included CSVs | yes | `Dataset/*.csv` |
| Crypto Fear & Greed | included JSON | yes | `Dataset/fng.json` |
| SEC fundamentals | EDGAR bulk archives (manual download) | yes | `Dataset/SEC.../` |
| Financial PhraseBank | Hugging Face | yes (public) | `Dataset/financial_phrasebank/` |
| Huge Stock Market | Kaggle | `KAGGLE_*` | `Dataset/Huge Stock Market Dataset/` |
| Stooq bulk EOD | <https://stooq.com> | yes | `Dataset/Stooq bulk EOD/` |

---

## 1. Daily OHLCV History (primary training backbone)

The model trains on *years* of daily bars stored in the `ohlcv_history` SQLite
table - separate from the 30-day live table.

```bash
python scripts/backfill_history.py
```

- Pulls `period="max"` daily OHLCV via yfinance for the 30-symbol universe
  (crypto, stocks and the S&P 500 / VIX / 10Y-yield market series).
- In production, `MARKET_STORE=mysql` plus `scripts/copy_history_to_tidb.py`
  keeps this table in TiDB so it survives redeploys; the daily 00:10 UTC job
  appends new bars.
- Upserts on `(symbol, date)`, so re-running is safe and idempotent.
- Use `--symbols AAPL BTC` to backfill a subset for a smoke test.

Verify afterwards:

```sql
SELECT symbol, COUNT(*), MIN(date), MAX(date) FROM ohlcv_history GROUP BY symbol;
```

Target: at least ~2,500 rows per symbol.

---

## 2. Binance Funding Rate + Klines

The primary crypto signal (positioning-derived, orthogonal to price).

```bash
python scripts/dl_binance.py                 # funding + 1d klines, from 2019
python scripts/dl_binance.py --start 2021    # narrower history
python scripts/dl_binance.py --oi            # ALSO pull open interest (slow)
```

- Keyless public bucket (`data.binance.vision`).
- Writes one combined CSV per symbol into `Dataset/binance/`.
- Open interest is opt-in because it only exists as per-day files (heavy).

For a faster, parallel open-interest pull:

```bash
python scripts/dl_binance_oi_parallel.py
```

---

## 3. Deribit DVOL (crypto implied volatility)

The crypto analog of VIX, fully backfillable so it can enter leak-free training.

```bash
python scripts/dl_deribit.py                 # ~6 years daily, BTC & ETH
python scripts/dl_deribit.py --years 8 --resolution 1D
```

- Keyless. Writes `Dataset/deribit/dvol_<CCY>.csv`.

---

## 4. Coin Metrics On-Chain

The free, keyless, multi-asset replacement for Glassnode.

```bash
python scripts/dl_coinmetrics.py
python scripts/dl_coinmetrics.py --metrics AdrActCnt,TxCnt,FeeTotUSD
```

- Keyless community API. Writes `Dataset/coinmetrics/onchain.csv` (long format).
- Default metrics: active addresses, transaction count, fees, supply.

---

## 5. Blockchain.com On-Chain (BTC)

BTC-specific on-chain series are staged directly as CSVs in `Dataset/`:
`hash-rate.csv`, `n-transactions.csv`, `n-unique-addresses.csv`,
`transaction-fees.csv`. Refresh from <https://www.blockchain.com/charts>.

---

## 6. Crypto Fear & Greed Index

Staged as `Dataset/fng.json` (2018→). Source API:
<https://api.alternative.me/fng/?limit=0&format=json>.

---

## 7. SEC Fundamentals

Point-in-time fundamentals from SEC EDGAR financial-statement data sets, staged
under `Dataset/SEC financial statement data sets/`. Download the quarterly zip
files by hand; SEC asks automated clients to send a User-Agent with a contact
email. The loader lives at
`backend/prediction/datasources/sec_fundamentals.py`; `Dataset/company_tickers.json`
maps tickers to CIK numbers. Bulk archives:
<https://www.sec.gov/dera/data/financial-statement-data-sets>.

---

## 8. Financial PhraseBank (sentiment fine-tune/eval)

14,787 finance sentences labelled by sentiment - the standard FinBERT
evaluation set, used by the gate-5 sentiment checks. Staged under
`Dataset/financial_phrasebank/`. Public on Hugging Face (`financial_phrasebank`);
no token needed.

---

## 9. Huge Stock Market Dataset (Kaggle - breadth/backfill)

7,195 stocks + 1,344 ETFs of historical prices, staged under
`Dataset/Huge Stock Market Dataset/`.

```bash
# Requires KAGGLE_USERNAME / KAGGLE_KEY in .env (or ~/.kaggle/kaggle.json)
pip install kaggle
kaggle datasets download -d borismarjanovic/price-volume-data-for-all-us-stocks-etfs \
  -p "Dataset/Huge Stock Market Dataset" --unzip
```

Use this for breadth and cross-validation only - never train production signals
solely on it.

---

## 10. Stooq Bulk EOD

Global end-of-day prices for cross-checking yfinance split/dividend adjustments.
Staged (zipped) under `Dataset/Stooq bulk EOD/`. Source:
<https://stooq.com/db/h/>.

---

## Dataset Roles Summary

| Block | Datasets | Status |
|---|---|---|
| Price backbone | yfinance OHLCV (incl. S&P 500, VIX, 10Y yield) | **Served**: features + labels |
| Price cross-checks | Stooq, Huge Stock Market | Validation only |
| Crypto positioning | Binance funding/OI, Deribit DVOL | Gate 2 failed: off |
| On-chain | Coin Metrics, blockchain.com | Gate 2 failed: off |
| Crypto regime | Fear & Greed | Gate 2 failed: off |
| Equity fundamentals | SEC | Gate 4 tie: off |
| Macro | FRED | Gate 1: hurt ranking Sharpe, off |
| Sentiment | Financial PhraseBank, news, Reddit | Off as a feature; live news drives the confidence tilt |
