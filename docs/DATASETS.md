# Datasets

How to download every dataset the prediction agent trains on. All datasets are
staged locally under `Dataset/`. The download scripts live in `scripts/` and are
run from the project root.

> Most sources are keyless and public. Kaggle requires `KAGGLE_USERNAME` /
> `KAGGLE_KEY` in `.env` (see [docs/API_KEYS.md](API_KEYS.md)).

---

## Quick Reference

| Dataset | Script | Keyless | Destination |
|---|---|---|---|
| Daily OHLCV history (train backbone) | `scripts/backfill_history.py` | yes (yfinance) | `ohlcv_history` table |
| Binance funding + 1d klines | `scripts/dl_binance.py` | yes | `Dataset/binance/` |
| Binance open interest | `scripts/dl_binance_oi_parallel.py` | yes | `Dataset/binance/` |
| Deribit DVOL (implied vol) | `scripts/dl_deribit.py` | yes | `Dataset/deribit/` |
| Coin Metrics on-chain | `scripts/dl_coinmetrics.py` | yes | `Dataset/coinmetrics/` |
| Blockchain.com on-chain | included CSVs | yes | `Dataset/*.csv` |
| Crypto Fear & Greed | included JSON | yes | `Dataset/fng.json` |
| SEC fundamentals | EDGAR (manual/build script) | `SEC_USER_AGENT` | `Dataset/SEC.../` |
| Financial PhraseBank | Hugging Face | `HUGGINGFACE_TOKEN` | `Dataset/financial_phrasebank/` |
| Huge Stock Market | Kaggle | `KAGGLE_*` | `Dataset/Huge Stock Market Dataset/` |
| Stooq bulk EOD | <https://stooq.com> | yes | `Dataset/Stooq bulk EOD/` |

---

## 1. Daily OHLCV History (primary training backbone)

The model trains on *years* of daily bars stored in the `ohlcv_history` SQLite
table — separate from the 30-day live table.

```bash
python scripts/backfill_history.py
```

- Pulls `period="max"` daily OHLCV via yfinance for all 15 stocks, 15 crypto,
  and macro series (S&P 500, VIX, 10Y yield).
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
under `Dataset/SEC financial statement data sets/`. Requires a descriptive
`SEC_USER_AGENT` (contact email) per SEC policy. The loader lives at
`backend/prediction/datasources/sec_fundamentals.py`; `Dataset/company_tickers.json`
maps tickers to CIK numbers. Bulk archives:
<https://www.sec.gov/dera/data/financial-statement-data-sets>.

---

## 8. Financial PhraseBank (sentiment fine-tune/eval)

14,787 finance sentences labelled by sentiment — the standard FinBERT
evaluation set. Staged under `Dataset/financial_phrasebank/`. Pull via Hugging
Face (`financial_phrasebank`) with `HUGGINGFACE_TOKEN`.

---

## 9. Huge Stock Market Dataset (Kaggle — breadth/backfill)

7,195 stocks + 1,344 ETFs of historical prices, staged under
`Dataset/Huge Stock Market Dataset/`.

```bash
# Requires KAGGLE_USERNAME / KAGGLE_KEY in .env (or ~/.kaggle/kaggle.json)
pip install kaggle
kaggle datasets download -d borismarjanovic/price-volume-data-for-all-us-stocks-etfs \
  -p "Dataset/Huge Stock Market Dataset" --unzip
```

Use this for breadth and cross-validation only — never train production signals
solely on it.

---

## 10. Stooq Bulk EOD

Global end-of-day prices for cross-checking yfinance split/dividend adjustments.
Staged (zipped) under `Dataset/Stooq bulk EOD/`. Source:
<https://stooq.com/db/h/>.

---

## Dataset Roles Summary

| Block | Datasets | Role in the model |
|---|---|---|
| Price backbone | yfinance OHLCV, Stooq, Huge Stock Market | features + labels |
| Crypto positioning | Binance funding/OI, Deribit DVOL | orthogonal crypto signal |
| On-chain | Coin Metrics, blockchain.com | crypto fundamentals |
| Crypto regime | Fear & Greed | regime feature |
| Equity fundamentals | SEC, FMP, Tiingo | equity feature block |
| Macro | FRED, VIX/yields | regime + equity features |
| Sentiment | Financial PhraseBank, news, Reddit | sentiment block |
