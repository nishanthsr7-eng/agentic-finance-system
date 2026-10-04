-- One-off cleanup for step 8a: remove NIFTY 50 from the live TiDB data.
-- Run once by hand against TiDB (e.g. in the TiDB Cloud SQL Editor).
-- Only the demo user (id 1) is changed; other users' rows are left alone.
-- Safe to re-run: every statement is a no-op once the rows are gone.

START TRANSACTION;

-- Demo user's ETF holding: NIFTYBEES → SPY (priced in USD, shown in INR).
DELETE FROM portfolio_holdings WHERE user_id = 1 AND symbol = 'SPY';
UPDATE portfolio_holdings
   SET symbol = 'SPY', name = 'SPDR S&P 500 ETF', asset_type = 'etf',
       quantity = 2, avg_price = 48900.00, currency = 'USD'
 WHERE user_id = 1 AND symbol = 'NIFTYBEES';

-- Demo user's monthly SIP title (step 8b regenerates these rows anyway).
UPDATE transactions SET title = 'Vanguard S&P 500 SIP'
 WHERE user_id = 1 AND title = 'Groww — NIFTY50 ETF';

-- Market search catalogue.
DELETE FROM asset_catalog WHERE symbol = '^NSEI';

-- Market data and forecasts that ingestion no longer refreshes.
DELETE FROM prediction_outcomes
 WHERE prediction_id IN (SELECT id FROM predictions WHERE symbol = 'NIFTY');
DELETE FROM predictions     WHERE symbol = 'NIFTY';
DELETE FROM ai_insights     WHERE symbol IN ('NIFTY', '^NSEI');
DELETE FROM price_snapshots WHERE symbol IN ('NIFTY', '^NSEI');
DELETE FROM ohlcv_daily     WHERE symbol IN ('NIFTY', '^NSEI');
DELETE FROM ohlcv_history   WHERE symbol IN ('NIFTY', '^NSEI');

COMMIT;

-- Check: every count should be 0.
SELECT
  (SELECT COUNT(*) FROM portfolio_holdings WHERE symbol LIKE '%NIFTY%') AS holdings,
  (SELECT COUNT(*) FROM transactions WHERE title LIKE '%NIFTY%')       AS txns,
  (SELECT COUNT(*) FROM asset_catalog WHERE symbol = '^NSEI')          AS catalog,
  (SELECT COUNT(*) FROM predictions WHERE symbol = 'NIFTY')            AS preds;
