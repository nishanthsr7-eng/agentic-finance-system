"""
Datasource — SEC EDGAR fundamentals (Financial Statement Data Sets)  →  point-in-time daily.

Source: ``Dataset/SEC financial statement data sets/extracted/<YYYYqQ>/{sub.txt,num.txt}``
    sub.txt — one row per filing: adsh, cik, name, form, period, fy, fp, **filed** (YYYYMMDD).
    num.txt — the numeric facts: adsh, tag, version, ddate, qtrs, segments, coreg, value.

LEAKAGE — THE ONE POINT-IN-TIME LOADER: a quarter's numbers are public only on the **filing date**
(``filed``), weeks after the reporting **period** ends. We therefore key every row on ``date = filed``
(NOT period/ddate). Using period-end would leak figures the market hadn't seen. Phase 4 derives
earnings_surprise / revision / valuation_z on top of these as-reported values.

Universe: restricted to the equities in ``ingestion.STOCK_META`` (mapped via
``Dataset/company_tickers.json``). Foreign filers (e.g. TSM, BABA) file 20-F and may be absent.

Parsing 45×~0.5 GB ``num.txt`` files is slow, so the assembled frame is CACHED to
``Dataset/coinmetrics/../sec_fundamentals.csv`` (next to the SEC sets) on first build; later calls
read the cache. Pass ``rebuild=True`` to force a re-parse.

    load_sec_fundamentals()              -> DataFrame[symbol, date, period, fy, fp, form, <metrics…>]
    load_sec_fundamentals(rebuild=True)  -> re-parse from the raw quarter zips' extracts
"""

from __future__ import annotations

import json

import pandas as pd

from . import DATASET_DIR, tidy

_SEC_DIR = DATASET_DIR / "SEC financial statement data sets" / "extracted"
_CACHE = DATASET_DIR / "SEC financial statement data sets" / "sec_fundamentals.csv"
_TICKERS = DATASET_DIR / "company_tickers.json"

# Metric -> ordered list of acceptable us-gaap tags (first present wins, per filing).
_METRIC_TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "op_income": ["OperatingIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
}
_ALL_TAGS = {t for tags in _METRIC_TAGS.values() for t in tags}
_NUM_COLS = ["adsh", "tag", "ddate", "qtrs", "segments", "coreg", "value"]


def _ticker_to_cik() -> dict[str, int]:
    """Our equity universe -> CIK, from the SEC ticker map intersected with STOCK_META."""
    try:
        from ...ingestion import STOCK_META
    except Exception:
        STOCK_META = {}
    cik_map = json.loads(_TICKERS.read_text())
    out = {}
    for v in cik_map.values():
        t = v["ticker"].upper()
        if not STOCK_META or t in STOCK_META:
            out[t] = int(v["cik_str"])
    return out


def _subs_for(quarter_dir, cik_to_ticker: dict[int, str]) -> pd.DataFrame:
    """Filings in this quarter belonging to our universe, keyed by filing date."""
    sub_fp = quarter_dir / "sub.txt"
    if not sub_fp.exists():
        return pd.DataFrame()
    sub = pd.read_csv(
        sub_fp, sep="\t", dtype=str, usecols=["adsh", "cik", "form", "period", "fy", "fp", "filed"]
    )
    sub["cik"] = pd.to_numeric(sub["cik"], errors="coerce")
    sub = sub[sub["cik"].isin(cik_to_ticker) & sub["form"].isin(["10-K", "10-Q", "20-F"])]
    if sub.empty:
        return sub
    sub["symbol"] = sub["cik"].map(cik_to_ticker)
    sub["date"] = pd.to_datetime(sub["filed"], format="%Y%m%d", errors="coerce")
    sub["period"] = pd.to_datetime(sub["period"], format="%Y%m%d", errors="coerce")
    return sub[["adsh", "symbol", "date", "period", "fy", "fp", "form"]].dropna(subset=["date"])


def _nums_for(quarter_dir, adsh_set: set[str]) -> pd.DataFrame:
    """Whitelisted facts for the given filings, reduced to one value per (adsh, metric)."""
    num_fp = quarter_dir / "num.txt"
    if not num_fp.exists() or not adsh_set:
        return pd.DataFrame()
    keep = []
    for chunk in pd.read_csv(num_fp, sep="\t", dtype=str, usecols=_NUM_COLS, chunksize=1_000_000):
        c = chunk[chunk["adsh"].isin(adsh_set) & chunk["tag"].isin(_ALL_TAGS)]
        # Consolidated company-level only: no segment/co-registrant breakdowns.
        c = c[c["segments"].isna() & c["coreg"].isna()]
        if not c.empty:
            keep.append(c[["adsh", "tag", "ddate", "qtrs", "value"]])
    if not keep:
        return pd.DataFrame()
    facts = pd.concat(keep, ignore_index=True)
    facts["ddate"] = pd.to_numeric(facts["ddate"], errors="coerce")
    facts["qtrs"] = pd.to_numeric(facts["qtrs"], errors="coerce")
    facts["value"] = pd.to_numeric(facts["value"], errors="coerce")
    facts = facts.dropna(subset=["value"])
    # Per (adsh, tag): take the latest reporting period (max ddate), then the longest span (max qtrs).
    facts = facts.sort_values(["ddate", "qtrs"]).drop_duplicates(["adsh", "tag"], keep="last")
    # Coalesce each metric's candidate tags into one column.
    rows = {}
    for adsh, g in facts.groupby("adsh"):
        by_tag = dict(zip(g["tag"], g["value"]))
        rows[adsh] = {
            m: next((by_tag[t] for t in tags if t in by_tag), None)
            for m, tags in _METRIC_TAGS.items()
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("adsh").reset_index()


def _build() -> pd.DataFrame:
    t2c = _ticker_to_cik()
    c2t = {c: t for t, c in t2c.items()}
    quarters = sorted(p for p in _SEC_DIR.iterdir() if p.is_dir()) if _SEC_DIR.exists() else []
    frames = []
    for qd in quarters:
        subs = _subs_for(qd, c2t)
        if subs.empty:
            continue
        nums = _nums_for(qd, set(subs["adsh"]))
        merged = subs.merge(nums, on="adsh", how="left") if not nums.empty else subs
        frames.append(merged)
        print(f"  {qd.name}: {len(subs)} filings", flush=True)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop(columns=["adsh"])
    metric_cols = list(_METRIC_TAGS)
    cols = ["symbol", "date", "period", "fy", "fp", "form", *metric_cols]
    out = out[[c for c in cols if c in out.columns]]
    return tidy(out)


def load_sec_fundamentals(rebuild: bool = False) -> pd.DataFrame:
    """Point-in-time fundamentals keyed (symbol, filing date). Builds + caches on first call."""
    metric_cols = list(_METRIC_TAGS)
    cols = ["symbol", "date", "period", "fy", "fp", "form", *metric_cols]
    if not rebuild and _CACHE.exists():
        df = pd.read_csv(_CACHE, parse_dates=["date", "period"])
        return df[[c for c in cols if c in df.columns]]
    if not _SEC_DIR.exists():
        return pd.DataFrame(columns=cols)
    df = _build()
    if not df.empty:
        df.to_csv(_CACHE, index=False)
    return df


if __name__ == "__main__":
    import sys

    df = load_sec_fundamentals(rebuild="--rebuild" in sys.argv)
    if df.empty:
        print("SEC: no data (extract the quarter zips into extracted/ first).")
        sys.exit(0)
    # Point-in-time invariant: a filing is only known on/after its filing date.
    assert (df["date"] >= df["period"]).all(), (
        "filing date precedes its own reporting period (leak!)"
    )
    assert df.duplicated(["symbol", "date"]).sum() == 0
    print(
        f"SEC fundamentals: {len(df):,} filings over {df.symbol.nunique()} tickers "
        f"({df.date.min().date()}..{df.date.max().date()})"
    )
    print("tickers:", sorted(df.symbol.unique()))
    cov = {m: f"{df[m].notna().mean():.0%}" for m in _METRIC_TAGS if m in df}
    print("metric coverage:", cov)
    print(df[df.symbol == "AAPL"].tail(3).to_string(index=False))
    print("OK")
