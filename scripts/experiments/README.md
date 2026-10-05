# Research gates

Every component of the prediction agent had to pass a measurable out-of-sample
check before it was allowed into the served model. These scripts are those
checks. They are kept so the results in the main README can be reproduced; the
API never imports them.

Run them from the repository root, e.g. `python scripts/experiments/gate0_finalize.py`.
Most need the datasets described in [docs/DATASETS.md](../../docs/DATASETS.md).

| Gate | Question it answers | Scripts | Result |
|---|---|---|---|
| 0 | What is the reproducible baseline? | `gate0_finalize.py`, `repro_baseline.py` | Price-only baseline locked |
| 1 | Do FRED macro features add out-of-sample signal? | `gate1_fred_macro_*.py` | AUC +0.005 but ranking Sharpe halved: off |
| 2 | Do crypto-specific features (funding, open interest, on-chain, DVOL) help? | `gate2_crypto_features_*.py` | Below the +0.01 bar: off |
| 3 | Does a model per asset class beat one shared model? | `gate3_asset_class_specialization.py` | No: pooled model kept |
| 4 | Do SEC equity fundamentals help? | `gate4_equity_fundamentals_*.py` | Tie: off |
| 5 | Is news sentiment (FinBERT, CryptoBERT, LLM) accurate and useful? | `gate5_*.py` | Too little history to test as a feature: off; used as a live tilt |
| 6 | Does a regime-conditional ensemble beat the plain model? | `gate6_regime_ensemble_*.py` | No (0.536 vs 0.537): off |
| 7 | Does portfolio construction improve risk-adjusted return? | `gate7_portfolio_audit.py` | Long-only top 10% + vol target, thin pass |
| 8 | Does the verifier only ever lower confidence, and does the flywheel run? | `gate8_llm_verifier_audit.py` | Pass |
| 9 | Does the model beat naive baselines at all? | `gate9_honesty_baselines_audit.py` | Not on raw accuracy; see the main README |

`fluxx_audit.py` audits the FLUX-X model end to end.

A gate that fails leaves its component switched off. The regime stack (gate 6)
is an example: it did not beat the plain model, so it is not served.
