# Singapore NIRC & GDP - reproducibility pipeline

Replication code for **"Singapore's Sovereign Wealth Strategy: A Cointegration Analysis of GDP,
Net Investment Returns Contribution, and Fiscal Sustainability (1994-2023)"** (Langer, 2026).

## Contents

| File | Description |
|---|---|
| `singapore_nirc_analysis.ipynb` | Colab-ready notebook (recommended entry point) |
| `singapore_nirc_analysis.py` | The same pipeline as a single script |
| `appendix_a_corrected.csv` | Appendix A dataset, with original and corrected CPI columns |

## Running it

**Google Colab:** File → Upload notebook → select the `.ipynb` → Runtime → Run all.
The first cell installs two small extras (`arch`, `ruptures`) if they are missing.

**Locally:** `python singapore_nirc_analysis.py` (Python 3.12; numpy, pandas, scipy,
matplotlib, statsmodels; `arch` and `ruptures` optional).

Set `FETCH_LIVE = False` in the configuration block for a fully offline, deterministic run
against the embedded dataset. Outputs are written to `nirc_outputs/`.

## Modules

M0 environment | M1 data assembly | M2 cross-source validation | M2b CPI base-break repair |
M3 respecification | M4 causal estimand | M5 unit roots | M6 ARDL bounds + ECM with
design-specific bootstrap | M7 residual diagnostics | M8 Toda-Yamamoto causality |
M9 VAR and impulse responses | M10 stability tests | M11 robustness ladder and out-of-sample |
M12 tax equivalence | M13 budget composition | M14 debt | M15 instrumental variable |
M16 export | M17 consolidated results.

## Reported run

Python 3.12.13, NumPy 2.0.2, pandas 2.2.2, SciPy 1.16.3, statsmodels 0.14.6, arch 8.0.0,
ruptures 1.1.10. Seed `20260518`. Runtime ≈ 19 s.

## Data sources

All free and key-less. Singapore open data portal (Singapore Open Data License v1.0);
World Bank Indicators API (CC BY 4.0); IMF World Economic Outlook via DBnomics.
Every retrieval is provenance-tagged at runtime and degrades to a flagged embedded
fallback when offline.

## Note for other researchers

`bounds_test` in statsmodels 0.14.6 silently ignores fixed regressors: it returns an
identical statistic with and without crisis dummies. All dummy-inclusive small-sample
inference in the paper therefore comes from the design-specific bootstrap implemented here.

## License

Code: MIT (see `LICENSE`). Third-party data remain subject to their providers' terms.
Research and educational use only; nothing here is investment, tax, or policy advice.
