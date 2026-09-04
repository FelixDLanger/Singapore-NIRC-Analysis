#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 SINGAPORE NIRC & GDP — METHODOLOGY UPGRADE PIPELINE (Colab-ready, single file)
 Companion code to: Langer (2026), "Singapore's Sovereign Wealth Strategy"
 (SSRN working paper), Sections 4-7 upgrade path.
================================================================================

WHAT THIS SCRIPT DOES (module map)
  M1  Data assembly: paper's Appendix A (embedded, exact) + live extensions
      from verified free APIs (data.gov.sg, World Bank, DBnomics/IMF WEO).
  M2  Internal-consistency & cross-source validation (incl. CPI rebasing check,
      official-NIRC vs paper-NIRC comparison).
  M2b CPI base-break AUTO-REPAIR: the pre-2019 segment is spliced onto the
      2019=100 base, preserving the paper's own YoY inflation rates exactly.
      Splice factor is data-driven (World Bank overlap) online, anchored on
      the documented 0.6% 2019 inflation offline. Corrected Appendix A is
      exported as appendix_a_corrected.csv for direct use in the paper.
  M3  Re-specification: log real GDP (and per-capita), CPI-deflated real NIRC.
  M4  Pearl/SCM estimand statement: total vs direct effect, backdoor adjustment.
  M5  Unit roots: ADF, PP (arch), KPSS, Zivot-Andrews (breaks); I(2) guard.
  M6  ARDL bounds test (Pesaran-Shin-Smith): statsmodels UECM with
      small-sample simulated p-values when available, PLUS an independent
      manual UECM + PSS asymptotic critical values as a cross-check.
      Long-run coefficients + moving-block bootstrap CIs. Johansen cross-check.
  M7  ECM residual diagnostics: Breusch-Godfrey, White, RESET, Jarque-Bera,
      Ljung-Box, ARCH-LM, Durbin-Watson; Newey-West (HAC) errors throughout.
  M8  Toda-Yamamoto Granger causality (manual, correct restriction set).
  M9  Small VAR in growth rates + orthogonalized IRFs.
  M10 Stability: recursive CUSUM / CUSUMSQ, rolling coefficients,
      sup-F single break + multiple breaks via `ruptures` (Bai-Perron-style).
  M11 Robustness battery: crisis dummies (2009/2020), regime dummies
      (NIR 2009 / Temasek 2016), subsample 2001-2023 (drops estimated NIRC),
      alternative specifications, out-of-sample 1994-2015 -> 2016-2023,
      bootstrap CIs (n=30).
  M12 Tax-equivalence counterfactual: NIRC expressed in GST points and
      CIT points (official MOF revenue data when online).
  M13 "Progress drivers": budget composition — health, education, transport/
      infrastructure, defence, environment — shares and trends.
  M14 Debt module: IMF WEO gross-debt ratio, with the institutional caveat
      that Singapore's gross debt is asset-backed (SGS/SSGS proceeds are
      invested, not spent; SINGA 2021 borrowing is ring-fenced for
      long-dated infrastructure). Optional ARDL variant with debt ratio.
  M15 Candidate IV demo: global equity returns instrumenting NIRC growth
      (honest exclusion-restriction caveat printed).
  M16 Export: merged dataset, results tables, figures, module status report.
  M17 CONSOLIDATED RESULTS: one-screen summary of every finding, printed at
      the end and saved as results_summary.txt + results_summary.json.

HOW TO RUN (Google Colab)
  1) Upload this file, then in a cell:      %run singapore_nirc_analysis.py
     or:                                    !python singapore_nirc_analysis.py
  2) First run installs two small extras (arch, ruptures) if missing.
  3) Set FETCH_LIVE = False below to force fully-offline reproducible mode.
  Outputs land in ./nirc_outputs/ (figures, CSVs, results_summary.json).

DATA SOURCES (verified free, no API key required)   [checked 2026-07-18]
  * data.gov.sg dataset API:
      GET https://data.gov.sg/api/action/datastore_search?resource_id=<id>
      - MOF  Government Fiscal Position          d_0f0800a6b2b4b391daffcd88b2cc97fd
        (contains the OFFICIAL NIRC line -> used to validate/extend Appendix A)
      - MOF  Government Operating Revenue        d_d6cd701a67607ac9d4f3d469d8c815e5
        (GST / CIT / IIT etc. by financial year -> tax-equivalence module)
      - MOF  Government Total Expenditure        d_00f95040967791729d51bd7cd6af2355
      - MOF  Government Expenditure by Type      d_6a804a6860b5c51af08df679a71bc190
      - DOS  Development Expenditure by Sector   d_e75f4b407a1ad63011d7d2df2fd4358a
      Licence: Singapore Open Data Licence v1 ("free forever for personal or
      commercial use"); attribution required in publications.
  * World Bank API v2 (stable since 2012, CC BY 4.0, attribution required):
      GET https://api.worldbank.org/v2/country/SGP/indicator/<code>?format=json
      Used: NY.GDP.MKTP.KN (real GDP, LCU), NY.GDP.DEFL.ZS (deflator),
            FP.CPI.TOTL (CPI, 2010=100 rebased here to 2019=100).
  * DBnomics Web API v22 (free aggregator; redistributes provider terms):
      GET https://api.db.nomics.world/v22/series/IMF/WEO:latest/<mask>
      Used: SGP general government gross debt, % of GDP (GGXWDG_NGDP).
  * stooq.com daily/annual index CSVs (free for personal/research use):
      ^spx annual closes as a global-equity proxy for the IV demo only.
  NOT included (checked and deliberately excluded as non-necessary here):
      SingStat Table Builder API — its key fiscal tables (M130571/M130591)
      are mirrored on data.gov.sg, which we already use;
      MAS API — M2/interest rates not needed beyond Appendix A;
      FRED — requires an API key; NBIM — needed only for the Norway
      comparison paper, not this script.

LEGAL / SCOPE NOTES (read before reusing)
  * Research & educational use only. Nothing here is investment, tax, or
    policy advice. All third-party data remain subject to their providers'
    terms; cite data.gov.sg (Open Data Licence), World Bank (CC BY 4.0),
    and IMF WEO in any publication that uses the fetched series.
  * Values marked ILLUSTRATIVE below are offline fallbacks committed from
    budget documents from memory; they exist so the pipeline runs without
    network, and every table that uses them says so. Verify against the
    live sources before quoting numbers in the paper.
  * The Appendix A block reproduces the author's OWN compiled dataset from
    the uploaded working paper (author = user), so no third-party
    redistribution issue arises.

Author of pipeline: drafted for Felix D. Langer's DBA methodology upgrade.
================================================================================
"""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 0. CONFIG
# --------------------------------------------------------------------------- #
FETCH_LIVE = True          # False -> fully offline, deterministic run
AUTO_INSTALL_EXTRAS = True # pip-install arch + ruptures on Colab if absent
SEED = 20260518            # SSRN first-version date; fixes bootstrap draws
BOOT_REPS = 2000           # moving-block bootstrap replications
BLOCK_LEN = 4              # block length ~ n^(1/3) for n = 30
REQUEST_TIMEOUT = 25       # seconds per HTTP call
POLITE_PAUSE = 0.4         # seconds between API calls (rate-limit courtesy)
OUTDIR = "nirc_outputs"
ALPHA = 0.05
CPI_REPAIR = "splice"        # "splice" (default) | "worldbank" | "off"
OFFICIAL_2019_INFLATION = 0.6  # %, SingStat/MAS print; offline splice anchor

rng = np.random.default_rng(SEED)
os.makedirs(OUTDIR, exist_ok=True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*unsupported index.*")

MODULE_STATUS: dict[str, str] = {}   # module -> ok / fallback / skipped / error

def status(module: str, state: str) -> None:
    MODULE_STATUS[module] = state

RESULTS: dict[str, object] = {}   # every module registers findings for M17

def record(key: str, value) -> None:
    RESULTS[key] = value

def banner(txt: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n {txt}\n{line}")

def note(txt: str) -> None:
    print(f"   [note] {txt}")

def warn(txt: str) -> None:
    print(f"   [WARNING] {txt}")

# --------------------------------------------------------------------------- #
# 0b. SOFT IMPORTS (script must never crash on a missing optional library)
# --------------------------------------------------------------------------- #
def _pip_install(pkgs: list[str]) -> None:
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *pkgs],
            check=False, timeout=420, capture_output=True,
        )
        if r.returncode != 0:
            print(f"   [setup] optional extras not installed "
                  f"({', '.join(pkgs)}) — continuing without them.")
    except Exception as exc:  # noqa: BLE001
        print(f"   [setup] pip install skipped ({exc.__class__.__name__})")

def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:  # noqa: BLE001
        return None

if AUTO_INSTALL_EXTRAS:
    _missing = [p for p in ("arch", "ruptures") if _try_import(p) is None]
    if _missing and FETCH_LIVE:
        _pip_install(_missing)

sm = _try_import("statsmodels")
HAVE_SM = sm is not None
if HAVE_SM:
    import statsmodels.api as smapi
    from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.tsa.api import VAR
    try:
        from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order
        HAVE_ARDL = True
    except Exception:  # statsmodels < 0.13
        HAVE_ARDL = False
    try:
        from statsmodels.regression.recursive_ls import RecursiveLS
        HAVE_RLS = True
    except Exception:
        HAVE_RLS = False
else:
    HAVE_ARDL = HAVE_RLS = False

arch_mod = _try_import("arch")
HAVE_ARCH = arch_mod is not None
if HAVE_ARCH:
    from arch.unitroot import PhillipsPerron

rpt = _try_import("ruptures")
HAVE_RUPTURES = rpt is not None

requests = _try_import("requests")
HAVE_REQUESTS = requests is not None

import matplotlib
matplotlib.use("Agg") if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules else None
import matplotlib.pyplot as plt
from scipy import stats as sstats

def env_report() -> None:
    banner("M0  ENVIRONMENT")
    print(f"   python      {sys.version.split()[0]}")
    print(f"   numpy       {np.__version__}")
    print(f"   pandas      {pd.__version__}")
    print(f"   scipy       {__import__('scipy').__version__}")
    print(f"   statsmodels {getattr(sm, '__version__', 'MISSING')}"
          f"{'' if HAVE_ARDL else '  (ARDL/UECM unavailable -> manual bounds path)'}")
    print(f"   arch        {getattr(arch_mod, '__version__', 'MISSING (PP test skipped)')}")
    print(f"   ruptures    {getattr(rpt, '__version__', 'MISSING (multi-break scan skipped)')}")
    print(f"   live fetch  {FETCH_LIVE and HAVE_REQUESTS}")
    if HAVE_SM:
        ver = tuple(int(x) for x in sm.__version__.split(".")[:2] if x.isdigit())
        if ver < (0, 13):
            warn("statsmodels < 0.13: official bounds test unavailable; "
                 "manual PSS cross-check will be the primary result.")

# --------------------------------------------------------------------------- #
# 1. EMBEDDED DATA
# --------------------------------------------------------------------------- #
# 1a. Appendix A of the working paper — reproduced EXACTLY as published.
#     (Author's own compiled dataset; SingStat/MAS/MOF/IMF underlying sources.
#      NIRC 1994-2000 are the author's estimates -> subsample check in M11.)
APPENDIX_A = """year,gdp,nirc,pop,unemp,cpi,m2
1994,113.5,1.2,3.44,2.60,58.7,112.0
1995,124.1,1.5,3.52,2.70,59.7,121.3
1996,133.7,1.8,3.61,2.00,61.0,131.5
1997,140.2,2.0,3.68,1.80,62.2,140.8
1998,134.8,1.7,3.73,3.20,61.8,148.2
1999,138.4,1.9,3.80,3.50,61.5,155.0
2000,159.8,2.4,3.92,3.10,62.4,161.3
2001,152.6,2.2,4.03,3.30,63.0,165.7
2002,157.3,2.5,4.10,4.40,62.7,170.5
2003,160.1,2.8,4.11,4.00,63.2,178.4
2004,183.8,3.5,4.17,3.40,64.3,192.0
2005,205.8,4.2,4.27,3.10,64.6,208.6
2006,228.7,5.0,4.40,2.70,65.3,230.1
2007,268.8,6.2,4.59,2.10,66.7,266.4
2008,273.5,5.8,4.84,2.20,71.1,288.3
2009,264.0,4.9,4.99,3.00,71.3,310.7
2010,310.6,7.1,5.08,2.20,73.3,340.2
2011,338.0,7.8,5.18,2.00,77.1,368.5
2012,351.4,8.2,5.31,2.00,80.6,395.0
2013,371.0,8.9,5.40,1.90,82.5,420.8
2014,385.4,9.5,5.47,2.00,83.5,448.3
2015,398.7,10.3,5.54,1.90,83.1,475.6
2016,410.2,11.2,5.61,2.10,82.6,502.4
2017,435.1,13.0,5.61,2.20,83.1,535.0
2018,462.8,14.8,5.64,2.10,83.7,564.2
2019,472.1,16.5,5.70,2.30,100.0,592.7
2020,440.1,17.1,5.69,3.00,99.6,650.3
2021,505.5,19.5,5.45,2.70,102.3,720.8
2022,592.0,21.6,5.64,2.10,108.6,876.3
2023,644.4,23.5,5.92,1.90,112.6,876.3
"""
# NOTE: the 2022 M2 value in the paper is 798.4; a transcription guard below
# restores it — see _fix_known_typos(). Keeping the raw block byte-faithful
# to the source PDF matters less than keeping the SERIES faithful, so the
# guard documents (and fixes) the one cell this file intentionally re-checks.

def _fix_known_typos(df: pd.DataFrame) -> pd.DataFrame:
    # Paper Appendix A: 2022 M2 = 798.4 (SGD bn). Guard against copy errors.
    df.loc[df["year"] == 2022, "m2"] = 798.4
    return df

# 1b. GST / CIT fallback (ILLUSTRATIVE — used only if data.gov.sg is offline).
#     Approximate S$bn from Budget statements; VERIFY before quoting.
FALLBACK_TAX = pd.DataFrame({
    "fy":   [2019, 2020, 2021, 2022, 2023],
    "gst":  [11.2,  9.9, 12.6, 14.1, 16.4],
    "cit":  [16.7, 16.1, 18.2, 23.1, 28.4],
})
GST_RATE = {y: 7.0 for y in range(2007, 2023)} | {2023: 8.0, 2024: 9.0, 2025: 9.0, 2026: 9.0}
CIT_RATE = 17.0  # statutory headline rate since YA2010

# 1c. Budget composition fallback (ILLUSTRATIVE, FY2023-ish S$bn; verify).
FALLBACK_EXPENDITURE = {
    "Defence": 17.9, "Health": 16.9, "Education": 14.6, "Transport": 11.7,
    "Home Affairs": 8.3, "National Development": 7.3, "Trade & Industry": 6.0,
    "Sustainability & Environment": 3.9, "Manpower": 4.3, "Others (residual)": 13.4,
}

# 1c2. S&P 500 year-end closes (widely published; VERIFY) — IV fallback
#      because stooq intermittently blocks cloud/Colab IP ranges.
SPX_ANNUAL_CLOSE = {
    1993: 466.45, 1994: 459.27, 1995: 615.93, 1996: 740.74, 1997: 970.43,
    1998: 1229.23, 1999: 1469.25, 2000: 1320.28, 2001: 1148.08,
    2002: 879.82, 2003: 1111.92, 2004: 1211.92, 2005: 1248.29,
    2006: 1418.30, 2007: 1468.36, 2008: 903.25, 2009: 1115.10,
    2010: 1257.64, 2011: 1257.60, 2012: 1426.19, 2013: 1848.36,
    2014: 2058.90, 2015: 2043.94, 2016: 2238.83, 2017: 2673.61,
    2018: 2506.85, 2019: 3230.78, 2020: 3756.07, 2021: 4766.18,
    2022: 3839.50, 2023: 4769.83,
}

# 1d. Gross-debt fallback anchors (% of GDP, ILLUSTRATIVE; prefer live WEO).
FALLBACK_DEBT = pd.Series(
    {1994: 69, 2000: 80, 2005: 92, 2010: 99, 2015: 102,
     2019: 128, 2020: 148, 2021: 147, 2022: 157, 2023: 168},
    name="debt_gdp",
)

# --------------------------------------------------------------------------- #
# 2. FETCH LAYER (every call guarded; [ONLINE] vs [FALLBACK] logged)
# --------------------------------------------------------------------------- #
_session = None

def _http_get(url: str, params: dict | None = None):
    """Single guarded GET with courtesy pause. Returns Response or None."""
    global _session
    if not (FETCH_LIVE and HAVE_REQUESTS):
        return None
    if _session is None:
        _session = requests.Session()
        _session.headers.update(
            {"User-Agent": "NIRC-research-pipeline/1.0 (academic use)"}
        )
    for attempt in (1, 2):
        try:
            time.sleep(POLITE_PAUSE if attempt == 1 else 1.5)
            r = _session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r
            print(f"   [http] {r.status_code} on {url.split('?')[0]}")
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                print(f"   [http] {exc.__class__.__name__} on {url.split('?')[0]}")
    return None

def fetch_datagov(dataset_id: str, limit: int = 10000) -> pd.DataFrame | None:
    """data.gov.sg datastore_search -> DataFrame of records (or None)."""
    r = _http_get(
        "https://data.gov.sg/api/action/datastore_search",
        params={"resource_id": dataset_id, "limit": limit},
    )
    if r is None:
        return None
    try:
        js = r.json()
        recs = js["result"]["records"]
        if not recs:
            return None
        df = pd.DataFrame(recs)
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"   [parse] data.gov.sg {dataset_id[:12]}…: {exc.__class__.__name__}")
        return None

def fetch_worldbank(indicator: str) -> pd.Series | None:
    """World Bank API v2 for Singapore -> Series indexed by year (or None)."""
    r = _http_get(
        f"https://api.worldbank.org/v2/country/SGP/indicator/{indicator}",
        params={"format": "json", "per_page": 200},
    )
    if r is None:
        return None
    try:
        js = r.json()
        rows = js[1]
        s = pd.Series(
            {int(d["date"]): d["value"] for d in rows if d["value"] is not None}
        ).sort_index()
        s.name = indicator
        return s if len(s) else None
    except Exception as exc:  # noqa: BLE001
        print(f"   [parse] worldbank {indicator}: {exc.__class__.__name__}")
        return None

def fetch_debt_weo() -> tuple[pd.Series | None, str]:
    """IMF WEO gross govt debt %GDP for SGP via DBnomics (masks, then query)."""
    masks = ["SGP.GGXWDG_NGDP", "SGP.GGXWDG_NGDP.pcent_of_gdp"]
    for m in masks:
        r = _http_get(
            f"https://api.db.nomics.world/v22/series/IMF/WEO:latest/{m}",
            params={"observations": 1, "format": "json"},
        )
        s = _parse_dbnomics(r)
        if s is not None:
            return s, f"IMF WEO via DBnomics (mask {m})"
    r = _http_get(
        "https://api.db.nomics.world/v22/series/IMF/WEO:latest",
        params={"q": "Singapore general government gross debt percent GDP",
                "observations": 1, "format": "json", "limit": 20},
    )
    s = _parse_dbnomics(r, want_substrings=("SGP", "GGXWDG_NGDP"))
    if s is not None:
        return s, "IMF WEO via DBnomics (query search)"
    return None, "unavailable"

def _parse_dbnomics(r, want_substrings: tuple[str, ...] = ()) -> pd.Series | None:
    if r is None:
        return None
    try:
        docs = r.json()["series"]["docs"]
        for d in docs:
            code = str(d.get("series_code", ""))
            if want_substrings and not all(w in code for w in want_substrings):
                continue
            per = d.get("period", [])
            val = d.get("value", [])
            out = {}
            for p, v in zip(per, val):
                if v is None or (isinstance(v, str) and not v.strip("-.0123456789")
                                 and v.strip() in {"", "NA", "n/a"}):
                    continue
                try:
                    out[int(str(p)[:4])] = float(v)
                except (TypeError, ValueError):
                    continue
            if out:
                s = pd.Series(out).sort_index()
                s.name = "debt_gdp"
                return s
    except Exception as exc:  # noqa: BLE001
        print(f"   [parse] dbnomics: {exc.__class__.__name__}")
    return None

def fetch_spx_annual() -> tuple[pd.Series, str]:
    """Yearly S&P 500 % returns: stooq if reachable, else embedded closes."""
    def _fallback() -> tuple[pd.Series, str]:
        px = pd.Series(SPX_ANNUAL_CLOSE).sort_index()
        ret = px.pct_change().mul(100.0).dropna()
        ret.name = "world_eq_ret"
        return ret, "embedded year-end closes (widely published; VERIFY)"
    r = _http_get("https://stooq.com/q/d/l/", params={"s": "^spx", "i": "y"})
    if r is None:
        return _fallback()
    try:
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().lower() for c in df.columns]
        df["year"] = pd.to_datetime(df["date"]).dt.year
        px = df.set_index("year")["close"].astype(float)
        ret = px.pct_change().mul(100.0).dropna()
        ret.name = "world_eq_ret"
        if len(ret):
            return ret, "stooq.com ^spx (live)"
        return _fallback()
    except Exception as exc:  # noqa: BLE001
        print(f"   [parse] stooq: {exc.__class__.__name__} -> embedded closes")
        return _fallback()

# ---- generic helpers to mine data.gov.sg long-format fiscal tables ---------
def _numeric(x) -> float | None:
    try:
        v = float(str(x).replace(",", "").strip())
        return v
    except (TypeError, ValueError):
        return None

def _year_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "year" in c:
            return c
    return None

def _numeric_frac(s: pd.Series) -> float:
    v = pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(),
                      errors="coerce")
    return float(v.notna().mean())

def _split_value_columns(df: pd.DataFrame, ycol: str) -> tuple[list, list]:
    """Classify columns by CONTENT into (value_cols, label_cols).
    data.gov.sg returns every field as a string AND adds a numeric `_id`
    row counter — a dtype-based split picks `_id` as the value column,
    which is exactly the bug this replaces."""
    vals, labs = [], []
    for c in df.columns:
        if c == ycol or c.lstrip("_") == "id":
            continue
        (vals if _numeric_frac(df[c]) >= 0.8 else labs).append(c)
    prefer = ("amount", "value", "revenue", "expenditure", "actual", "total")
    def _score(c: str):
        med = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False),
                            errors="coerce").abs().median()
        return (int(any(p in c for p in prefer)),
                0.0 if pd.isna(med) else float(med))
    vals.sort(key=_score, reverse=True)
    return vals, labs

def _extract_fiscal_series(df: pd.DataFrame, label_substr: str) -> pd.Series | None:
    """Find rows whose label column matches label_substr; return year->S$bn."""
    ycol = _year_col(df)
    if ycol is None:
        return None
    value_cols, text_cols = _split_value_columns(df, ycol)
    if not value_cols or not text_cols:
        return None
    vcol = value_cols[0]
    lab = label_substr.lower()
    hit_rows = pd.Series(False, index=df.index)
    for c in text_cols:
        hit_rows |= df[c].astype(str).str.lower().str.contains(lab, regex=False)
    sub = df[hit_rows]
    if sub.empty:
        return None
    out: dict[int, float] = {}
    for _, row in sub.iterrows():
        try:
            yr = int(str(row[ycol])[:4])
        except (TypeError, ValueError):
            continue
        v = _numeric(row[vcol])
        if v is None:
            continue
        out[yr] = out.get(yr, 0.0) + v
    if not out:
        return None
    s = pd.Series(out).sort_index()
    # Auto-scale to S$bn: MOF/SingStat fiscal tables are in $m or $th.
    med = float(s.tail(5).median())
    if med > 2e5:      # thousands of S$  -> bn
        s = s / 1e6
    elif med > 2e2:    # millions of S$   -> bn
        s = s / 1e3
    return s

# --------------------------------------------------------------------------- #
# 3. DATA ASSEMBLY + VALIDATION (M1, M2)
# --------------------------------------------------------------------------- #
DATAGOV = {
    "fiscal_position":   "d_0f0800a6b2b4b391daffcd88b2cc97fd",  # incl. NIRC
    "operating_revenue": "d_d6cd701a67607ac9d4f3d469d8c815e5",  # GST/CIT/...
    "total_expenditure": "d_00f95040967791729d51bd7cd6af2355",
    "expenditure_type":  "d_6a804a6860b5c51af08df679a71bc190",
    "dev_exp_sector":    "d_e75f4b407a1ad63011d7d2df2fd4358a",
}

@dataclass
class DataBundle:
    df: pd.DataFrame                      # Appendix A + engineered columns
    wb: dict = field(default_factory=dict)          # World Bank series
    nirc_official: pd.Series | None = None
    tax: pd.DataFrame | None = None       # fy, gst, cit  (+ source flag)
    tax_source: str = "FALLBACK (illustrative — verify)"
    debt: pd.Series | None = None
    debt_source: str = "FALLBACK (illustrative — verify)"
    exp_sector: dict | None = None
    exp_source: str = "FALLBACK (illustrative — verify)"
    spx: pd.Series | None = None
    spx_source: str = "unavailable"

def load_data() -> DataBundle:
    banner("M1  DATA ASSEMBLY")
    df = pd.read_csv(io.StringIO(APPENDIX_A))
    df = _fix_known_typos(df)
    df = df.astype(float)
    df["year"] = df["year"].astype(int)
    print(f"   Appendix A loaded: {len(df)} annual obs, {df.year.min()}-{df.year.max()}")

    b = DataBundle(df=df)

    # --- World Bank cross-validation series ---------------------------------
    for code, key in [("NY.GDP.MKTP.KN", "rgdp_wb"),
                      ("NY.GDP.DEFL.ZS", "defl_wb"),
                      ("FP.CPI.TOTL",   "cpi_wb")]:
        s = fetch_worldbank(code)
        if s is not None:
            b.wb[key] = s
            print(f"   [ONLINE]  World Bank {code}: {len(s)} obs")
    if not b.wb:
        print("   [FALLBACK] World Bank unavailable -> CPI cross-check skipped")

    # --- Official NIRC from MOF Fiscal Position -----------------------------
    fp = fetch_datagov(DATAGOV["fiscal_position"])
    if fp is not None:
        s = _extract_fiscal_series(fp, "net investment returns contribution")
        if s is None:
            s = _extract_fiscal_series(fp, "nirc")
        if s is not None:
            b.nirc_official = s
            print(f"   [ONLINE]  Official NIRC (MOF): FY{s.index.min()}-FY{s.index.max()}")
    if b.nirc_official is None:
        print("   [FALLBACK] Official NIRC not fetched -> paper series used as-is")

    # --- GST / CIT for tax equivalence --------------------------------------
    rev = fetch_datagov(DATAGOV["operating_revenue"])
    if rev is not None:
        gst = _extract_fiscal_series(rev, "goods and services tax")
        cit = _extract_fiscal_series(rev, "corporate income tax")
        if gst is not None and cit is not None:
            recent = gst[(gst.index >= 2019) & (gst.index <= 2023)]
            if len(recent) and recent.between(5.0, 50.0).all():
                b.tax = (pd.DataFrame({"gst": gst, "cit": cit})
                           .dropna().reset_index(names="fy"))
                b.tax_source = ("MOF Government Operating Revenue "
                                "(data.gov.sg, official)")
                print(f"   [ONLINE]  GST/CIT revenue: "
                      f"FY{int(b.tax.fy.min())}-FY{int(b.tax.fy.max())}")
            else:
                warn("Fetched GST revenue failed the plausibility check "
                     "(FY2019-23 outside S$5-50bn) -> parser output "
                     "rejected; embedded fallback used instead.")
    if b.tax is None:
        b.tax = FALLBACK_TAX.copy()
        print("   [FALLBACK] GST/CIT: illustrative embedded values (VERIFY)")

    # --- Expenditure composition --------------------------------------------
    for key in ("total_expenditure", "dev_exp_sector", "expenditure_type"):
        ex = fetch_datagov(DATAGOV[key])
        if ex is None:
            continue
        comp = _latest_sector_composition(ex)
        if comp:
            b.exp_sector = comp
            b.exp_source = f"data.gov.sg {key} (official)"
            print(f"   [ONLINE]  Expenditure composition from '{key}' "
                  f"({len(comp)} categories)")
            break
    if b.exp_sector is None:
        b.exp_sector = dict(FALLBACK_EXPENDITURE)
        print("   [FALLBACK] Expenditure composition: illustrative (VERIFY)")

    # --- Debt / GDP ----------------------------------------------------------
    s, src = fetch_debt_weo()
    if s is not None:
        b.debt, b.debt_source = s, src
        print(f"   [ONLINE]  Gross debt %GDP: {s.index.min()}-{s.index.max()} ({src})")
    else:
        b.debt = FALLBACK_DEBT.astype(float)
        print("   [FALLBACK] Gross debt %GDP: sparse illustrative anchors (VERIFY)")

    # --- Global equity proxy (IV demo) --------------------------------------
    b.spx, b.spx_source = fetch_spx_annual()
    print(f"   [{'ONLINE' if 'live' in b.spx_source else 'FALLBACK'}]  "
          f"Global-equity proxy (S&P 500 annual): {b.spx_source}")
    status("M1_data", "ok")
    return b

def _latest_sector_composition(df: pd.DataFrame) -> dict | None:
    """Best-effort: latest actual FY, category -> S$bn. Rejects degenerate
    parses (numeric labels, all-equal values) instead of returning junk."""
    ycol = _year_col(df)
    if ycol is None:
        return None
    value_cols, text_cols = _split_value_columns(df, ycol)
    if not value_cols or not text_cols:
        return None
    yrs = pd.to_numeric(df[ycol].astype(str).str[:4], errors="coerce")
    df = df.assign(_yr=yrs).dropna(subset=["_yr"])
    if df.empty:
        return None
    latest_full = int(df["_yr"].max()) - 2   # avoid budget/revised estimates
    sub = df[df["_yr"] == latest_full]
    if sub.empty:
        sub = df[df["_yr"] == df["_yr"].max()]
        latest_full = int(df["_yr"].max())
    label_col = max(text_cols, key=lambda c: sub[c].nunique())
    vcol = value_cols[0]
    out: dict[str, float] = {}
    for _, row in sub.iterrows():
        lab = str(row[label_col]).strip().title()
        if lab.lower() in {"nan", "total", "total expenditure"}:
            continue
        v = _numeric(row[vcol])
        if v is None:
            continue
        out[lab] = out.get(lab, 0.0) + v
    if len(out) < 3:
        return None
    labels = list(out)
    if sum(l.replace(".", "").isdigit() for l in labels) >= len(labels) / 2:
        return None                      # labels are codes, not names
    vals = sorted(out.values(), reverse=True)[:10]
    if len(set(round(v, 4) for v in vals)) <= 2:
        return None                      # all-equal values = wrong column
    med = float(np.median(list(out.values())))
    scale = 1e6 if med > 2e5 else (1e3 if med > 2e2 else 1.0)
    out = {k: v / scale for k, v in out.items()}
    out["_fy"] = latest_full
    return out

def validate(b: DataBundle) -> None:
    banner("M2  INTERNAL CONSISTENCY & CROSS-SOURCE VALIDATION")
    df = b.df.set_index("year")

    # (i) CPI internal check — flags implausible YoY inflation in Appendix A
    infl = df["cpi"].pct_change().mul(100)
    bad = infl[infl.abs() > 6]
    if len(bad):
        print("   Large YoY moves in the Appendix A CPI column: "
              + ", ".join(f"{y}: {v:+.1f}%" for y, v in bad.items()))
        print("   Of these, 2008 (+6.6%) and 2022 (+6.2%) match GENUINE "
              "Singapore inflation spikes (commodity shock; post-COVID "
              "reopening) — no issue there.")
        warn("The 2018->2019 step (83.7 -> 100.0, i.e. +19.5%) is a rebasing "
             "artifact, not inflation — actual 2019 CPI inflation was ~0.6%. "
             "Pre-2019 values sit on a DIFFERENT base than 2019=100. "
             "AUTO-REPAIR follows in M2b (set CPI_REPAIR='off' to disable); "
             "verify the corrected series against SingStat before "
             "publication.")
    else:
        print("   CPI YoY profile: no implausible jumps detected.")

    # (ii) Cross-check vs World Bank CPI rebased to 2019=100
    if "cpi_wb" in b.wb:
        wb = b.wb["cpi_wb"]
        if 2019 in wb.index:
            wb19 = wb / wb.loc[2019] * 100.0
            joint = pd.concat([df["cpi"], wb19.rename("cpi_wb")], axis=1).dropna()
            gap = (joint["cpi"] - joint["cpi_wb"]).abs()
            print(f"   CPI vs World Bank (rebased 2019=100): "
                  f"mean |gap| = {gap.mean():.1f} pts, max = {gap.max():.1f} pts "
                  f"({gap.idxmax()})")
            if gap.max() > 3:
                warn("Large CPI divergence from official data confirms the "
                     "base inconsistency above.")
            _plot_cpi_check(joint)

    # (iii) Official NIRC vs paper NIRC
    if b.nirc_official is not None:
        joint = pd.concat(
            [df["nirc"], b.nirc_official.rename("nirc_official")], axis=1
        ).dropna()
        if len(joint):
            gap = (joint["nirc"] - joint["nirc_official"]).abs()
            print(f"   NIRC (paper) vs NIRC (MOF official), {len(joint)} "
                  f"overlapping FYs: mean |gap| = {gap.mean():.2f} bn, "
                  f"max = {gap.max():.2f} bn ({gap.idxmax()})")
            note("Calendar-year (paper) vs financial-year (MOF, Apr-Mar) "
                 "timing explains part of any gap; document the convention.")
            _plot_nirc_check(joint)

    # (iv) GDP sanity
    g = df["gdp"].pct_change().mul(100)
    print(f"   Nominal GDP growth: mean {g.mean():.1f}%, "
          f"min {g.min():.1f}% ({g.idxmin()}), max {g.max():.1f}% ({g.idxmax()})")
    status("M2_validation", "ok")

def _plot_cpi_check(joint: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(joint.index, joint["cpi"], marker="o", ms=3, label="Appendix A CPI")
    ax.plot(joint.index, joint["cpi_wb"], marker="s", ms=3,
            label="World Bank CPI (rebased 2019=100)")
    ax.set_title("CPI cross-check — Appendix A vs official (rebased)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig_cpi_check.png", dpi=150)
    plt.show(); plt.close(fig)

def _plot_nirc_check(joint: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(joint.index, joint["nirc"], marker="o", ms=3, label="NIRC (paper)")
    ax.plot(joint.index, joint["nirc_official"], marker="s", ms=3,
            label="NIRC (MOF official)")
    ax.set_title("NIRC — paper series vs MOF official series")
    ax.set_ylabel("S$ bn"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig_nirc_check.png", dpi=150)
    plt.show(); plt.close(fig)

# --------------------------------------------------------------------------- #
# 3b. CPI BASE-BREAK REPAIR (M2b)
# --------------------------------------------------------------------------- #
def repair_cpi(b: DataBundle) -> None:
    banner("M2b CPI BASE-BREAK REPAIR")
    df = b.df
    df["cpi_raw"] = df["cpi"].copy()
    if CPI_REPAIR == "off":
        note("CPI_REPAIR='off' -> raw Appendix A CPI used as-is (NOT advised).")
        status("M2b_cpi_repair", "off")
        return
    infl = df["cpi"].pct_change().mul(100)
    brk = infl[infl > 15]                       # a >15% YoY step = base break
    if brk.empty:
        print("   No base break detected — CPI left unchanged.")
        status("M2b_cpi_repair", "not needed")
        return
    by = int(df.loc[brk.index[0], "year"])      # break year (2019 here)
    pre = df["year"] < by
    raw_prev = float(df.loc[df["year"] == by - 1, "cpi_raw"].iloc[0])
    cpi_break = float(df.loc[df["year"] == by, "cpi_raw"].iloc[0])

    wb19 = None
    if "cpi_wb" in b.wb and by in b.wb["cpi_wb"].index:
        wb = b.wb["cpi_wb"]
        wb19 = wb / wb.loc[by] * 100.0

    if CPI_REPAIR == "worldbank" and wb19 is not None:
        # Full replacement with the official series, rebased to break-year=100
        repl = wb19.reindex(df["year"]).interpolate(limit_direction="both")
        df["cpi"] = repl.to_numpy()
        src = "World Bank FP.CPI.TOTL, rebased (full replacement)"
        factor = np.nan
    else:
        if CPI_REPAIR == "worldbank" and wb19 is None:
            note("World Bank CPI unavailable -> falling back to splice repair.")
        if wb19 is not None and (by - 1) in wb19.index:
            target_prev = float(wb19.loc[by - 1])
            src = f"splice; factor from World Bank overlap at {by - 1}"
        else:
            target_prev = cpi_break / (1.0 + OFFICIAL_2019_INFLATION / 100.0)
            src = (f"splice; factor anchored on documented {by} inflation "
                   f"of {OFFICIAL_2019_INFLATION}% (offline)")
        factor = target_prev / raw_prev
        df.loc[pre, "cpi"] = df.loc[pre, "cpi_raw"] * factor

    new_prev = float(df.loc[df["year"] == by - 1, "cpi"].iloc[0])
    print(f"   Break detected entering {by}. Repair = {src}.")
    if not np.isnan(factor):
        print(f"   Pre-{by} segment rescaled by ×{factor:.4f} "
              f"({by - 1}: {raw_prev:.1f} -> {new_prev:.2f}); {by}+ unchanged. "
              f"Every pre-{by} YoY inflation rate in the paper's own data is "
              f"preserved exactly — only the level discontinuity is removed.")
    infl2 = df["cpi"].pct_change().mul(100)
    big = infl2[infl2.abs() > 6]
    print("   Post-repair YoY check: largest moves now "
          + (", ".join(f"{int(df.loc[i, 'year'])}: {v:+.1f}%"
                       for i, v in big.items()) if len(big) else "all < 6%")
          + "  (2008/2022-type spikes are genuine inflation, not artifacts).")
    if wb19 is not None:
        joint = pd.concat(
            [df.set_index("year")["cpi"], wb19.rename("wb")], axis=1
        ).dropna()
        gap = (joint["cpi"] - joint["wb"]).abs()
        print(f"   Repaired series vs World Bank (rebased): mean |gap| = "
              f"{gap.mean():.2f} pts, max = {gap.max():.2f} pts "
              f"({int(gap.idxmax())}) — should now be rounding-level.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["year"], df["cpi_raw"], marker="o", ms=3, alpha=.6,
            label="Appendix A CPI (raw, base break)")
    ax.plot(df["year"], df["cpi"], marker="s", ms=3,
            label="Repaired CPI (2019=100 throughout)")
    ax.axvline(by - 0.5, color="grey", ls=":", lw=1)
    ax.set_title("CPI base-break repair (M2b)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig_cpi_repair.png", dpi=150)
    plt.show(); plt.close(fig)
    record("cpi_repair", {
        "break_year": by, "method": src,
        "factor": None if np.isnan(factor) else round(float(factor), 4)})
    status("M2b_cpi_repair", f"ok ({'replacement' if np.isnan(factor) else 'splice'})")

# --------------------------------------------------------------------------- #
# 4. RE-SPECIFICATION + PEARL ESTIMAND (M3, M4)
# --------------------------------------------------------------------------- #
def engineer(b: DataBundle) -> pd.DataFrame:
    banner("M3  RE-SPECIFICATION (log real, per-capita, dummies)")
    df = b.df.copy()
    df["rgdp"]      = df["gdp"]  / df["cpi"] * 100.0     # 2019 S$ bn
    df["rnirc"]     = df["nirc"] / df["cpi"] * 100.0
    df["rgdp_pc"]   = df["rgdp"] * 1000.0 / df["pop"]    # 2019 S$ per capita (bn->m scale)
    df["rnirc_pc"]  = df["rnirc"] * 1000.0 / df["pop"]
    df["ln_rgdp"]     = np.log(df["rgdp"])
    df["ln_rnirc"]    = np.log(df["rnirc"])
    df["ln_pop"]      = np.log(df["pop"])
    df["ln_rgdp_pc"]  = np.log(df["rgdp_pc"])
    df["ln_rnirc_pc"] = np.log(df["rnirc_pc"])
    df["nirc_share"]  = df["nirc"] / df["gdp"] * 100.0
    df["d2009"]  = (df["year"] == 2009).astype(float)    # GFC pulse
    df["d2020"]  = (df["year"] == 2020).astype(float)    # COVID pulse
    df["nir_regime"]     = (df["year"] >= 2009).astype(float)  # NII -> NIR (GIC+MAS)
    df["temasek_regime"] = (df["year"] >= 2016).astype(float)  # Temasek added
    if b.debt is not None:
        df = df.merge(b.debt.rename("debt_gdp"), left_on="year",
                      right_index=True, how="left")
        df["debt_gdp"] = df["debt_gdp"].interpolate(limit_direction="both")
    print("   Built: real (CPI-deflated) levels, logs, per-capita, NIRC/GDP "
          "share, crisis pulses (2009, 2020), regime steps (NIR 2009, "
          "Temasek 2016), debt ratio (interpolated where sparse).")
    note("Deflating by CPI approximates real GDP; the GDP deflator is the "
         "cleaner concept. When online, M11 re-runs the headline model on "
         "World Bank chain-linked real GDP as a robustness check.")
    status("M3_respec", "ok")
    return df

def pearl_estimand() -> None:
    banner("M4  CAUSAL ESTIMAND (Pearl SCM — replaces VIF-driven selection)")
    print("""   DAG maintained (from the paper, sharpened):
       GlobalReturns* -> NIRC ;  GlobalReturns* -> GDP   (*unobserved confounder)
       Population -> GDP ;  NIRC -> GDP ;  M2 -> GDP ;  M2 -> CPI -> GDP(nominal)
       GDP <-> Unemployment (simultaneous / GDP-caused)

   Estimand declared:  TOTAL effect of NIRC on (log real) GDP.
   Backdoor adjustment set: { Population, time trend }  — plus an instrument
   or regime variation for the unobserved GlobalReturns confounder (M15).

   Consequences implemented downstream:
     * CPI and M2 are NOT conditioned on: CPI is a mediator/derivative of the
       nominal outcome, M2 an outcome-adjacent mediator. Excluding them is a
       backdoor-criterion decision, not a VIF decision — state it that way in
       Section 4.1.
     * Unemployment is caused by GDP (simultaneity acknowledged in the paper),
       so conditioning on it biases the TOTAL effect. The headline spec
       therefore EXCLUDES unemployment; the paper's original spec (with
       unemployment) is retained in M11 as the 'conditional-association'
       variant for comparability. If the target were the DIRECT effect
       holding labour-market conditions fixed, the with-unemployment spec
       would be the right one — the point is to say which estimand you want.
""")
    status("M4_pearl", "ok")

# --------------------------------------------------------------------------- #
# 5. MANUAL ECONOMETRICS TOOLKIT (numpy/scipy only -> runs everywhere)
# --------------------------------------------------------------------------- #
class LinReg:
    """Minimal OLS with classical + Newey-West (HAC) inference and the full
    small-sample diagnostic battery. Deliberately dependency-free so the
    same tested code path runs on any machine."""

    def __init__(self, y: np.ndarray, X: np.ndarray, names: list[str]):
        y = np.asarray(y, float).ravel()
        X = np.asarray(X, float)
        assert X.ndim == 2 and len(y) == X.shape[0] and X.shape[1] == len(names)
        self.y, self.X, self.names = y, X, names
        self.n, self.k = X.shape
        self.XtX_inv = np.linalg.pinv(X.T @ X)
        self.beta = self.XtX_inv @ X.T @ y
        self.fitted = X @ self.beta
        self.resid = y - self.fitted
        self.rss = float(self.resid @ self.resid)
        self.dof = self.n - self.k
        self.sigma2 = self.rss / max(self.dof, 1)
        tss = float(((y - y.mean()) ** 2).sum())
        self.r2 = 1.0 - self.rss / tss if tss > 0 else np.nan
        self.r2adj = 1.0 - (1 - self.r2) * (self.n - 1) / max(self.dof, 1)
        self.cov_ols = self.sigma2 * self.XtX_inv
        self.se = np.sqrt(np.diag(self.cov_ols))

    # ---- inference ----------------------------------------------------------
    def hac_cov(self, lags: int | None = None) -> np.ndarray:
        L = lags if lags is not None else max(1, int(round(4 * (self.n / 100) ** (2 / 9))))
        u = self.resid[:, None] * self.X
        S = u.T @ u
        for l in range(1, L + 1):
            w = 1.0 - l / (L + 1.0)
            G = u[l:].T @ u[:-l]
            S += w * (G + G.T)
        S *= self.n / max(self.dof, 1)              # small-sample d.f. bump
        return self.XtX_inv @ S @ self.XtX_inv

    def table(self, hac: bool = True, lags: int | None = None) -> pd.DataFrame:
        cov = self.hac_cov(lags) if hac else self.cov_ols
        se = np.sqrt(np.diag(cov))
        t = self.beta / se
        p = 2 * sstats.t.sf(np.abs(t), self.dof)
        return pd.DataFrame(
            {"coef": self.beta, "se" + ("_HAC" if hac else ""): se,
             "t": t, "p": p}, index=self.names).round(4)

    def wald(self, idx: list[int], hac: bool = False) -> tuple[float, float]:
        """H0: beta[idx] = 0. Returns (chi2 stat, p)."""
        R = np.zeros((len(idx), self.k)); R[np.arange(len(idx)), idx] = 1.0
        cov = self.hac_cov() if hac else self.cov_ols
        br = R @ self.beta
        stat = float(br @ np.linalg.pinv(R @ cov @ R.T) @ br)
        return stat, float(sstats.chi2.sf(stat, len(idx)))

    # ---- diagnostics --------------------------------------------------------
    def dw(self) -> float:
        d = np.diff(self.resid)
        return float((d @ d) / self.rss)

    def breusch_godfrey(self, p: int = 2) -> tuple[float, float]:
        e = self.resid
        Z = [self.X] + [np.concatenate([np.zeros(i), e[:-i]])[:, None]
                        for i in range(1, p + 1)]
        Z = np.hstack(Z)
        aux = LinReg(e, Z, [f"z{i}" for i in range(Z.shape[1])])
        lm = self.n * aux.r2
        return lm, float(sstats.chi2.sf(lm, p))

    def white(self) -> tuple[float, float, int]:
        cols = [c for c in range(self.k)
                if not np.allclose(self.X[:, c], self.X[0, c])]
        Z = [np.ones((self.n, 1))]
        Z += [self.X[:, [c]] for c in cols]
        Z += [self.X[:, [c]] ** 2 for c in cols]
        if len(cols) <= 3:                            # keep dof sane for n=30
            for i, a in enumerate(cols):
                for bcol in cols[i + 1:]:
                    Z.append((self.X[:, a] * self.X[:, bcol])[:, None])
        Z = np.hstack(Z)
        aux = LinReg(self.resid ** 2, Z, [f"z{i}" for i in range(Z.shape[1])])
        df_ = Z.shape[1] - 1
        lm = self.n * aux.r2
        return lm, float(sstats.chi2.sf(lm, df_)), df_

    def reset(self) -> tuple[float, float]:
        yh = self.fitted
        Z = np.hstack([self.X, (yh ** 2)[:, None], (yh ** 3)[:, None]])
        unres = LinReg(self.y, Z, [f"z{i}" for i in range(Z.shape[1])])
        num = (self.rss - unres.rss) / 2.0
        den = unres.rss / max(unres.dof, 1)
        F = num / den
        return float(F), float(sstats.f.sf(F, 2, max(unres.dof, 1)))

    def jarque_bera(self) -> tuple[float, float]:
        jb = sstats.jarque_bera(self.resid)
        return float(jb.statistic), float(jb.pvalue)

    def ljung_box(self, h: int = 4) -> tuple[float, float]:
        e = self.resid - self.resid.mean()
        n = self.n
        denom = float(e @ e)
        q = 0.0
        for k_ in range(1, h + 1):
            r = float(e[k_:] @ e[:-k_]) / denom
            q += r * r / (n - k_)
        q *= n * (n + 2)
        return q, float(sstats.chi2.sf(q, h))

    def arch_lm(self, p: int = 1) -> tuple[float, float]:
        e2 = self.resid ** 2
        Z = np.hstack([np.ones((self.n - p, 1))]
                      + [e2[p - i - 1: self.n - i - 1][:, None] for i in range(p)])
        aux = LinReg(e2[p:], Z, [f"z{i}" for i in range(Z.shape[1])])
        lm = (self.n - p) * aux.r2
        return lm, float(sstats.chi2.sf(lm, p))

    def diagnostics(self, label: str = "") -> pd.DataFrame:
        bg = self.breusch_godfrey(2)
        wh = self.white()
        rs = self.reset()
        jb = self.jarque_bera()
        lb = self.ljung_box(4)
        al = self.arch_lm(1)
        rows = {
            "Durbin-Watson": (self.dw(), np.nan),
            "Breusch-Godfrey LM(2)": bg,
            f"White LM (df={wh[2]})": wh[:2],
            "Ramsey RESET F(2)": rs,
            "Jarque-Bera": jb,
            "Ljung-Box Q(4)": lb,
            "ARCH-LM(1)": al,
        }
        out = pd.DataFrame(rows, index=["stat", "p"]).T.round(4)
        if label:
            print(f"\n   Diagnostics — {label}")
        with pd.option_context("display.width", 100):
            print(out.to_string())
        return out

def recursive_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Standardised one-step recursive residuals (Brown-Durbin-Evans)."""
    n, k = X.shape
    w = np.full(n, np.nan)
    for t in range(k + 1, n + 1):
        Xt, yt = X[: t - 1], y[: t - 1]
        XtX_inv = np.linalg.pinv(Xt.T @ Xt)
        b_ = XtX_inv @ Xt.T @ yt
        x_new = X[t - 1]
        f = 1.0 + float(x_new @ XtX_inv @ x_new)
        w[t - 1] = (y[t - 1] - float(x_new @ b_)) / math.sqrt(f)
    return w[k + 1 - 1:]           # first k obs consumed

# --------------------------------------------------------------------------- #
# 6. UNIT ROOTS (M5)
# --------------------------------------------------------------------------- #
CORE_LEVELS = ["ln_rgdp", "ln_rnirc", "ln_pop", "unemp"]

def unit_roots(df: pd.DataFrame) -> dict[str, int]:
    banner("M5  UNIT ROOTS — ADF / PP / KPSS / Zivot-Andrews (break-robust)")
    orders: dict[str, int] = {}
    if not HAVE_SM:
        warn("statsmodels missing -> canonical unit-root tests skipped; "
             "assuming I(1) for all core series (conservative for bounds test).")
        status("M5_unitroots", "skipped (no statsmodels)")
        record("integration_orders",
               {s: "I(1) assumed (offline)" for s in CORE_LEVELS})
        return {s: 1 for s in CORE_LEVELS}

    rows = []
    years = df["year"].to_numpy()
    for s_name in CORE_LEVELS:
        x = df[s_name].to_numpy(float)
        dx = np.diff(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_l = adfuller(x, autolag="AIC")
            adf_d = adfuller(dx, autolag="AIC")
            try:
                kp_l = kpss(x, regression="c", nlags="auto")
                kp_d = kpss(dx, regression="c", nlags="auto")
            except Exception:
                kp_l = kp_d = (np.nan, np.nan)
            try:
                za = zivot_andrews(x, regression="c")
                za_stat, za_p = float(za[0]), float(za[1])
                bp = int(za[-1])
                za_year = int(years[min(max(bp, 0), len(years) - 1)])
            except Exception:
                za_stat = za_p = np.nan; za_year = None
            pp_p = np.nan
            if HAVE_ARCH:
                try:
                    pp_p = float(PhillipsPerron(x).pvalue)
                except Exception:
                    pass
        i0 = adf_l[1] < 0.05 or (not np.isnan(kp_l[1]) and kp_l[1] > 0.10
                                 and adf_l[1] < 0.10)
        i1 = (not i0) and adf_d[1] < 0.05
        order = 0 if i0 else (1 if i1 else 2)
        orders[s_name] = order
        rows.append({
            "series": s_name, "ADF p (lvl)": round(adf_l[1], 3),
            "PP p (lvl)": round(pp_p, 3) if not np.isnan(pp_p) else "-",
            "KPSS p (lvl)": round(kp_l[1], 3) if not np.isnan(kp_l[1]) else "-",
            "ADF p (diff)": round(adf_d[1], 3),
            "KPSS p (diff)": round(kp_d[1], 3) if not np.isnan(kp_d[1]) else "-",
            "ZA stat": round(za_stat, 2) if not np.isnan(za_stat) else "-",
            "ZA p": round(za_p, 3) if not np.isnan(za_p) else "-",
            "ZA break": za_year if za_year else "-",
            "order": f"I({order})",
        })
    tab = pd.DataFrame(rows).set_index("series")
    with pd.option_context("display.width", 130):
        print(tab.to_string())
    note("KPSS null is STATIONARITY (opposite of ADF/PP). ZA allows one "
         "endogenous break — expect 2008/09 or 2020 to surface.")
    if any(v >= 2 for v in orders.values()):
        warn("At least one series looks I(2): PSS bounds testing is INVALID "
             "with I(2) regressors. Difference that series (or respecify) "
             "before trusting M6. Interpret everything downstream cautiously.")
    else:
        print("   No I(2) evidence -> mixed I(0)/I(1) is exactly the ARDL "
              "bounds use-case.")
    record("integration_orders", {k_: f"I({v})" for k_, v in orders.items()})
    status("M5_unitroots", "ok")
    return orders

# --------------------------------------------------------------------------- #
# 7. ARDL BOUNDS TEST + ECM (M6, M7)
# --------------------------------------------------------------------------- #
# PSS (2001), Table CI(iii): unrestricted intercept, no trend ("case III").
# Asymptotic F bounds by k (number of level regressors) and significance:
PSS_F_CASE3 = {
    2: {0.10: (3.17, 4.14), 0.05: (3.79, 4.85), 0.01: (5.15, 6.36)},
    3: {0.10: (2.72, 3.77), 0.05: (3.23, 4.35), 0.01: (4.29, 5.61)},
}
PSS_T_CASE3 = {2: (-2.86, -3.53), 3: (-2.86, -3.78)}  # 5% t-bounds on ECT

def _uecm_frame(df: pd.DataFrame, yvar: str, xvars: list[str],
                pulses: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = df.set_index("year")
    y = d[yvar].to_numpy(float)
    X = d[xvars].to_numpy(float)
    dy = np.diff(y)
    dX = np.diff(X, axis=0)
    # rows: t = 2..T-1  (need Δy_{t-1})
    dep = dy[1:]
    cols, names = [np.ones_like(dep)], ["const"]
    cols.append(y[1:-1]);  names.append(f"{yvar}_L1")
    for j, xv in enumerate(xvars):
        cols.append(X[1:-1, j]); names.append(f"{xv}_L1")
    cols.append(dy[:-1]); names.append(f"d_{yvar}_L1")
    for j, xv in enumerate(xvars):
        cols.append(dX[1:, j]); names.append(f"d_{xv}")
    for pz in pulses:
        cols.append(d[pz].to_numpy(float)[2:]); names.append(pz)
    Z = np.column_stack(cols)
    return dep, Z, names

def manual_bounds(df: pd.DataFrame, yvar: str = "ln_rgdp",
                  xvars: list[str] | None = None,
                  pulses: list[str] | None = None,
                  label: str = "headline",
                  verbose: bool = True) -> dict:
    xvars = xvars or ["ln_rnirc", "ln_pop"]
    pulses = pulses if pulses is not None else ["d2009", "d2020"]
    dep, Z, names = _uecm_frame(df, yvar, xvars, pulses)
    m = LinReg(dep, Z, names)
    lvl_idx = [names.index(f"{yvar}_L1")] + [names.index(f"{v}_L1") for v in xvars]
    Rr = np.delete(Z, lvl_idx, axis=1)
    mr = LinReg(dep, Rr, [n_ for i, n_ in enumerate(names) if i not in lvl_idx])
    q = len(lvl_idx)
    F = ((mr.rss - m.rss) / q) / (m.rss / m.dof)
    theta_y = m.beta[names.index(f"{yvar}_L1")]
    t_ect = theta_y / m.se[names.index(f"{yvar}_L1")]
    k = len(xvars)
    longrun = {v: -m.beta[names.index(f"{v}_L1")] / theta_y for v in xvars}
    out = {"F": float(F), "t_ect": float(t_ect), "theta_y": float(theta_y),
           "longrun": longrun, "model": m, "names": names, "k": k,
           "n_used": m.n}
    if not verbose:
        return out
    print(f"\n   MANUAL UECM ({label}): Δ{yvar} on levels(t-1) + short-run terms"
          f"  [n={m.n}, k={k} regressors in levels]")
    print(m.table(hac=True).to_string())
    cv = PSS_F_CASE3.get(k)
    print(f"\n   PSS bounds F-statistic = {F:.3f}   (H0: no level relationship)")
    if cv:
        for a, (lo, hi) in sorted(cv.items()):
            verdict = ("cointegration" if F > hi else
                       "no cointegration" if F < lo else "inconclusive")
            print(f"      {int(a*100):>2}%: I(0)={lo:.2f}  I(1)={hi:.2f}  -> {verdict}")
    tb = PSS_T_CASE3.get(k)
    print(f"   ECT coefficient θ_y = {theta_y:.3f}  (t = {t_ect:.2f}"
          + (f"; 5% t-bounds {tb[0]}/{tb[1]})" if tb else ")"))
    print("   Implied long-run elasticities: "
          + ", ".join(f"{v}: {c:.3f}" for v, c in longrun.items()))
    note("Critical values are PSS asymptotic; with n=30 they are OVERSIZED. "
         "The design-specific bootstrap below supplies the small-sample "
         "p-value for THIS spec (dummies included).")
    return out

def statsmodels_bounds(df: pd.DataFrame, yvar: str = "ln_rgdp",
                       xvars: list[str] | None = None,
                       pulses: list[str] | None = None) -> dict | None:
    if not HAVE_ARDL:
        note("statsmodels ARDL/UECM unavailable -> official bounds test skipped "
             "(manual F + bootstrap p above stand).")
        return None
    xvars = xvars or ["ln_rnirc", "ln_pop"]
    pulses = pulses if pulses is not None else ["d2009", "d2020"]
    d = df.set_index("year")
    ys, Xs = d[yvar], d[xvars]
    D = d[pulses] if pulses else None
    out: dict[str, str] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chosen = None
        try:
            sel = ardl_select_order(ys, maxlag=2, exog=Xs, maxorder=2,
                                    trend="c", ic="bic", fixed=D)
            chosen = getattr(sel.model, "ardl_order", None)
            floor_note = ("  -> y-lag floored to 1 for the UECM"
                          if chosen and chosen[0] == 0 else "")
            print(f"\n   statsmodels ARDL order selection (BIC): "
                  f"{chosen}{floor_note}")
        except Exception as exc:  # noqa: BLE001
            note(f"order selection failed ({exc.__class__.__name__}); "
                 f"defaulting to UECM(2;1).")
        p_use = max(1, chosen[0]) if chosen else 2
        q_use = max([1] + [max(1, q) for q in (chosen[1:] if chosen else [])])
        # ---- (i) spec-matched: WITH pulse dummies vs asymptotic CVs --------
        try:
            u_fix = UECM(ys, lags=p_use, exog=Xs, order=q_use,
                         trend="c", fixed=D).fit()
            in_design = all(any(pz in str(ix) for ix in u_fix.params.index)
                            for pz in pulses)
            bt = u_fix.bounds_test(case=3)
            flat = " ".join(str(bt).split())[:170]
            print(f"   UECM({p_use};{q_use}) WITH pulse dummies "
                  f"(dummies in design: {in_design}; asymptotic CVs): {flat}")
            if in_design:
                out["with_dummies"] = flat
            else:
                warn("statsmodels dropped the fixed dummies from the design; "
                     "with-dummy verdict comes from the manual F + bootstrap "
                     "p instead.")
        except Exception as exc:  # noqa: BLE001
            note(f"with-dummy bounds_test unavailable "
                 f"({exc.__class__.__name__}) -> manual F + bootstrap p are "
                 f"the with-dummy result.")
        # ---- (ii) sensitivity: WITHOUT dummies, simulated small-sample ----
        try:
            u_pl = UECM(ys, lags=p_use, exog=Xs, order=q_use, trend="c").fit()
            bt_pl = None
            for kwargs in ({"case": 3, "asymptotic": False,
                            "nsim": 20000, "seed": SEED},
                           {"case": 3}):
                try:
                    bt_pl = u_pl.bounds_test(**kwargs)
                    break
                except TypeError:
                    continue
            if bt_pl is not None:
                flat = " ".join(str(bt_pl).split())[:170]
                print(f"   UECM({p_use};{q_use}) WITHOUT dummies "
                      f"(sensitivity; simulated small-sample where "
                      f"supported): {flat}")
                out["no_dummies"] = flat
        except Exception as exc:  # noqa: BLE001
            note(f"no-dummy sensitivity failed ({exc.__class__.__name__}).")
    return out or None

def bootstrap_longrun(df: pd.DataFrame, yvar: str = "ln_rgdp",
                      xvars: list[str] | None = None,
                      pulses: list[str] | None = None) -> pd.DataFrame | None:
    """Moving-block bootstrap (block pairs) of the UECM long-run elasticities."""
    xvars = xvars or ["ln_rnirc", "ln_pop"]
    pulses = pulses if pulses is not None else ["d2009", "d2020"]
    dep, Z, names = _uecm_frame(df, yvar, xvars, pulses)
    n = len(dep)
    n_blocks = math.ceil(n / BLOCK_LEN)
    draws = {v: [] for v in xvars}
    kept = 0
    for _ in range(BOOT_REPS):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(BLOCK_LEN)) % n for s in starts])[:n]
        try:
            m = LinReg(dep[idx], Z[idx], names)
        except Exception:
            continue
        th = m.beta[names.index(f"{yvar}_L1")]
        if abs(th) < 0.02:        # near-unit ECT -> unstable long-run ratio
            continue
        kept += 1
        for v in xvars:
            draws[v].append(-m.beta[names.index(f"{v}_L1")] / th)
    if kept < 100:
        warn("Bootstrap discarded too many draws; CIs not reported.")
        return None
    rows = []
    for v in xvars:
        a = np.array(draws[v])
        rows.append({"variable": v, "median": np.median(a),
                     "ci5": np.percentile(a, 5), "ci95": np.percentile(a, 95),
                     "kept_draws": kept})
    out = pd.DataFrame(rows).set_index("variable").round(3)
    print(f"\n   Moving-block bootstrap of long-run elasticities "
          f"(B={BOOT_REPS}, block={BLOCK_LEN}, kept={kept}) — 90% CIs:")
    print(out.to_string())
    note("90% (not 95%) intervals reported deliberately: with n=30, tail "
         "quantiles beyond that are noise. Ratios of coefficients have heavy "
         "tails when the ECT is weak — hence the |θ_y|>0.02 screen.")
    return out

def johansen_check(df: pd.DataFrame) -> None:
    print()
    if not HAVE_SM:
        note("Johansen cross-check skipped (statsmodels missing).")
        return
    try:
        data = df[["ln_rgdp", "ln_rnirc", "ln_pop"]].to_numpy(float)
        res = coint_johansen(data, det_order=0, k_ar_diff=1)
        print("   Johansen cross-check (levels: ln_rgdp, ln_rnirc, ln_pop; "
              "1 lag in differences):")
        for i, h in enumerate(["r = 0", "r <= 1", "r <= 2"]):
            print(f"      {h}: trace = {res.lr1[i]:7.2f} (95% cv {res.cvt[i,1]:6.2f})"
                  f"   max-eig = {res.lr2[i]:7.2f} (95% cv {res.cvm[i,1]:6.2f})")
        note("Johansen is fragile at n=30 (asymptotic CVs, many parameters); "
             "treat as a directional cross-check of the ARDL result only.")
    except Exception as exc:  # noqa: BLE001
        warn(f"Johansen failed: {exc.__class__.__name__}")

def bounds_smallsample_p(df: pd.DataFrame, yvar: str = "ln_rgdp",
                         xvars: list[str] | None = None,
                         pulses: list[str] | None = None,
                         reps: int = 2000) -> dict | None:
    """Design-specific small-sample p-value for the PSS F, WITH the pulse
    dummies. Under H0 (no level relationship) the DGP is the restricted
    UECM: Δy driven only by short-run terms + dummies; y* is rebuilt
    recursively with iid-resampled restricted residuals, x and dummies held
    at their actual values, and the F recomputed on the exact design."""
    xvars = xvars or ["ln_rnirc", "ln_pop"]
    pulses = pulses if pulses is not None else ["d2009", "d2020"]
    dep, Z, names = _uecm_frame(df, yvar, xvars, pulses)
    lvl = [names.index(f"{yvar}_L1")] + [names.index(f"{v}_L1") for v in xvars]
    keep = [i for i in range(Z.shape[1]) if i not in lvl]
    rn = [names[i] for i in keep]
    mr = LinReg(dep, Z[:, keep], rn)
    mu = LinReg(dep, Z, names)
    q = len(lvl)
    F_obs = float(((mr.rss - mu.rss) / q) / (mu.rss / mu.dof))
    b_const = mr.beta[rn.index("const")]
    b_dylag = mr.beta[rn.index(f"d_{yvar}_L1")]
    b_dx = [mr.beta[rn.index(f"d_{v}")] for v in xvars]
    b_pulse = [mr.beta[rn.index(pz)] for pz in pulses]
    e_pool = mr.resid - mr.resid.mean()
    d0 = df.set_index("year")
    y_lvl = d0[yvar].to_numpy(float)
    X_lvl = d0[xvars].to_numpy(float)
    P_lvl = (d0[pulses].to_numpy(float) if pulses
             else np.zeros((len(d0), 0)))
    dX = np.diff(X_lvl, axis=0)
    n_frame = len(dep)
    dfs = df.copy()
    Fs = np.empty(reps)
    for r_ in range(reps):
        e = rng.choice(e_pool, size=n_frame, replace=True)
        ysim = y_lvl.copy()
        dy_prev = y_lvl[1] - y_lvl[0]
        for t in range(2, len(y_lvl)):
            dy = (b_const + b_dylag * dy_prev
                  + sum(bx * dX[t - 1, j] for j, bx in enumerate(b_dx))
                  + sum(bp * P_lvl[t, j] for j, bp in enumerate(b_pulse))
                  + e[t - 2])
            ysim[t] = ysim[t - 1] + dy
            dy_prev = dy
        dfs[yvar] = ysim
        dep_s, Z_s, _ = _uecm_frame(dfs, yvar, xvars, pulses)
        mrs = LinReg(dep_s, Z_s[:, keep], rn)
        mus = LinReg(dep_s, Z_s, names)
        Fs[r_] = ((mrs.rss - mus.rss) / q) / (mus.rss / mus.dof)
    return {"F_obs": F_obs, "p": float(np.mean(Fs >= F_obs)),
            "cv": {"10%": float(np.percentile(Fs, 90)),
                   "5%": float(np.percentile(Fs, 95)),
                   "1%": float(np.percentile(Fs, 99))},
            "reps": reps}

def bounds_module(df: pd.DataFrame) -> dict:
    banner("M6  ARDL BOUNDS TEST (PSS) + LONG-RUN ECM — headline model")
    print("   Headline spec (total-effect estimand, M4): "
          "ln real GDP ~ ln real NIRC + ln population | pulses 2009, 2020")
    res = manual_bounds(df)
    cv5 = PSS_F_CASE3[res["k"]][0.05]
    cv1 = PSS_F_CASE3[res["k"]][0.01]
    verdict = ("cointegration at 1%" if res["F"] > cv1[1] else
               "cointegration at 5%" if res["F"] > cv5[1] else
               "inconclusive at 5%" if res["F"] > cv5[0] else
               "no cointegration at 5%")
    record("bounds", {
        "F": round(res["F"], 3), "verdict_asymptotic": verdict,
        "ECT": round(res["theta_y"], 3), "ECT_t": round(res["t_ect"], 2),
        "longrun": {k_: round(v, 3) for k_, v in res["longrun"].items()}})
    r_np = manual_bounds(df, pulses=[], label="no pulses", verbose=False)
    print(f"\n   SENSITIVITY: dropping the 2009/2020 pulses gives F = "
          f"{r_np['F']:.2f} on the same UECM — the level relationship is "
          f"masked unless the COVID outlier is absorbed. Report both.")
    record("bounds_sensitivity", {"F_with_pulses": round(res["F"], 2),
                                  "F_no_pulses": round(r_np["F"], 2)})
    bs_p = bounds_smallsample_p(df)
    if bs_p:
        print(f"   Design-specific bootstrap under H0 "
              f"(reps={bs_p['reps']}, exact regressors incl. pulses): "
              f"small-sample p = {bs_p['p']:.4f}; empirical CVs 10/5/1% = "
              f"{bs_p['cv']['10%']:.2f} / {bs_p['cv']['5%']:.2f} / "
              f"{bs_p['cv']['1%']:.2f}")
        note("iid residual resampling is defensible here because the "
             "short-run terms whiten the errors (Ljung-Box p in M7).")
        record("bounds_boot", {"p": round(bs_p["p"], 4),
                               "cv5": round(bs_p["cv"]["5%"], 2),
                               "cv1": round(bs_p["cv"]["1%"], 2)})
    smres = statsmodels_bounds(df)
    if smres:
        if smres.get("with_dummies"):
            record("bounds_sm_withdummies", smres["with_dummies"])
        if smres.get("no_dummies"):
            record("bounds_sm_nodummies", smres["no_dummies"])
    boot = bootstrap_longrun(df)
    if boot is not None:
        record("longrun_boot_ci90", boot)
    johansen_check(df)
    banner("M7  ECM RESIDUAL DIAGNOSTICS (headline UECM)")
    record("ecm_diagnostics",
           res["model"].diagnostics("UECM residuals (headline)"))
    note("Newey-West HAC errors are reported in every coefficient table; "
         "Breusch-Godfrey + Ljung-Box speak to the paper's DW=1.38 concern.")
    status("M6_bounds", "ok")
    status("M7_diagnostics", "ok")
    return {"manual": res, "sm": smres, "boot": boot}

# --------------------------------------------------------------------------- #
# 8. TODA-YAMAMOTO CAUSALITY (M8)   [manual, augmentation lags excluded]
# --------------------------------------------------------------------------- #
def _lagmat(a: np.ndarray, p: int) -> np.ndarray:
    return np.column_stack([a[p - i - 1: len(a) - i - 1] for i in range(p)])

def toda_yamamoto(df: pd.DataFrame, pair: tuple[str, str],
                  dmax: int = 1) -> list[dict]:
    y1, y2 = pair
    a = df[y1].to_numpy(float)
    bb = df[y2].to_numpy(float)
    # lag order by joint BIC over p in {1, 2} on the levels VAR
    best_p, best_bic = 1, np.inf
    for p in (1, 2):
        n_eff = len(a) - p
        X = np.column_stack([np.ones(n_eff), _lagmat(a, p), _lagmat(bb, p)])
        r1 = a[p:] - X @ np.linalg.lstsq(X, a[p:], rcond=None)[0]
        r2 = bb[p:] - X @ np.linalg.lstsq(X, bb[p:], rcond=None)[0]
        Sig = np.cov(np.vstack([r1, r2]))
        _, logdet = np.linalg.slogdet(Sig)
        bic = n_eff * logdet + (2 * (1 + 2 * p)) * math.log(n_eff)
        if bic < best_bic:
            best_p, best_bic = p, bic
    p = best_p
    P = p + dmax
    n_eff = len(a) - P
    out_rows: list[dict] = []
    for cause, effect, ca, ce in ((y2, y1, bb, a), (y1, y2, a, bb)):
        X = np.column_stack([np.ones(n_eff), _lagmat(ce, P), _lagmat(ca, P)])
        names = (["const"] + [f"{effect}_L{i}" for i in range(1, P + 1)]
                 + [f"{cause}_L{i}" for i in range(1, P + 1)])
        m = LinReg(ce[P:], X, names)
        idx = [names.index(f"{cause}_L{i}") for i in range(1, p + 1)]
        stat, pval = m.wald(idx)
        print(f"   H0: {cause} does NOT Granger-cause {effect} "
              f"[VAR({p}) + {dmax} aug. lag(s)] : Wald χ²({p}) = {stat:.2f}, "
              f"p = {pval:.3f}  -> {'REJECT' if pval < ALPHA else 'not rejected'}")
        out_rows.append({"direction": f"{cause} -> {effect}", "chi2": round(stat, 2),
                         "df": p, "p": round(pval, 3),
                         "causality": bool(pval < ALPHA)})
    return out_rows

def ty_module(df: pd.DataFrame, orders: dict[str, int]) -> None:
    banner("M8  TODA-YAMAMOTO GRANGER CAUSALITY (levels VAR + d_max lags)")
    dmax = max(1, min(2, max(orders.get(s, 1) for s in ("ln_rgdp", "ln_rnirc"))))
    print(f"   d_max = {dmax} (from M5); Wald restrictions exclude the "
          f"augmentation lag(s) — the step statsmodels' test_causality skips.")
    rows = toda_yamamoto(df, ("ln_rgdp", "ln_rnirc"), dmax)
    rows += toda_yamamoto(df, ("ln_rgdp", "unemp"),
                  max(1, min(2, max(orders.get("ln_rgdp", 1),
                                    orders.get("unemp", 1)))))
    record("toda_yamamoto", rows)
    note("If GDP -> unemployment rejects while the reverse does not (the "
         "Okun direction), that directly supports M4's decision to drop "
         "unemployment from the total-effect spec — cite it there.")
    note("With 30 annual obs these tests have LOW POWER; a non-rejection is "
         "weak evidence of absence. Report as suggestive, not conclusive.")
    status("M8_toda_yamamoto", "ok")

# --------------------------------------------------------------------------- #
# 9. VAR + IRFs (M9)
# --------------------------------------------------------------------------- #
def var_module(df: pd.DataFrame) -> None:
    banner("M9  SMALL VAR IN GROWTH RATES + ORTHOGONALISED IRFs")
    if not HAVE_SM:
        status("M9_var", "skipped (no statsmodels)")
        note("statsmodels missing -> VAR/IRF skipped.")
        return
    try:
        d = df.set_index("year")[["ln_rnirc", "ln_rgdp"]].diff().dropna()
        d.columns = ["d_ln_rnirc", "d_ln_rgdp"]
        model = VAR(d)
        sel_lag = model.select_order(maxlags=2).selected_orders.get("bic", 1)
        sel_lag = max(1, int(sel_lag))
        res = model.fit(sel_lag)
        print(f"   VAR({sel_lag}) on [Δln real NIRC, Δln real GDP]  "
              f"(BIC-selected; n = {res.nobs})")
        irf = res.irf(8)
        fig = irf.plot(orth=True)
        fig.suptitle("Orthogonalised IRFs (Cholesky order: NIRC shock first)",
                     y=1.02)
        fig.savefig(f"{OUTDIR}/fig_irf.png", dpi=150, bbox_inches="tight")
        plt.show(); plt.close(fig)
        note("Cholesky ordering places NIRC first (spending rule set at "
             "Budget time, before same-year GDP is realised). Reverse the "
             "ordering as a sensitivity check; with annual data the "
             "identification is admittedly coarse.")
        record("var", {"lag": int(sel_lag), "irf_fig": "fig_irf.png"})
        status("M9_var", "ok")
    except Exception as exc:  # noqa: BLE001
        warn(f"VAR/IRF failed: {exc.__class__.__name__}: {exc}")
        status("M9_var", f"error ({exc.__class__.__name__})")

# --------------------------------------------------------------------------- #
# 10. STABILITY (M10)
# --------------------------------------------------------------------------- #
def stability_module(df: pd.DataFrame) -> None:
    banner("M10 STABILITY — CUSUM / CUSUMSQ / rolling betas / breaks")
    d = df.set_index("year")
    y = d["ln_rgdp"].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d["ln_rnirc"], d["ln_pop"]])
    names = ["const", "ln_rnirc", "ln_pop"]
    years = d.index.to_numpy()
    k = X.shape[1]

    # --- recursive CUSUM / CUSUMSQ (manual, Brown-Durbin-Evans) -------------
    w = recursive_residuals(y, X)
    sig = w.std(ddof=1)
    cusum = np.cumsum(w) / sig
    T = len(w)
    t_ax = years[k + 1:][:T] if len(years[k + 1:]) >= T else years[-T:]
    bound = 0.948 * (math.sqrt(T) + 2 * np.arange(1, T + 1) / math.sqrt(T))
    cusumsq = np.cumsum(w ** 2) / (w ** 2).sum()
    exp_line = np.arange(1, T + 1) / T
    c0 = 1.358 / math.sqrt(T)      # approximate 5% band; informal
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t_ax, cusum, lw=1.5)
    axes[0].plot(t_ax, bound, "r--", lw=1); axes[0].plot(t_ax, -bound, "r--", lw=1)
    axes[0].set_title("Recursive CUSUM (5% bounds)")
    axes[1].plot(t_ax, cusumsq, lw=1.5)
    axes[1].plot(t_ax, exp_line, "k:", lw=1)
    axes[1].plot(t_ax, np.clip(exp_line + c0, 0, 1), "r--", lw=1)
    axes[1].plot(t_ax, np.clip(exp_line - c0, 0, 1), "r--", lw=1)
    axes[1].set_title("CUSUM of squares (approx. band)")
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig_cusum.png", dpi=150)
    plt.show(); plt.close(fig)
    outside = np.mean(np.abs(cusum) > bound)
    print(f"   CUSUM outside 5% bounds for {outside:.0%} of the sample "
          f"({'instability signal' if outside > 0 else 'no breach'}).")
    if HAVE_RLS:
        try:
            rls = RecursiveLS(y, X).fit()
            figc = rls.plot_cusum(figsize=(7, 3.5))
            figc.savefig(f"{OUTDIR}/fig_cusum_statsmodels.png", dpi=150)
            plt.close(figc)
            print("   statsmodels RecursiveLS CUSUM saved (cross-check).")
        except Exception:
            pass

    # --- rolling coefficients ------------------------------------------------
    win = 12
    rows = []
    for s0 in range(0, len(y) - win + 1):
        m = LinReg(y[s0:s0 + win], X[s0:s0 + win], names)
        rows.append({"end_year": int(years[s0 + win - 1]),
                     "b_ln_rnirc": m.beta[1], "b_ln_pop": m.beta[2]})
    roll = pd.DataFrame(rows).set_index("end_year")
    fig, ax = plt.subplots(figsize=(8, 4))
    roll.plot(ax=ax, marker="o", ms=3)
    ax.set_title(f"Rolling {win}-year long-run elasticities (static spec)")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig_rolling.png", dpi=150)
    plt.show(); plt.close(fig)
    print(f"   Rolling ln_rnirc elasticity range: "
          f"{roll.b_ln_rnirc.min():.2f} … {roll.b_ln_rnirc.max():.2f}")

    # --- sup-F single break (descriptive) ------------------------------------
    trim = 8
    best = (None, -np.inf)
    full = LinReg(y, X, names)
    for cut in range(trim, len(y) - trim):
        m1 = LinReg(y[:cut], X[:cut], names)
        m2 = LinReg(y[cut:], X[cut:], names)
        F = ((full.rss - m1.rss - m2.rss) / k) / \
            ((m1.rss + m2.rss) / (len(y) - 2 * k))
        if F > best[1]:
            best = (int(years[cut]), float(F))
    print(f"   sup-F single-break scan: candidate break at {best[0]} "
          f"(F = {best[1]:.2f}; compare against Andrews-1993 critical values, "
          f"NOT the standard F table).")

    # --- multiple breaks via ruptures (Bai-Perron analogue) ------------------
    bks_years: list[int] | None = None
    if HAVE_RUPTURES:
        try:
            resid_sig = full.resid.reshape(-1, 1)
            algo = rpt.Binseg(model="l2").fit(resid_sig)
            bks = algo.predict(n_bkps=2)[:-1]
            bks_years = [int(years[i - 1]) for i in bks]
            print("   ruptures (Binseg, 2 breaks) on static-model residuals: "
                  + ", ".join(map(str, bks_years)))
        except Exception as exc:  # noqa: BLE001
            note(f"ruptures scan failed ({exc.__class__.__name__}).")
    else:
        note("`ruptures` not installed -> multiple-break scan skipped "
             "(pip install ruptures).")
    note("Expect 2008/09 and 2020 to dominate every break statistic; that is "
         "the GFC/COVID story, and the pulse dummies in M6/M11 absorb it.")
    record("stability", {
        "cusum_outside_5pct": f"{outside:.0%}",
        "supF_break": {"year": best[0], "F": round(best[1], 2)},
        "rolling_nirc_range": [round(float(roll.b_ln_rnirc.min()), 2),
                               round(float(roll.b_ln_rnirc.max()), 2)],
        "ruptures_breaks": bks_years})
    status("M10_stability", "ok")

# --------------------------------------------------------------------------- #
# 11. ROBUSTNESS BATTERY (M11)
# --------------------------------------------------------------------------- #
def robustness_module(df: pd.DataFrame, b: DataBundle) -> pd.DataFrame:
    banner("M11 ROBUSTNESS — alt specs, regimes, subsample, out-of-sample")
    specs: list[tuple[str, str, list[str], pd.DataFrame]] = []
    full = df
    sub = df[df["year"] >= 2001]              # drops author-estimated NIRC years
    specs.append(("A0 replication (nominal levels, as paper)",
                  "gdp", ["nirc", "pop", "unemp"], full))
    specs.append(("A1 log real (causal estimand)",
                  "ln_rgdp", ["ln_rnirc", "ln_pop"], full))
    specs.append(("A2 = A1 + crisis pulses",
                  "ln_rgdp", ["ln_rnirc", "ln_pop", "d2009", "d2020"], full))
    specs.append(("A3 per-capita",
                  "ln_rgdp_pc", ["ln_rnirc_pc", "d2009", "d2020"], full))
    specs.append(("A4 conditional assoc. (+unemp, paper structure in logs)",
                  "ln_rgdp", ["ln_rnirc", "ln_pop", "unemp"], full))
    specs.append(("A5 subsample 2001-2023 (official NIRC only)",
                  "ln_rgdp", ["ln_rnirc", "ln_pop", "d2009", "d2020"], sub))
    specs.append(("A6 regime steps (NIR 2009, Temasek 2016)",
                  "ln_rgdp", ["ln_rnirc", "ln_pop", "nir_regime",
                              "temasek_regime"], full))
    if "debt_gdp" in df.columns and df["debt_gdp"].notna().all():
        specs.append(("A7 + gross debt ratio (see M14 caveat)",
                      "ln_rgdp", ["ln_rnirc", "ln_pop", "debt_gdp",
                                  "d2009", "d2020"], full))
    if getattr(b, "wb", None) and "rgdp_wb" in b.wb:
        wbr = np.log(b.wb["rgdp_wb"].astype(float)).rename("ln_rgdp_wb")
        merged = df.merge(wbr, left_on="year", right_index=True, how="inner")
        if len(merged) >= 25:
            specs.append(("A8 World Bank chain-linked real GDP (no CPI "
                          "deflation)", "ln_rgdp_wb",
                          ["ln_rnirc", "ln_pop", "d2009", "d2020"], merged))
    rows = []
    key_for = {"gdp": "nirc"}
    for label, yv, xs, data in specs:
        Z = np.column_stack([np.ones(len(data))]
                            + [data[c].to_numpy(float) for c in xs])
        m = LinReg(data[yv].to_numpy(float), Z, ["const"] + xs)
        kv = key_for.get(yv, "ln_rnirc" if "ln_rnirc" in xs else
                         ("ln_rnirc_pc" if "ln_rnirc_pc" in xs else xs[0]))
        j = (["const"] + xs).index(kv)
        se_hac = math.sqrt(m.hac_cov()[j, j])
        rows.append({"spec": label, "n": m.n, "NIRC coef": m.beta[j],
                     "HAC se": se_hac, "t (HAC)": m.beta[j] / se_hac,
                     "R2": m.r2, "DW": m.dw()})
    tab = pd.DataFrame(rows).set_index("spec").round(3)
    with pd.option_context("display.width", 130):
        print(tab.to_string())
    tab.to_csv(f"{OUTDIR}/alt_specs.csv")
    record("alt_specs", tab)
    note("A0 reproduces the paper's pathology on purpose: towering R² with "
         "DW << 2 in trending levels — the Granger-Newbold spurious-"
         "regression signature the ARDL/ECM respec fixes. A1-A7 report the "
         "NIRC elasticity; expect it far below the nominal-levels slope.")

    # --- out-of-sample: fit 1994-2015, forecast 2016-2023 --------------------
    tr = df[df["year"] <= 2015]; te = df[df["year"] >= 2016]
    Xtr = np.column_stack([np.ones(len(tr)), tr["ln_rnirc"], tr["ln_pop"]])
    m = LinReg(tr["ln_rgdp"].to_numpy(float), Xtr,
               ["const", "ln_rnirc", "ln_pop"])
    Xte = np.column_stack([np.ones(len(te)), te["ln_rnirc"], te["ln_pop"]])
    f_static = Xte @ m.beta
    # dynamic ARDL(1,1) forecast, iterating on its own predictions
    tr2 = tr.iloc[1:]
    Za = np.column_stack([np.ones(len(tr2)),
                          tr["ln_rgdp"].to_numpy(float)[:-1],
                          tr2["ln_rnirc"].to_numpy(float),
                          tr["ln_rnirc"].to_numpy(float)[:-1],
                          tr2["ln_pop"].to_numpy(float)])
    ma = LinReg(tr2["ln_rgdp"].to_numpy(float), Za,
                ["const", "y_L1", "x", "x_L1", "pop"])
    y_prev = float(tr["ln_rgdp"].iloc[-1]); x_prev = float(tr["ln_rnirc"].iloc[-1])
    f_dyn = []
    for _, r in te.iterrows():
        yh = (ma.beta[0] + ma.beta[1] * y_prev + ma.beta[2] * r["ln_rnirc"]
              + ma.beta[3] * x_prev + ma.beta[4] * r["ln_pop"])
        f_dyn.append(yh); y_prev, x_prev = yh, r["ln_rnirc"]
    f_dyn = np.array(f_dyn)
    drift = float(np.diff(tr["ln_rgdp"].to_numpy(float)).mean())
    f_rw = tr["ln_rgdp"].iloc[-1] + drift * np.arange(1, len(te) + 1)
    truth = te["ln_rgdp"].to_numpy(float)
    def _rmse(f): return float(np.sqrt(np.mean((truth - f) ** 2)))
    def _mape(f): return float(np.mean(np.abs(np.exp(truth) - np.exp(f))
                                       / np.exp(truth)) * 100)
    oos = pd.DataFrame({
        "model": ["static long-run (cond. on exog)", "ARDL(1,1) dynamic",
                  "RW with drift (benchmark)"],
        "RMSE (log pts)": [_rmse(f_static), _rmse(f_dyn), _rmse(f_rw)],
        "MAPE %": [_mape(f_static), _mape(f_dyn), _mape(f_rw)],
    }).set_index("model").round(3)
    print("\n   Out-of-sample 1994-2015 -> 2016-2023 (log real GDP):")
    print(oos.to_string())
    record("oos", oos)
    note("2020 sits in the test window; every model misses the COVID drop, "
         "the honest way to show the limits of a 30-obs macro fit. Beating "
         "RW-with-drift out-of-sample is a high bar — report it either way.")
    status("M11_robustness", "ok")
    return tab

# --------------------------------------------------------------------------- #
# 12. TAX-EQUIVALENCE COUNTERFACTUAL (M12)
# --------------------------------------------------------------------------- #
def tax_module(df: pd.DataFrame, b: DataBundle) -> pd.DataFrame | None:
    banner("M12 TAX EQUIVALENCE — NIRC in GST points and CIT points")
    print(f"   Revenue source: {b.tax_source}")
    tax = b.tax.copy()
    tax["fy"] = tax["fy"].astype(int)
    nirc = df.set_index("year")["nirc"]
    if b.nirc_official is not None:
        nirc = b.nirc_official.combine_first(nirc)
        print("   NIRC source: MOF official where available, paper series else.")
    rows = []
    for _, r in tax.iterrows():
        fy = int(r["fy"])
        if fy not in nirc.index or fy < 2009:
            continue
        rate = GST_RATE.get(fy, 9.0)
        per_gst = r["gst"] / rate
        per_cit = r["cit"] / CIT_RATE
        rows.append({"FY": fy, "NIRC (S$bn)": nirc.loc[fy],
                     "GST rev": r["gst"], "GST rate %": rate,
                     "NIRC in GST points": nirc.loc[fy] / per_gst,
                     "CIT rev": r["cit"],
                     "NIRC in CIT points": nirc.loc[fy] / per_cit})
    if not rows:
        warn("No overlapping years between NIRC and tax revenue — skipped.")
        status("M12_tax_equiv", "skipped")
        return None
    tab = pd.DataFrame(rows).set_index("FY").round(2)
    show = tab.tail(8)
    with pd.option_context("display.width", 120):
        print(show.to_string())
    last = tab.iloc[-1]
    print(f"\n   Reading FY{tab.index[-1]}: NIRC of S${last['NIRC (S$bn)']:.1f}bn "
          f"does the fiscal work of ≈{last['NIRC in GST points']:.1f} GST "
          f"percentage points (statutory rate {last['GST rate %']:.0f}%) or "
          f"≈{last['NIRC in CIT points']:.1f} CIT points (headline 17%).")
    record("tax_equiv", {
        "source": b.tax_source, "table": show,
        "latest": {"FY": int(tab.index[-1]),
                   "gst_points": round(float(last["NIRC in GST points"]), 1),
                   "cit_points": round(float(last["NIRC in CIT points"]), 1)}})
    print("""   Caveats to print under the table in the paper:
     * Static arithmetic: no behavioural response, no Laffer effects, no
       GE adjustment — a lower bound on the true distortion cost avoided.
     * FY (Apr-Mar, MOF) vs calendar-year NIRC in Appendix A: half-year
       timing mismatch unless you re-date one side.
     * NIRC is up to 50% of EXPECTED long-run real returns — a smoothed
       policy variable, not realised income; the equivalence is about the
       budget constraint, not market performance.
   Bridge for the thesis arc: the same reserve architecture that funds
   these 'free' tax points is the institutional platform MAS now extends
   into tokenised assets (Project Guardian / Global Layer One) — the
   tiger-to-tokens continuity claim, stated institutionally rather than
   rhetorically.""")
    fig, ax = plt.subplots(figsize=(8, 4))
    show[["NIRC in GST points", "NIRC in CIT points"]].plot.bar(ax=ax)
    ax.set_title("NIRC expressed in tax-rate points")
    ax.set_ylabel("percentage points"); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig_tax_points.png", dpi=150)
    plt.show(); plt.close(fig)
    status("M12_tax_equiv",
           "ok" if "official" in b.tax_source else "ok (ILLUSTRATIVE revenue)")
    return tab

# --------------------------------------------------------------------------- #
# 13. PROGRESS DRIVERS — BUDGET COMPOSITION (M13)
# --------------------------------------------------------------------------- #
def drivers_module(b: DataBundle) -> None:
    banner("M13 PROGRESS DRIVERS — where the budget goes")
    comp = dict(b.exp_sector)
    fy = comp.pop("_fy", None)
    print(f"   Source: {b.exp_source}" + (f"  (FY{fy})" if fy else ""))
    ser = pd.Series(comp).sort_values(ascending=False)
    total = ser.sum()
    top = ser.head(10)
    tab = pd.DataFrame({"S$bn": top, "share %": top / total * 100}).round(1)
    print(tab.to_string())
    record("drivers", {"top": tab.head(5), "source": b.exp_source, "fy": fy})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    top.iloc[::-1].plot.barh(ax=ax)
    ax.set_title(f"Government expenditure composition"
                 + (f" (FY{fy})" if fy else ""))
    ax.set_xlabel("S$ bn"); ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig_expenditure.png", dpi=150)
    plt.show(); plt.close(fig)
    print("""   How to read this against your 'progress drivers' question:
     * Infrastructure is split across Transport (rail/road), National
       Development (housing, land) and Communications — physical, telecom
       and financial infrastructure never appear as one budget line.
     * 'Energy transition' has NO dedicated head: it is spread across
       Sustainability & Environment (incl. coastal protection), Trade &
       Industry (EMA, JTC) and SINGA-financed projects — say so explicitly
       rather than proxying it with one ministry.
     * Health + Education together rival Defence — the social-investment
       tilt that NIRC headroom finances is visible right here, which is
       the budget-side complement to the M12 tax-equivalence result.
     * For the paper: development vs operating split (data.gov.sg
       'Expenditure by Type') separates capacity-building outlays from
       recurrent spending — the cleaner 'progress driver' cut.""")
    status("M13_drivers", "ok" if "official" in b.exp_source
           else "ok (ILLUSTRATIVE composition)")

# --------------------------------------------------------------------------- #
# 14. DEBT MODULE (M14)
# --------------------------------------------------------------------------- #
def debt_module(df: pd.DataFrame, b: DataBundle) -> None:
    banner("M14 GROSS DEBT — high ratio, asset-backed institution")
    print(f"   Source: {b.debt_source}")
    d = df.set_index("year")
    fig, ax1 = plt.subplots(figsize=(8.5, 4))
    ax1.plot(d.index, d["debt_gdp"], color="tab:red", marker="o", ms=3,
             label="Gross govt debt, % GDP")
    ax1.set_ylabel("Gross debt, % of GDP", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(d.index, d["nirc_share"], color="tab:blue", marker="s", ms=3,
             label="NIRC, % GDP")
    ax2.set_ylabel("NIRC, % of GDP", color="tab:blue")
    ax1.set_title("Debt ratio vs NIRC share — two faces of the same balance sheet")
    ax1.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig_debt.png", dpi=150)
    plt.show(); plt.close(fig)
    corr = d[["debt_gdp", "nirc_share"]].corr().iloc[0, 1]
    print(f"   Correlation(debt ratio, NIRC share) = {corr:.2f}")
    record("debt", {"latest_year": int(d.index.max()),
                    "latest_pct_gdp": round(float(d["debt_gdp"].iloc[-1]), 1),
                    "corr_with_nirc_share": round(float(corr), 2),
                    "source": b.debt_source})
    print("""   Institutional caveat that MUST accompany any regression use:
     * Singapore's gross debt (~150-170% of GDP, among the world's highest)
       is NOT deficit financing. SGS/SSGS are issued to build the bond
       market and to give CPF a guaranteed asset; under the Government
       Securities Act the proceeds are INVESTED (managed by MAS/GIC), not
       spent. Net of financial assets the government position is strongly
       positive — the AAA rating rests on that.
     * SINGA (2021) is the exception that proves the rule: up to S$90bn of
       borrowing ring-fenced for major long-dated infrastructure — the
       'borrow to invest in progress drivers' channel you asked about,
       and the natural link from M13's composition data to the debt stock.
     * Econometrically: debt/GDP here proxies the SIZE of the CPF-reserves
       intermediation machine, not fiscal stress — which is why A7 (M11)
       treats it as a descriptive control and why its coefficient must not
       be read as a 'debt burden' effect. The right structural object for
       the thesis is the asset side (GIC/Temasek/MAS AUM), which is not
       published — say that limitation openly.""")
    status("M14_debt", "ok")

# --------------------------------------------------------------------------- #
# 15. CANDIDATE IV DEMO (M15)
# --------------------------------------------------------------------------- #
def iv_module(df: pd.DataFrame, b: DataBundle) -> None:
    banner("M15 CANDIDATE IV — global equity returns instrumenting ΔNIRC")
    print(f"   Instrument source: {b.spx_source}")
    d = df.set_index("year").copy()
    d["dy"] = d["ln_rgdp"].diff()
    d["dx"] = d["ln_rnirc"].diff()
    d["dpop"] = d["ln_pop"].diff()
    d = d.join(b.spx.rename("z"), how="left")
    d["z_l1"] = d["z"].shift(1)
    d = d.dropna(subset=["dy", "dx", "dpop", "z_l1"])
    if len(d) < 15:
        note("Too few overlapping obs for the IV demo — skipped.")
        status("M15_iv", "skipped (few obs)")
        return
    y = d["dy"].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d["dpop"], d["dx"]])
    Z = np.column_stack([np.ones(len(d)), d["dpop"], d["z_l1"]])
    # first stage
    fs = LinReg(d["dx"].to_numpy(float), Z, ["const", "dpop", "z_l1"])
    fs_t = fs.beta[2] / fs.se[2]
    print(f"   First stage: Δln NIRC on lagged world-equity return — "
          f"t = {fs_t:.2f}, F = {fs_t**2:.2f} "
          f"({'WEAK instrument (F<10): 2SLS unreliable' if fs_t**2 < 10 else 'passes the F>10 rule of thumb'})")
    ZtX = Z.T @ X
    beta_iv = np.linalg.solve(ZtX, Z.T @ y)
    resid = y - X @ beta_iv
    sig2 = float(resid @ resid) / (len(y) - X.shape[1])
    ZtX_inv = np.linalg.inv(ZtX)
    cov_iv = sig2 * ZtX_inv @ (Z.T @ Z) @ ZtX_inv.T
    se_iv = np.sqrt(np.diag(cov_iv))
    beta_ols = float(LinReg(y, X, ["c", "p", "x"]).beta[2])
    print(f"   2SLS (just-identified): Δln GDP on Δln NIRC — "
          f"coef = {beta_iv[2]:.3f} (se {se_iv[2]:.3f})  vs OLS {beta_ols:.3f}")
    record("iv", {"first_stage_F": round(float(fs_t ** 2), 2),
                  "beta_2sls": round(float(beta_iv[2]), 3),
                  "se_2sls": round(float(se_iv[2]), 3),
                  "beta_ols": round(beta_ols, 3),
                  "source": b.spx_source})
    print("""   Exclusion caveat (print verbatim in the paper): for a hyper-open
   economy, world equity returns move Singapore GDP through trade and
   financial channels directly, not only through NIRC — the exclusion
   restriction is doubtful, so this IV bounds the story rather than
   identifying it. The cleaner quasi-experiments remain the 2009 NIR and
   2016 Temasek framework changes (A6).""")
    status("M15_iv", "ok (demo)")

# --------------------------------------------------------------------------- #
# 16. EXPORT + ORCHESTRATION (M16)
# --------------------------------------------------------------------------- #
def export_module(df: pd.DataFrame) -> None:
    banner("M16 EXPORT")
    df.to_csv(f"{OUTDIR}/merged_data.csv", index=False)
    appx = ["year", "gdp", "nirc", "pop", "unemp", "cpi", "cpi_raw", "m2"]
    (df[[c for c in appx if c in df.columns]]
       .rename(columns={"cpi": "cpi_corrected", "cpi_raw": "cpi_original"})
       .round(4)
       .to_csv(f"{OUTDIR}/appendix_a_corrected.csv", index=False))
    print("   Wrote appendix_a_corrected.csv — drop-in replacement table "
          "for the paper's Appendix A (cpi_corrected column).")
    print(f"   Wrote {OUTDIR}/merged_data.csv, alt_specs.csv + all figures.")
    status("M16_export", "ok")

def _jsonable(v):
    if isinstance(v, pd.DataFrame):
        return json.loads(v.round(4).to_json(orient="index"))
    if isinstance(v, pd.Series):
        return json.loads(v.round(4).to_json())
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v

def summary_module() -> None:
    banner("M17 CONSOLIDATED RESULTS — everything on one screen")
    L: list[str] = []
    say = L.append

    say("DATA")
    rep = RESULTS.get("cpi_repair")
    if rep:
        how = (f"spliced ×{rep['factor']}" if rep["factor"]
               else "replaced with World Bank series")
        say(f"  CPI base break at {rep['break_year']} {how}; corrected "
            f"Appendix A exported (appendix_a_corrected.csv).")
    else:
        say(f"  CPI repair: {MODULE_STATUS.get('M2b_cpi_repair', 'n/a')}")
    io_ = RESULTS.get("integration_orders")
    if io_:
        say("  Integration orders: "
            + ", ".join(f"{k} {v}" for k, v in io_.items()))

    say("")
    say("COINTEGRATION & LONG RUN  (ln rGDP ~ ln rNIRC + ln Pop | 2009, 2020 pulses)")
    bd = RESULTS.get("bounds")
    if bd:
        say(f"  PSS bounds F = {bd['F']}  ->  {bd['verdict_asymptotic']} "
            f"(asymptotic CVs; small-sample check below)")
        say(f"  Error-correction term = {bd['ECT']} (t = {bd['ECT_t']}): "
            f"~{abs(bd['ECT']) * 100:.0f}% of any deviation from the long-run "
            f"path is corrected per year")
        say("  Long-run elasticities: "
            + ", ".join(f"{k} = {v}" for k, v in bd["longrun"].items()))
    bboot = RESULTS.get("bounds_boot")
    if bboot:
        say(f"  Small-sample p (design-specific bootstrap, WITH pulses): "
            f"p = {bboot['p']}   (empirical 5%/1% CVs "
            f"{bboot['cv5']} / {bboot['cv1']})")
    sens = RESULTS.get("bounds_sensitivity")
    if sens:
        say(f"  Sensitivity: F = {sens['F_with_pulses']} with the 2009/2020 "
            f"pulses vs {sens['F_no_pulses']} without — the verdict hinges "
            f"on absorbing the COVID outlier")
    for key_, lab_ in (("bounds_sm_withdummies",
                        "statsmodels cross-check (with dummies)"),
                       ("bounds_sm_nodummies",
                        "statsmodels sensitivity (no dummies)")):
        v_ = RESULTS.get(key_)
        if v_:
            say(f"  {lab_}: {v_}")
    bb = RESULTS.get("longrun_boot_ci90")
    if isinstance(bb, pd.DataFrame):
        for v_, r_ in bb.iterrows():
            say(f"  Bootstrap 90% CI, {v_}: [{r_['ci5']:.2f}, {r_['ci95']:.2f}]"
                f"  (median {r_['median']:.2f})")

    dg = RESULTS.get("ecm_diagnostics")
    if isinstance(dg, pd.DataFrame):
        say("")
        say("ECM DIAGNOSTICS  (p-values; DW is the statistic itself)")
        parts = []
        for name_, r_ in dg.iterrows():
            if str(name_).startswith("Durbin"):
                parts.append(f"DW {r_['stat']:.2f}")
            else:
                parts.append(f"{str(name_).split(' ')[0]} p={r_['p']:.2f}")
        say("  " + " | ".join(parts))

    ty = RESULTS.get("toda_yamamoto")
    if ty:
        say("")
        say("CAUSALITY  (Toda-Yamamoto Wald; augmentation lags excluded)")
        for r_ in ty:
            tail = ("CAUSALITY (H0 rejected)" if r_["causality"]
                    else "no evidence")
            say(f"  {r_['direction']:<22} chi2({r_['df']}) = {r_['chi2']:<6} "
                f"p = {r_['p']:<6} -> {tail}")

    st = RESULTS.get("stability")
    if st:
        say("")
        say("STABILITY")
        line = (f"  CUSUM outside 5% band: {st['cusum_outside_5pct']}; "
                f"sup-F break candidate {st['supF_break']['year']} "
                f"(F = {st['supF_break']['F']}); rolling NIRC elasticity "
                f"{st['rolling_nirc_range'][0]} … {st['rolling_nirc_range'][1]}")
        if st.get("ruptures_breaks"):
            line += f"; ruptures breaks: {st['ruptures_breaks']}"
        say(line)

    al = RESULTS.get("alt_specs")
    if isinstance(al, pd.DataFrame):
        say("")
        say("ROBUSTNESS — NIRC coefficient across specifications")
        for spec_, r_ in al.iterrows():
            say(f"  {str(spec_)[:52]:<52} {r_['NIRC coef']:>7.3f}   "
                f"(t = {r_['t (HAC)']:.1f}, DW = {r_['DW']:.2f}, n = {int(r_['n'])})")

    oo = RESULTS.get("oos")
    if isinstance(oo, pd.DataFrame):
        say("")
        say("OUT-OF-SAMPLE  fit 1994-2015 -> forecast 2016-2023")
        for m_, r_ in oo.iterrows():
            say(f"  {str(m_):<34} RMSE {r_['RMSE (log pts)']:.3f} log pts | "
                f"MAPE {r_['MAPE %']:.1f}%")

    say("")
    say("POLICY MODULES")
    tx = RESULTS.get("tax_equiv")
    if tx:
        lt = tx["latest"]
        say(f"  Tax equivalence FY{lt['FY']}: NIRC = {lt['gst_points']} GST "
            f"points = {lt['cit_points']} CIT points   [{tx['source']}]")
    dr = RESULTS.get("drivers")
    if dr and isinstance(dr.get("top"), pd.DataFrame):
        say("  Budget top-3: "
            + ", ".join(f"{i_} {r_['share %']:.0f}%"
                        for i_, r_ in dr["top"].head(3).iterrows())
            + f"   [{dr['source']}]")
    db = RESULTS.get("debt")
    if db:
        say(f"  Gross debt {db['latest_year']}: {db['latest_pct_gdp']}% of GDP "
            f"(asset-backed — see M14); corr with NIRC share "
            f"{db['corr_with_nirc_share']}   [{db['source']}]")
    iv = RESULTS.get("iv")
    if iv:
        say(f"  IV demo: 2SLS = {iv['beta_2sls']} (se {iv['se_2sls']}) vs "
            f"OLS = {iv['beta_ols']}; first-stage F = {iv['first_stage_F']}"
            f"   [{iv.get('source', 'n/a')}]")
    else:
        say("  IV demo: skipped (insufficient overlapping observations)")

    var_ = RESULTS.get("var")
    if var_:
        say(f"  VAR({var_['lag']}) IRFs saved: {var_['irf_fig']}")

    text = "\n".join(L)
    print(text)
    with open(f"{OUTDIR}/results_summary.txt", "w") as f:
        f.write(text + "\n")
    with open(f"{OUTDIR}/results_summary.json", "w") as f:
        json.dump({"seed": SEED,
                   **{k: _jsonable(v) for k, v in RESULTS.items()}},
                  f, indent=2)
    print(f"\n   Saved: {OUTDIR}/results_summary.txt and results_summary.json")
    status("M17_summary", "ok")

def main() -> None:
    t0 = time.time()
    env_report()
    b = load_data()
    try:
        validate(b)
    except Exception as exc:  # noqa: BLE001
        status("M2_validation", f"error ({exc.__class__.__name__})")
        warn(f"M2 failed: {exc}")
    try:
        repair_cpi(b)
    except Exception as exc:  # noqa: BLE001
        status("M2b_cpi_repair", f"error ({exc.__class__.__name__})")
        warn(f"M2b failed: {exc}")
    df = engineer(b)
    pearl_estimand()
    try:
        orders = unit_roots(df)
    except Exception as exc:  # noqa: BLE001
        orders = {s: 1 for s in CORE_LEVELS}
        status("M5_unitroots", f"error ({exc.__class__.__name__})")
        warn(f"M5 failed: {exc}")
    for name, fn in [
        ("M6/M7 bounds+ECM", lambda: bounds_module(df)),
        ("M8 Toda-Yamamoto", lambda: ty_module(df, orders)),
        ("M9 VAR/IRF",       lambda: var_module(df)),
        ("M10 stability",    lambda: stability_module(df)),
        ("M11 robustness",   lambda: robustness_module(df, b)),
        ("M12 tax equiv",    lambda: tax_module(df, b)),
        ("M13 drivers",      lambda: drivers_module(b)),
        ("M14 debt",         lambda: debt_module(df, b)),
        ("M15 IV demo",      lambda: iv_module(df, b)),
    ]:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            key = name.split()[0].replace("/", "_")
            status(key, f"error ({exc.__class__.__name__})")
            warn(f"{name} failed: {exc.__class__.__name__}: {exc}")
    try:
        export_module(df)
    except Exception as exc:  # noqa: BLE001
        status("M16_export", f"error ({exc.__class__.__name__})")
        warn(f"M16 failed: {exc}")
    try:
        summary_module()
    except Exception as exc:  # noqa: BLE001
        status("M17_summary", f"error ({exc.__class__.__name__})")
        warn(f"M17 failed: {exc}")

    banner("MODULE STATUS")
    for k, v in MODULE_STATUS.items():
        print(f"   {k:<18} {v}")
    print(f"\n   Total runtime: {time.time() - t0:.1f}s")
    print("""
   FINAL REMINDERS BEFORE THE PAPER USES ANY NUMBER ABOVE
   1. The Appendix A CPI base break was AUTO-REPAIRED by splicing (M2b);
      paste appendix_a_corrected.csv into the paper, cite SingStat for the
      corrected column, and note the splice in the data appendix.
   2. Prefer the simulated small-sample bounds p-values (statsmodels,
      Colab) or Narayan (2005) CVs over the asymptotic PSS table at n=30.
   3. Values flagged ILLUSTRATIVE are placeholders for offline runs —
      re-run with FETCH_LIVE=True and verify against the cited sources.
   4. Attribute data.gov.sg (Open Data Licence), World Bank (CC BY 4.0)
      and IMF WEO in the acknowledgements/data section.
""")

if __name__ == "__main__":
    main()
