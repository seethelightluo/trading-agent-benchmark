"""miner_2 cycle 2032-09-16: re-validate ALL effective library factors on
fresh out-of-sample window 2026-07-16..2032-09-15 (online start onward).

Benchmark universe: 15 intentionally tradable cross-asset instruments.
Obs-only macros (DXY USDCNY USDJPY EURUSD VIX) used for computation only.
Gates (shared 15-name): |IC| >= 0.0070 AND |ICIR| >= 0.0840.
h=10 forward returns. Pure offline research: reads ../persistent csv, no live account.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = Path("../persistent")
STOCK_DIR = DATA_DIR / "stock_data"
IDX_DIR = DATA_DIR / "index_data"

ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

VISIBLE_END = "2032-09-15"
OOS_START = "2026-07-16"
HORIZON = 10
GATE_IC = 0.0070
GATE_ICIR = 0.0840

# factor_id -> direction as persisted in ensemble/library
FACTOR_DIRSIGN = {
 "beta_VIX_60": -1, "kaufman_eff_20d": 1, "mom_120d_skip5": 1, "bb_width_20d": 1,
 "cny_beta_60": 1, "vol_z_20d": 1, "ac1_120d": -1, "mom_10d_skip5": 1,
 "dxy_corr_change_20_60": 1, "skew_20d": 1, "days_since_high_60": -1,
 "kurt_20d": 1, "mom_10_vixreg": 1, "rng_pos_20d": 1, "streak_len_14": 1,
 "vix_beta_cond_60x20": -1, "vix_roc_20d": 1,
}

def load_panel(fname, end=VISIBLE_END):
    f = STOCK_DIR / fname
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] <= end].set_index("date")[["close"]].astype(float)
    return df["close"]

def load_macro(name, end=VISIBLE_END):
    f = IDX_DIR / f"{name}.csv"
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] <= end]
    return df.set_index("date")["close"].astype(float)

closes = {a: load_panel(f"{a}.csv") for a in ASSETS}
cdf = pd.DataFrame(closes).sort_index().astype(float)
r = cdf.pct_change()
mac = {m: load_macro(m) for m in MACRO}
vix = mac["VIX"]; dxy = mac["DXY"]; cny = mac["USDCNY"]; usdjpy = mac["USDJPY"]

f10 = cdf.shift(-HORIZON) / cdf - 1.0

def mk_ci(idx):
    # common datetime index for factor panels
    return pd.DatetimeIndex(sorted(set(idx)))

def eval_ic(fv):
    """fv: date x asset factor panel. returns dict of metrics."""
    ci = fv.index.intersection(f10.index)
    ics, dates = [], []
    for t in ci:
        if str(t.date()) < OOS_START:
            continue
        f = fv.loc[t].values.astype(float)
        fr = f10.loc[t].values.astype(float)
        ok = ~(np.isnan(f) | np.isnan(fr))
        if ok.sum() >= 8:
            rho, _ = spearmanr(f[ok], fr[ok])
            if not np.isnan(rho):
                ics.append(rho); dates.append(t)
    ic_arr = np.array(ics)
    if len(ic_arr) < 10:
        return dict(ic=0.0, icir=0.0, n_dates=len(ic_arr), hit=0.0, abs_ic=0.0)
    icm = ic_arr.mean(); icstd = ic_arr.std(ddof=1)
    icir = icm/icstd if icstd > 1e-10 else 0.0
    hit = float((ic_arr > 0).mean()) if icm > 0 else float((ic_arr < 0).mean())
    return dict(ic=float(icm), icir=float(icir), n_dates=int(len(ic_arr)),
                abs_ic=float(abs(ic_arr).mean()), hit=float(hit))

def beta_panel(ref, window=60):
    ci = cdf.index.intersection(ref.index)
    out = pd.DataFrame(index=ci, columns=cdf.columns, dtype=float)
    for s in cdf.columns:
        ri = r[s].reindex(ci)
        rr = ref.reindex(ci).pct_change()
        beta = ri.rolling(window).cov(rr) / rr.rolling(window).var()
        out.loc[ci, s] = beta.values
    return out

def report(nm, fv):
    m = eval_series(fv)
    sign = FACTOR_DIRS.get(nm, 1)
    # interpret metric in signed-effective direction: pass if effective-layer IC/CIR meet gate