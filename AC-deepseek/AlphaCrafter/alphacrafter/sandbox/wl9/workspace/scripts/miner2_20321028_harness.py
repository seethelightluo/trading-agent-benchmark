"""miner_2 cycle 2032-10-28: shared validation harness.

Reads ../persistent cross-asset panels. Visible through 2032-10-27 (current date 2032-10-28).
Gates (shared 15-name): |IC| >= 0.0070 AND |ICIR| >= 0.0840 at h=10.
Pure offline research. Reusable by factor-exploration scripts.
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

VISIBLE_END = "2032-10-27"
OOS_START = "2026-07-16"
HORIZON = 10
GATE_IC = 0.0070
GATE_ICIR = 0.0840


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
vix = mac["VIX"]
dxy = mac["DXY"]
cny = mac["USDCNY"]
usdjpy = mac["USDJPY"]
eurusd = mac["EURUSD"]

f10 = cdf.shift(-HORIZON) / cdf - 1.0


def eval_ic(fv):
    """fv: date x asset factor panel. Returns metrics evaluated on OOS window h=10."""
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
                ics.append(rho)
                dates.append(t)
    ic_arr = np.array(ics)
    if len(ic_arr) < 10:
        return dict(ic=0.0, icir=0.0, n_dates=len(ic_arr), abs_ic=0.0, hit=0.0)
    icm = ic_arr.mean()
    icstd = ic_arr.std(ddof=1)
    icir = icm / icstd if icstd > 1e-10 else 0.0
    pos = (ic_arr > 0).mean() if icm > 0 else (ic_arr < 0).mean()
    return dict(ic=float(icm), icir=float(icir), n_dates=int(len(ic_arr)),
                abs_ic=float(abs(ic_arr).mean()), hit=float(pos))


def eval_series(fv):
    """fv: date x asset panel. Runs eval_ic and returns metrics + full oos_ic series."""
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
                ics.append(rho)
                dates.append(t)
    res = eval_ic(fv)
    res["oos_ic_series"] = list(zip([str(d.date()) for d in dates], [float(x) for x in ics]))
    return res


def coverage(fv):
    ok = (~np.isnan(fv.values)).mean()
    return float(ok)
