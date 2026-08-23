"""
miner3_20270603_exp01_revalidate_all.py
Re-validate ALL 17 currently EFFECTIVE factors using data up to 2027-06-02.
Compute IC/ICIR at horizon=10 across the 15-asset cross-asset universe.
Flag any factor that fails the gates (|IC|<0.007 or |ICIR|<0.084) for deprecation.
"""
import sys, os, json, base64, io, zlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

DATA_DIR = Path("../persistent")
STOCK_DIR = DATA_DIR / "stock_data"
IDX_DIR = DATA_DIR / "index_data"
FACTOR_DIR = Path("factors")
VISIBLE_END = "2027-06-02"
CURRENT_DATE = "2027-06-03"

ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

print(f"=== REVALIDATION: {CURRENT_DATE} ===")
print(f"Visible through: {VISIBLE_END}")

# Load closes
closes = {}
for a in ASSETS:
    f = STOCK_DIR / f"{a}.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] <= VISIBLE_END]
    if len(df) >= 200:
        closes[a] = df.set_index("date")["close"].astype(float)
print(f"Loaded {len(closes)}/{len(ASSETS)} assets")

# Load macro
macro_data = {}
for m in MACRO:
    f = IDX_DIR / f"{m}.csv"
    if f.exists():
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= VISIBLE_END]
        macro_data[m] = df.set_index("date")["close"].astype(float)

# Align common dates
all_dates = sorted({d for s in closes.values() for d in s.index})
all_dates = [d for d in all_dates if d.weekday() < 5]
common = pd.DatetimeIndex(all_dates)
print(f"Total trading days: {len(common)}")

# Build panel arrays
assets_used = [s for s in ASSETS if s in closes]
N = len(assets_used)
T = len(common)
C = np.column_stack([closes[s].reindex(common).values for s in assets_used])
R = np.full((T, N), np.nan)
for t in range(1, T):
    R[t] = C[t] / C[t-1] - 1.0
R[0] = 0.0

# Forward returns at horizon=10
HORIZON = 10
fwd_ret = np.full((T, N), np.nan)
for t in range(T - HORIZON):
    fwd_ret[t] = C[t + HORIZON] / C[t] - 1.0

# Macro arrays
dxy = macro_data['DXY'].reindex(common).values if 'DXY' in macro_data else np.full(T, np.nan)
vix = macro_data['VIX'].reindex(common).values if 'VIX' in macro_data else np.full(T, np.nan)
cny = macro_data['USDCNY'].reindex(common).values if 'USDCNY' in macro_data else np.full(T, np.nan)
jpy = macro_data['USDJPY'].reindex(common).values if 'USDJPY' in macro_data else np.full(T, np.nan)

def rolling_mean(arr, w):
    out = np.full_like(arr, np.nan)
    for t in range(w-1, T):
        out[t] = np.nanmean(arr[t-w+1:t+1], axis=0)
    return out

def rolling_std(arr, w):
    out = np.full_like(arr, np.nan)
    for t in range(w-1, T):
        out[t] = np.nanstd(arr[t-w+1:t+1], axis=0, ddof=1)
    return out

def rolling_sum(arr, w):
    out = np.full_like(arr, np.nan)
    for t in range(w-1, T):
        out[t] = np.nansum(arr[t-w+1:t+1], axis=0)
    return out

def rolling_skew(arr, w):
    out = np.full_like(arr, np.nan)
    for t in range(w-1, T):
        seg = arr[t-w+1:t+1]
        m = np.nanmean(seg, axis=0)
        s = np.nanstd(seg, axis=0, ddof=1)
        n = np.sum(~np.isnan(seg), axis=0)
        skew = np.nanmean(((seg - m) / np.maximum(s,1e-10))**3, axis=0) * (n/(n-1))**(1.5) if n > 2 else np.nan
        out[t] = skew
    return out

def rolling_kurt(arr, w):
    out = np.full_like(arr, np.nan)
    for t in range(w-1, T):
        seg = arr[t-w+1:t+1]
        m = np.nanmean(seg, axis=0)
        s = np.nanstd(seg, axis=0, ddof=1)
        n = np.sum(~np.isnan(seg), axis=0)
        kurt = np.nanmean(((seg - m) / np.maximum(s,1e-10))**4, axis=0)
        out[t] = kurt - 3.0
    return out

def rank_ic_2d(factor, fwd, min_valid=8):
    ics = []
    for t in range(T):
        f = factor[t]
        r = fwd[t]
        valid = ~(np.isnan(f) | np.isnan(r))
        n = np.sum(valid)
        if n >= min_valid:
            fv = f[valid]
            rv = r[valid]
            if np.std(fv) > 1e-10 and np.std(rv) > 1e-10:
                rho, _ = spearmanr(fv, rv)
                if not np.isnan(rho):
                    ics.append(rho)
    if len(ics) < 10:
        return 0.0, 0.0, 0
    ic_arr = np.array(ics)
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr, ddof=1)
    icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
    return ic_mean, icir, len(ics)

results = {}
def evaluate_factor(name, factor_arr):
    ic, icir, n = rank_ic_2d(factor_arr, fwd_ret)
    passes = abs(ic) >= 0.0070 and abs(icir) >= 0.0840
    flag = "PASS" if passes else "FAIL"
    print(f"  {name:30s}  IC={ic:+.6f}  ICIR={icir:+.6f}  n={n:4d}  {flag}")
    results[name] = {"ic": ic, "icir": icir, "n_dates": n, "passed": passes}
    return ic, icir, n, passes

print(f"\n{'='*70}")
print(f"REVALIDATING ALL FACTORS (horizon={HORIZON}, gate: |IC|>=.007, |ICIR|>=.084)")
print(f"{'='*70}")

# ===== FACTOR 1: bb_width_20d =====
bbw = np.full((T, N), np.nan)
for t in range(19, T):
    seg = C[t-19:t+1]
    m = np.nanmean(seg, axis=0)
    s = np.nanstd(seg, axis=0, ddof=1)
    bbw[t] = 4 * s / np.maximum(m, 1e-10)
evaluate_factor('bb_width_20d', bbw)

# ===== FACTOR 2: kaufman_eff_20d