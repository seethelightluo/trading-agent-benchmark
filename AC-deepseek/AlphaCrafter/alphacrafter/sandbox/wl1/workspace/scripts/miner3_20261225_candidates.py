"""miner_3: re-validate prior strong low-rho candidates + new ideas through 2026-12-24.

Prior candidates (cycle-8 screen, window through 2026-08): xs_dev_5 (ic=-0.065,
icir=-0.180, rho=0.467) and bond_beta_diff_60 (ic=-0.034, icir=-0.105, rho=0.017).
Check whether they still hold on the extended window and whether they pass gates
with correlation computed against the FULL effective library (incl. nclv family).
Also add new candidate ideas:
  - drawdown_120: -close/rolling_max(close,120) (distance from high)
  - mom_accel: momentum of momentum (10d vs 60d slope)
  - yield_spread_dyn: US10Y-CN10Y spread change * asset beta to spread
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_20261008_lib import load_close_panel

close = load_close_panel(days=2500)
print(f"panel dates={close.shape[0]} assets={close.shape[1]} "
      f"range={close.index.min().date()}..{close.index.max().date()}")
lr = close.pct_change()

# ---- Full effective library for correlation audit ----
def build_full_lib(close):
    ret = close.pct_change()
    lib = {
        "rev_1d": -(close / close.shift(1) - 1),
        "rev_2d": -(close / close.shift(2) - 1),
        "rev_3d": -(close / close.shift(3) - 1),
        "rev_5d": -(close / close.shift(5) - 1),
        "nclv_1d": (close / close.shift(1) - 1).rank(axis=1),
        "nclv_2d": (close / close.shift(2) - 1).rank(axis=1),
        "nclv_3d": (close / close.shift(3) - 1).rank(axis=1),
        "nclv_5d": (close / close.shift(5) - 1).rank(axis=1),
        "mom_120d_skip5": close.shift(5) / close.shift(125) - 1,
        "vol_of_vol20x60": ret.rolling(20).std().rolling(60).std(),
    }
    return lib

def max_lib_corr(f, lib):
    best = 0.0
    for name, lf in lib.items():
        corrs = []
        for dt in f.index:
            if dt not in lf.index:
                continue
            a, b = f.loc[dt], lf.loc[dt]
            m = a.notna() & b.notna()
            if m.sum() < 8:
                continue
            c = a[m].rank().corr(b[m].rank())
            if np.isfinite(c):
                corrs.append(c)
        if corrs:
            best = max(best, abs(np.mean(corrs)))
    return best

def validate(f, close, name, lib, horizon=5):
    fwd = close.shift(-horizon) / close - 1.0
    ics = []
    for dt in f.index:
        ff, rr = f.loc[dt], fwd.loc[dt]
        m = ff.notna() & rr.notna()
        if m.sum() < 8:
            continue
        ic = ff[m].rank().corr(rr[m].rank())
        if np.isfinite(ic):
            ics.append(ic)
    ics = pd.Series(ics)
    if len(ics) == 0:
        print(f"{name:28s} h={horizon} no data")
        return None
    ic = ics.mean(); icir = ic / ics.std(ddof=1) if ics.std(ddof=1) > 0 else 0
    hit = float((ics > 0).mean()) if ic > 0 else float((ics < 0).mean())
    rho = max_lib_corr(f, lib)
    ok = "PASS" if abs(ic) >= 0.007 and abs(icir) >= 0.084 and rho < 0.5 else "fail"
    print(f"{name:28s} h={horizon} IC={ic:+.5f} ICIR={icir:+.5f} hit={hit:.3f} "
          f"n={len(ics):5d} rho={rho:.3f}  {ok}")
    return {"ic": ic, "icir": icir, "hit": hit, "n": len(ics), "rho": rho}

lib = build_full_lib(close)

# ---- Candidate 1: xs_dev_5 (cross-sectional deviation of 5d return) ----
ret5 = close.pct_change(5)
factor = ret5.sub(ret5.mean(axis=1), axis=0)
print("=== xs_dev_5 ===")
for h in (3, 5, 10):
    validate(factor, close, "xs_dev_5", lib, horizon=h)

# ---- Candidate 2: bond_beta_diff_60 ----
def roll_beta(x, m, win=60, minp=30):
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        cov = x[s].rolling(win, min_periods=minp).cov(m)
        var = m.rolling(win, min_periods=minp).var()
        out[s] = cov / (var + 1e-12)
    return out

logr = np.log(close / close.shift(1))
b_us = roll_beta(logr, logr["US10Y"])
b_cn = roll_beta(logr, logr["CN10Y"])
f_bond = b_us - b_cn
print("=== bond_beta_diff_60 ===")
for h in (5, 10):
    validate(f_bond, close, "bond_beta_diff_60", lib, horizon=h)

# ---- Candidate 3: drawdown_120 (distance from 120d high) ----
f_dd = -(close / close.rolling(120).max() - 1)
print("=== drawdown_120 ===")
for h in (5, 10, 20):
    validate(f_dd, close, "drawdown_120", lib, horizon=h)

# ---- Candidate 4: mom_accel (10d mom minus 60d mom, skip5) ----
f_acc = close.shift(5) / close.shift(15) - 1 - (close.shift(5) / close.shift(65) - 1)
print("=== mom_accel_10x60 ===")
for h in (5, 10):
    validate(f_acc, close, "mom_accel_10x60", lib, horizon=h)

# ---- Candidate 5: yield_spread_beta (beta to US10Y-CN10Y spread change) ----
spread = np.log(close["US10Y"]) - np.log(close["CN10Y"])
dspr = spread.diff()
b_spr = roll_beta(logr, dspr)
print("=== spread_beta_60 ===")
for h in (5, 10):
    validate(b_spr, close, "spread_beta_60", lib, horizon=h)
