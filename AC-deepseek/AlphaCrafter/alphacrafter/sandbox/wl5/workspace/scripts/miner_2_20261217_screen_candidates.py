# -*- coding: utf-8 -*-
"""miner_2 2026-12-17: screen candidate factor ideas on 15-asset cross-asset universe.
Screens: trend efficiency, range position, vol term structure, drawdown depth,
momentum acceleration, jump intensity, parkinson efficiency, downside beta,
risk-adjusted momentum (sharpe), skewness. Prints compact IC/ICIR/rho table.
Data used: closes through 2026-12-16 (prev completed day; current 2026-12-17).
"""
import sys, math
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

# ensure kurt_20 is included in library correlation set
L.LIB_FACTORS = [f for f in L.LIB_FACTORS if f != 'kurt_20'] + ['kurt_20']

C, V, H, Lw, O = L.load_close_panel(4000)
ret = C.pct_change()
fwd10 = ret.shift(-10)

def ic_table(fac, label):
    s = L.rank_ic(fac, fwd10)
    if s is None or len(s) < 30:
        print(f"{label:28s} n={0 if s is None else len(s)} INSUFFICIENT")
        return
    ic = s.mean(); icir = s.mean()/s.std() if s.std()>0 else 0
    hit = (s>0).mean()
    rhos, maxrho = L.library_max_rho(fac)
    # decay at 1,5,10,20
    dec = {}
    for h in (1,5,10,20):
        ss = L.rank_ic(fac, ret.shift(-h))
        dec[h] = round(ss.mean(),4) if (ss is not None and len(ss)>=20) else float('nan')
    cov = fac.notna().mean().mean()
    print(f"{label:28s} n={len(s):4d} IC={ic:+.4f} ICIR={icir:+.3f} hit={hit:.3f} "
          f"maxrho={maxrho:+.3f} dec1={dec[1]:+.4f} dec5={dec[5]:+.4f} dec10={dec[10]:+.4f} dec20={dec[20]:+.4f} cov={cov:.2f}")

def rolling_vol(x, w):
    return x.rolling(w).std(ddof=0)

# 1) Trend efficiency ratio (Kaufman) 30d / 60d
def eff_ratio(w):
    num = (C - C.shift(w)).abs()
    den = C.diff().abs().rolling(w).sum()
    return (num / den).clip(0, 1)

# 2) Range position: mean of daily (close-low)/(high-low) over w
def range_pos(w):
    rp = (C - Lw) / (H - Lw).replace(0, np.nan)
    return rp.rolling(w).mean()

# 3) Vol term structure: rv5/rv60 - 1, rv10/rv60 - 1
def vol_term(w1, w2):
    rv1 = rolling_vol(ret, w1) * np.sqrt(252)
    rv2 = rolling_vol(ret, w2) * np.sqrt(252)
    return (rv1 / rv2) - 1.0

# 4) Drawdown depth from 60d / 120d high
def dd_depth(w):
    return (C / C.rolling(w).max()) - 1.0

# 5) Momentum acceleration: (ret60-ret20), (ret20-ret10), skip5
def mom(w, skip=5):
    return C.shift(skip) / C.shift(skip + w) - 1.0

# 6) Jump intensity: count of |ret|>1.5*vol60 in 60d
def jump_count(w=60, k=1.5):
    vol = rolling_vol(ret, w)
    j = (ret.abs() > k * vol).astype(float)
    return j.rolling(w).sum()

# 7) Parkinson efficiency: realized vol / parkinson vol (close-based vs range-based)
def parkinson_eff(w):
    rv = rolling_vol(ret, w) * np.sqrt(252)
    hl = np.log(H / Lw)
    pv = np.sqrt((hl ** 2).rolling(w).mean() / (4 * np.log(2))) * np.sqrt(252)
    return rv / pv

# 8) Downside beta vs SPX (60d, SPX<0 days)
def downside_beta(w=60):
    spx = C['SPX'].pct_change()
    out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
    for s in C.columns:
        a = C[s].pct_change()
        m = (spx < 0) & a.notna() & spx.notna()
        cov = (a * spx).where(m).rolling(w).mean() - a.where(m).rolling(w).mean() * spx.where(m).rolling(w).mean()
        var = (spx ** 2).where(m).rolling(w).mean() - spx.where(m).rolling(w).mean() ** 2
        out[s] = cov / var
    return out.replace([np.inf, -np.inf], np.nan)

# 9) Risk-adjusted momentum (sharpe-like) 60d
def sharpe60():
    mu = ret.rolling(60).mean()
    sd = rolling_vol(ret, 60)
    return mu / sd

# 10) Skewness 60d
def skew(w=60):
    return ret.rolling(w).skew()

print("=== CANDIDATE SCREEN (admission horizon 10; gate |IC|>=0.007, |ICIR|>=0.084) ===")
print(f"data through {C.index.max().date()}\n")
cands = [
    ("eff_ratio_30", eff_ratio(30)),
    ("eff_ratio_60", eff_ratio(60)),
    ("range_pos_10", range_pos(10)),
    ("range_pos_20", range_pos(20)),
    ("vol_term_5x60", vol_term(5, 60)),
    ("vol_term_10x60", vol_term(10, 60)),
    ("dd_depth_60", dd_depth(60)),
    ("dd_depth_120", dd_depth(120)),
    ("mom_accel_60x20", mom(60) - mom(20)),
    ("mom_accel_20x10", mom(20) - mom(10)),
    ("jump_count_60", jump_count(60, 1.5)),
    ("parkinson_eff_20", parkinson_eff(20)),
    ("downside_beta_60", downside_beta(60)),
    ("sharpe_60", sharpe60()),
    ("skew_60", skew(60)),
]
for name, fac in cands:
    try:
        ic_table(fac, name)
    except Exception as e:
        print(name, "ERROR", e)
