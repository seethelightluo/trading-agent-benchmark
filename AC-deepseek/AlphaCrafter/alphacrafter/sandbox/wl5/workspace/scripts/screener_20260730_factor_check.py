"""Screener cycle 2026-07-30: verify persisted factor signals on visible data and
assess recent (last ~60/120 trading days) cross-sectional performance."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, load_panel)

close = closes_panel()
macro = macro_closes()
frames = load_panel()
rets = close.pct_change()
vix_ret = macro["VIX"].pct_change()

factors = {}
factors["mom_10d_skip5"] = close.shift(5) / close.shift(15) - 1.0
factors["mom_120d_skip5"] = close.shift(5) / close.shift(125) - 1.0

beta = {}
for s in close.columns:
    d = pd.concat([rets[s], vix_ret], axis=1).dropna()
    beta[s] = (d[s].rolling(60).cov(d["VIX"]) / d["VIX"].rolling(60).var()).reindex(close.index)
beta = pd.DataFrame(beta)
vix_chg = (macro["VIX"] / macro["VIX"].shift(20) - 1.0).reindex(close.index)
factors["vix_beta_cond_60x20"] = (-beta.mul(vix_chg, axis=0))
rv20 = rets.rolling(20).std()
factors["vol_of_vol20x60"] = rv20.rolling(60).std()

fr = forward_returns(close, 10)
print("=== FACTOR RECENT PERFORMANCE (cross-sectional rank IC, h=10) ===")
def stats(s):
    if s is None or len(s) < 10:
        return None
    ic = s.mean()
    icir = ic / s.std() * np.sqrt(len(s)) if s.std() > 0 else np.nan
    return dict(n=len(s), ic=round(float(ic), 4), icir=round(float(icir), 3),
                hit=round(float((s > 0).mean()), 3))

ic_map = {}
for name, f in factors.items():
    ics = ic_series(f, fr).dropna()
    ic_map[name] = ics
    full = stats(ics)
    recent120 = stats(ics[ics.index >= ics.index[-1] - pd.Timedelta(days=240)])
    recent60 = stats(ics[ics.index >= ics.index[-1] - pd.Timedelta(days=120)])
    print(f"\n{name}")
    print("  full     :", full)
    print("  last120td:", recent120)
    print("  last60td :", recent60)

print("\n=== pairwise IC-series correlation (full sample) ===")
df_ic = pd.DataFrame(ic_map)
print(df_ic.corr().round(3).to_string())
