"""miner_1 exploration scan: novel factor candidates for deep-risk-off regime.

Current date 2028-10-24; visible data through 2028-10-23 (previous completed
trading day). Candidates are structurally distinct from the active library:
  A. usdjpy_beta_cond_60x20  - asset beta to USDJPY * USDJPY 20d momentum
  B. us10y_beta_cond_60x20   - asset beta to US10Y price * US10Y 20d momentum
  C. skew_20d_skip5          - rolling skewness of daily returns (20d, skip5)
  D. ret_autocorr_20d        - 1-day return autocorrelation (20d window)
  E. xau_beta_cond_60x20     - asset beta to XAU * XAU 20d momentum
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (ASSETS, load_close, load_macro, daily_ic, ic_stats,
                          summarize, rank_turnover, coverage_stats,
                          library_panel, max_lib_corr)

END = "2028-10-23"

close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
print(f"universe={len(close.columns)} assets, dates={len(close)} "
      f"({close.index[0].date()} -> {close.index[-1].date()})")


def beta_cond(asset_ret, signal, beta_win=60, cond_win=20, min_periods=30):
    sig_r = signal.pct_change()
    cov = asset_ret.rolling(beta_win, min_periods=min_periods).cov(sig_r)
    var = sig_r.rolling(beta_win, min_periods=min_periods).var()
    beta = cov.divide(var, axis=0)
    mom = signal / signal.shift(cond_win) - 1.0
    return beta.multiply(mom, axis=0)


cands = {}
# A. USDJPY-conditional beta (yen carry / risk-on gauge; USDJPY ~172 extreme)
cands["usdjpy_beta_cond_60x20"] = beta_cond(ret, macro["USDJPY"])
# B. US10Y-conditional beta (bond-trend conditioning; US10Y close ~ bond price)
cands["us10y_beta_cond_60x20"] = beta_cond(ret, close["US10Y"])
# C. return skewness 20d skip 5
cands["skew_20d_skip5"] = ret.shift(5).rolling(20, min_periods=12).skew()
# D. 1-day return autocorrelation (return persistence, distinct from level mom)
cands["ret_autocorr_20d"] = ret.rolling(20, min_periods=12).apply(
    lambda x: pd.Series(x).autocorr() if len(x) > 3 else np.nan, raw=False)
# E. XAU-conditional beta (gold-anchored risk-off)
cands["xau_beta_cond_60x20"] = beta_cond(ret, close["XAU"])

lib = library_panel(close, macro)
fwd10 = close.shift(-10) / close - 1.0

rows = []
for name, f in cands.items():
    st = summarize(f, close, horizons=(5, 10, 20))
    ic10 = st[10]
    cov_ = coverage_stats(f, fwd10)
    to = rank_turnover(f, window=10)
    rho, pairs = max_lib_corr(f, lib)
    rows.append(dict(factor=name, ic10=ic10["ic"], icir10=ic10["icir"],
                     hit10=ic10["hit"], n10=ic10["n"],
                     ic5=st[5]["ic"], ic20=st[20]["ic"],
                     cov_asset=cov_["coverage_asset_days"],
                     cov_dates=cov_["coverage_dates_ge8"],
                     turn=to, max_rho=round(rho, 3), pairs=pairs))

res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(res.to_string(index=False))
print("\n--- detail: top-3 rho pairs per candidate ---")
for _, r in res.iterrows():
    top = sorted(r["pairs"].items(), key=lambda kv: -abs(kv[1]))[:3]
    print(f"{r['factor']}: " + ", ".join(f"{k}={v}" for k, v in top))
