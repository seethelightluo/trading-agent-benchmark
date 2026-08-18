"""miner_1 scan round 2: skew/autocorr/persistence variants.

Goal: find a candidate passing |IC|>=0.007 & |ICIR|>=0.084 at h=10 with
max library rho clearly < 0.5 (robust margin vs quarantine gate).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, summarize, rank_turnover,
                          coverage_stats, library_panel, max_lib_corr)

END = "2028-10-23"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
fwd10 = close.shift(-10) / close - 1.0

cands = {}

def skew_variant(skip, win, minp):
    return ret.shift(skip).rolling(win, min_periods=minp).skew()

# skew variants
cands["skew_10d_skip3"] = skew_variant(3, 10, 6)
cands["skew_20d_skip5"] = skew_variant(5, 20, 12)
cands["skew_30d_skip10"] = skew_variant(10, 30, 15)
cands["skew_60d_skip10"] = skew_variant(10, 60, 30)
# skew of downside-only returns (left-tail asymmetry)
neg = ret.where(ret < 0, 0.0)
cands["skew_neg_20d_skip5"] = neg.shift(5).rolling(20, min_periods=12).skew()

# autocorrelation variants
cands["ret_autocorr_60d"] = ret.rolling(60, min_periods=30).apply(
    lambda x: pd.Series(x).autocorr() if len(x) > 3 else np.nan, raw=False)
# sign persistence: |net sign streak| fraction of up-days in window
up = (ret > 0).astype(float)
cands["up_ratio_20d_skip5"] = up.shift(5).rolling(20, min_periods=12).mean()
# negative-day clustering: mean abs return on down days / mean abs return on up days
up_abs = ret.where(ret > 0, 0.0).abs()
dn_abs = ret.where(ret < 0, 0.0).abs()
cands["asym_vol_20d"] = dn_abs.rolling(20, min_periods=12).mean() / \
    up_abs.rolling(20, min_periods=12).mean().replace(0, np.nan)

lib = library_panel(close, macro)
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
                     turn=to, max_rho=round(rho, 3)))

res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res.to_string(index=False))
print("\n--- top rho pair per candidate ---")
for name, f in cands.items():
    _, pairs = max_lib_corr(f, lib)
    top = sorted(pairs.items(), key=lambda kv: -abs(kv[1]))[:2]
    print(f"{name}: " + ", ".join(f"{k}={v}" for k, v in top))
