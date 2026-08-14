"""miner_3 2035-08-20 - candidate: Parkinson range-vol ratio 20d (intraday structure).

Motivation: close-to-close vol ignores intraday information. Parkinson vol from
ln(H/L) captures true daily range. Assets whose intraday range is elevated relative
to their close-close vol (stress/illiquidity/efficiency gap) may carry a forward
risk premium or penalty. This is a distinct information set vs all persisted and
rejected vol factors (vol_ratio_20_60, vol_of_vol, vol_z_20 all use close returns
only). Direction discovered empirically (sign-ambiguous).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, forward_returns, rank_ic_series, summarize_ic,
    coverage_metrics, turnover_rank, decay_profile, max_library_corr,
    library_signals,
)

panels = load_panels(days=3500)
closes = close_panel(panels)
rets = closes.pct_change()
hi = pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns}, axis=1).reindex(closes.index)
lo = pd.concat({a: panels[a]["low"].astype(float) for a in closes.columns}, axis=1).reindex(closes.index)
print(f"closes {closes.shape} last {closes.index.max().date()}", flush=True)

# Parkinson vol: sqrt( mean( ln(H/L)^2 ) / (4 ln 2) ), 20d
park = (np.log(hi / lo)).pow(2).rolling(20, min_periods=10).mean()
park_vol = np.sqrt(park / (4.0 * np.log(2.0)))
close_vol = rets.rolling(20, min_periods=10).std()
sig = park_vol / close_vol.replace(0, np.nan)
sig = sig.replace([np.inf, -np.inf], np.nan)

def eval_sig(name, s, direction, window=None):
    s_use = s if window is None else s.loc[window[0]:window[1]]
    c_use = closes if window is None else closes.loc[window[0]:window[1]]
    fwd = forward_returns(c_use, 10)
    ics = rank_ic_series(s_use, fwd, min_valid=8)
    m = summarize_ic(ics, direction)
    m.update(coverage_metrics(s_use, min_valid=8))
    m["turnover_10d_rank"] = turnover_rank(s_use, 10)
    m["decay_ic_by_horizon"] = decay_profile(s_use, c_use, (1, 2, 3, 5, 10, 20), 8, direction)
    return m

for direction in (1, -1):
    m = eval_sig("park_ratio20", sig, direction)
    print(f"park_ratio20 dir={direction:+d} | ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} "
          f"n={m['n_ic_dates']:5d} cov8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
          f"decay={m['decay_ic_by_horizon']}")

win = ("2033-08-20", "2035-08-17")
for direction in (1, -1):
    m = eval_sig("park_ratio20_r2y", sig, direction, win)
    print(f"park_ratio20_r2y dir={direction:+d} | ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} "
          f"n={m['n_ic_dates']:5d} cov8={m['coverage_dates_ge8']:.3f}")

lib = library_signals(panels, closes, rets)
corr, key = max_library_corr(sig, lib)
print(f"max_abs_library_correlation={corr} ({key})")
