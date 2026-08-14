"""miner_3 2035-08-20 - candidate: daily return serial correlation 20d (trend quality).

Motivation: autocorrelation of daily returns distinguishes trending vs mean-reverting
microstructure. Persisted/rejected momentum factors use multi-day cumulative returns;
serial correlation is the statistically orthogonal building block of trend
persistence. Sign-ambiguous a priori -> evaluate both signs.
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
print(f"closes {closes.shape} last {closes.index.max().date()}", flush=True)

# serial correlation of daily returns over 20d window (corr of r_t vs r_{t-1})
r1 = rets.shift(1)
sig = rets.rolling(20, min_periods=12).corr(r1)
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
    m = eval_sig("ac20", sig, direction)
    print(f"ac20 dir={direction:+d} | ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} "
          f"n={m['n_ic_dates']:5d} cov8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
          f"decay={m['decay_ic_by_horizon']}")

win = ("2033-08-20", "2035-08-17")
for direction in (1, -1):
    m = eval_sig("ac20_r2y", sig, direction, win)
    print(f"ac20_r2y dir={direction:+d} | ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} "
          f"n={m['n_ic_dates']:5d} cov8={m['coverage_dates_ge8']:.3f}")

lib = library_signals(panels, closes, rets)
corr, key = max_library_corr(sig, lib)
print(f"max_abs_library_correlation={corr} ({key})")
