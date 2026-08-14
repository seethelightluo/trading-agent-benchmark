"""miner_3 2035-08-20 - candidate: Kaufman Efficiency Ratio 60d (trend efficiency).

Motivation: raw momentum factors (ts_mom, mom_*) capture direction but not path
structure. The Kaufman efficiency ratio |P_t - P_{t-n}| / sum(|daily ret|) isolates
how much of the cumulative move came from a smooth persistent trend vs choppy
oscillation. In cross-asset momentum regimes, high-efficiency assets (clean trends)
tend to keep trending (momentum continuation), while low-efficiency assets (noise)
should underperform. One factor family per script -> only ER tested here.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, forward_returns, rank_ic_series, summarize_ic,
    coverage_metrics, turnover_rank, decay_profile, max_library_corr,
    library_signals, TRADABLE,
)

panels = load_panels(days=3500)
closes = close_panel(panels)
rets = closes.pct_change()
print(f"closes {closes.shape} last {closes.index.max().date()}", flush=True)

# --- Kaufman efficiency ratio: |net move| / total path length, 60d window ---
n = 60
net = (closes - closes.shift(n)).abs()
path = rets.abs().rolling(n, min_periods=40).sum()
sig = net / path.replace(0, np.nan)
sig = sig.replace([np.inf, -np.inf], np.nan)

# full-sample + recent-2y evaluation, both signs
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
    m = eval_sig("er60", sig, direction)
    print(f"er60 dir={direction:+d} | ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} "
          f"n={m['n_ic_dates']:5d} cov8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
          f"decay={m['decay_ic_by_horizon']}")

# recent 2y
win = ("2033-08-20", "2035-08-17")
for direction in (1, -1):
    m = eval_sig("er60_r2y", sig, direction, win)
    print(f"er60_r2y dir={direction:+d} | ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.2f} "
          f"n={m['n_ic_dates']:5d} cov8={m['coverage_dates_ge8']:.3f}")

# library correlation (existing effective factors recomputed)
lib = library_signals(panels, closes, rets)
corr, key = max_library_corr(sig, lib)
print(f"max_abs_library_correlation={corr} ({key})")
