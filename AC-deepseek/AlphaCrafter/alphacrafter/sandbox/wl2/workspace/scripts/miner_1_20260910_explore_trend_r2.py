"""miner_1 2026-09-10: Factor A exploration - trend linearity R2 of log price on time (vectorized).

Idea: assets whose price path is well described by a linear trend (high R2) tend to
persist; choppy paths (low R2) revert. Distinct from momentum (magnitude) and
calmness (vol regime).
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from miner_1_20260910_utils import (load_panel, align_close, forward_returns,
                                    daily_ic, summarize_ic, turnover_rank, coverage)

panel = load_panel(days=2500)
close = align_close(panel)
print(f'panel assets={len(panel)}  close dates={close.shape[0]}x{close.shape[1]}  '
      f'range {close.index.min().date()}..{close.index.max().date()}')
fwd = forward_returns(close, 10)
print('fwd10 non-null per asset:', fwd.notna().sum().min(), '..', fwd.notna().sum().max())


def trend_r2(logp, window):
    """Rolling R2 = corr(logp, t)^2 using rolling cov/var (vectorized)."""
    t = pd.Series(np.arange(len(logp)), index=logp.index, dtype=float)
    # rolling corr per column of logp with t
    def _col_r2(x):
        r = x.rolling(window, min_periods=max(10, window // 2)).corr(t)
        return r ** 2
    out = logp.apply(_col_r2)
    return out


def trend_tstat(logp, window):
    """t-stat of linear trend slope b/se via rolling statistics."""
    n = len(logp)
    t = pd.Series(np.arange(n), index=logp.index, dtype=float)
    def _col_ts(x):
        w = max(10, window // 2)
        var_t = t.rolling(window, min_periods=w).var()
        cov = x.rolling(window, min_periods=w).cov(t)
        var_y = x.rolling(window, min_periods=w).var()
        b = cov / var_t
        # R2 = cov^2/(var_y*var_t)
        r2 = (cov ** 2) / (var_y * var_t)
        n_eff = x.rolling(window, min_periods=w).count()
        ss_res = var_y * (n_eff - 1) * (1 - r2)
        sxx = var_t * (n_eff - 1)
        se = np.sqrt(ss_res / ((n_eff - 2) * sxx))
        return b / se
    return logp.apply(_col_ts)


logp = np.log(close)
results = {}
for win in [20, 40, 60, 120]:
    r2 = trend_r2(logp, win)
    cov, d8 = coverage(r2, close)
    ics = daily_ic(r2, fwd, min_assets=8)
    s = summarize_ic(ics, f'trend_r2_{win}')
    print(f'  coverage={cov:.3f} dates_ge8={d8:.3f} turnover10={turnover_rank(r2):.3f}')
    results[f'trend_r2_{win}'] = s
for win in [20, 60]:
    ts = trend_tstat(logp, win)
    cov, d8 = coverage(ts, close)
    ics = daily_ic(ts, fwd, min_assets=8)
    s = summarize_ic(ics, f'trend_tstat_{win}')
    print(f'  coverage={cov:.3f} dates_ge8={d8:.3f} turnover10={turnover_rank(ts):.3f}')
    results[f'trend_tstat_{win}'] = s

print('\nDONE')
