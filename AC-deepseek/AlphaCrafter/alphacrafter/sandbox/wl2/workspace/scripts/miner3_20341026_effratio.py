"""miner_3 candidate 1: Kaufman efficiency ratio (trend quality).

Motivation: raw momentum only measures net move; ER measures how much of the
realized path is directional (smooth trend) vs choppy (range). In a 15-instrument
cross-asset trend-following universe, high-ER (smooth trend) assets should keep
trending, low-ER (choppy) assets should be avoided. Distinct from momentum and
from vol (a steady climb has low vol and high ER, but ER is normalized by path
length, not vol).

IC gate: |IC| >= 0.007, |ICIR| >= 0.084 (10d horizon, daily cross-section).
Data cutoff 2034-10-25.
"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner3_20341026_lib_utils import (load_all, close_panel, ret_panel, forward_ret,
                                       daily_spearman_ic, ic_stats, coverage_stats,
                                       rank_turnover, decay_ics, print_stats, gate_pass)

data = load_all()
px = close_panel(data)
ret = ret_panel(data)
fwd10 = forward_ret(px, 10)
print(f'panel: {px.index[0].date()} .. {px.index[-1].date()} rows={len(px)} assets={len(px.columns)}')

path = ret.abs()

for n in [20, 40, 60, 120]:
    er = (px - px.shift(n)).abs() / path.rolling(n).sum()
    m = ic_stats(daily_spearman_ic(er, fwd10))
    cov = coverage_stats(er)
    to = rank_turnover(er)
    dec = decay_ics(er, px)
    # recent 2y
    er_r = er[er.index >= '2032-10-01']
    fwd_r = fwd10.reindex(er_r.index)
    m_r = ic_stats(daily_spearman_ic(er_r, fwd_r))
    r_str = f" recent2y ic={m_r['ic']:.4f} icir={m_r['icir']:.3f}" if m_r else ' recent2y NONE'
    flag = 'PASS' if gate_pass(m) else 'FAIL'
    print(f'eff_ratio_{n}: full ic={m["ic"]:+.4f} icir={m["icir"]:+.3f} hit={m["ic_hit_ratio"]:.2f} '
          f'n={m["n_ic_dates"]} cov8={cov["coverage_dates_ge8"]:.2f} to={to:.3f} | {flag} |{r_str}')
    if n == 60:
        print_stats(f'eff_ratio_{n} detail', m, cov, to, dec)
