"""miner_3 candidate 3: lottery-demand / tail asymmetry factor.

Motivation: in a cross-asset universe heavy on crypto and commodities, assets
with lottery-like payoff profiles (large up-tails relative to down-tails over a
recent window) tend to be overbought and underperform forward (negative expected
IC), while assets with large downside tails (fear) tend to bounce/be cheap
(positive IC for a reversed sign). We use max_daily_ret / |min_daily_ret| over a
window as an interpretable asymmetry proxy, plus 60d skewness as a robustness
check. Both are distinct from gain_loss_20 (which sums returns, not extremes).

IC gate: |IC| >= 0.007, |ICIR| >= 0.084 (10d horizon).
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

for w in [20, 60]:
    mx = ret.rolling(w).max()
    mn = ret.rolling(w).min()
    asym = mx / (mn.abs() + 1e-12)
    m = ic_stats(daily_spearman_ic(asym, fwd10))
    cov = coverage_stats(asym)
    to = rank_turnover(asym)
    dec = decay_ics(asym, px)
    asym_r = asym[asym.index >= '2032-10-01']
    fwd_r = fwd10.reindex(asym_r.index)
    m_r = ic_stats(daily_spearman_ic(asym_r, fwd_r))
    r_str = f" recent2y ic={m_r['ic']:.4f} icir={m_r['icir']:.3f}" if m_r else ' recent2y NONE'
    flag = 'PASS' if gate_pass(m) else 'FAIL'
    print(f'maxmin_ratio_{w}: full ic={m["ic"]:+.4f} icir={m["icir"]:+.3f} hit={m["ic_hit_ratio"]:.2f} '
          f'n={m["n_ic_dates"]} cov8={cov["coverage_dates_ge8"]:.2f} to={to:.3f} | {flag} |{r_str}')

# skewness 60d
for w in [30, 60]:
    sk = ret.rolling(w).skew()
    m = ic_stats(daily_spearman_ic(sk, fwd10))
    cov = coverage_stats(sk)
    to = rank_turnover(sk)
    sk_r = sk[sk.index >= '2032-10-01']
    fwd_r = fwd10.reindex(sk_r.index)
    m_r = ic_stats(daily_spearman_ic(sk_r, fwd_r))
    r_str = f" recent2y ic={m_r['ic']:.4f} icir={m_r['icir']:.3f}" if m_r else ' recent2y NONE'
    flag = 'PASS' if gate_pass(m) else 'FAIL'
    print(f'skew_{w}: full ic={m["ic"]:+.4f} icir={m["icir"]:+.3f} hit={m["ic_hit_ratio"]:.2f} '
          f'n={m["n_ic_dates"]} cov8={cov["coverage_dates_ge8"]:.2f} to={to:.3f} | {flag} |{r_str}')
    if w == 60:
        print_stats(f'skew_{w} detail', m, cov, to, decay_ics(sk, px))
