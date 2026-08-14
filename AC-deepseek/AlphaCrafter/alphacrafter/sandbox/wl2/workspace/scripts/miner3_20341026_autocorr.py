"""miner_3 candidate 2: AR(1) autocorrelation / trend persistence t-stat.

Motivation: positive serial correlation of daily returns = trending dynamics;
negative = mean reversion. The t-stat of an AR(1) slope over a trailing window
gives a normalized, sign-aware persistence measure. Cross-asset regimes differ
(trending commodities vs mean-reverting bonds), so the factor should separate
them. Distinct from efficiency ratio (path smoothness) and momentum (net move).

t = slope / se(slope) of regressing r_t on r_{t-1} over window w.
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


def ar1_tstat(r, w):
    out = {}
    for col in r.columns:
        s = r[col]
        x = s.shift(1)
        y = s
        df = pd.concat([x, y], axis=1).dropna()
        xv = df.iloc[:, 0].values
        yv = df.iloc[:, 1].values
        n = len(xv)
        t = np.full(n, np.nan)
        for i in range(w - 1, n):
            a = xv[i - w + 1:i + 1]
            b = yv[i - w + 1:i + 1]
            xm, ym = a.mean(), b.mean()
            sxx = ((a - xm) ** 2).sum()
            if sxx < 1e-14:
                continue
            slope = ((a - xm) * (b - ym)).sum() / sxx
            resid = b - (ym + slope * (a - xm))
            sse = (resid ** 2).sum()
            se = np.sqrt(sse / (w - 2) / sxx) if w > 2 and sse > 0 else np.nan
            t[i] = slope / se if se and np.isfinite(se) and se > 0 else np.nan
        out[col] = pd.Series(t, index=df.index)
    return pd.DataFrame(out).reindex(r.index)


for w in [20, 60, 120]:
    ac = ar1_tstat(ret, w)
    m = ic_stats(daily_spearman_ic(ac, fwd10))
    cov = coverage_stats(ac)
    to = rank_turnover(ac)
    dec = decay_ics(ac, px)
    ac_r = ac[ac.index >= '2032-10-01']
    fwd_r = fwd10.reindex(ac_r.index)
    m_r = ic_stats(daily_spearman_ic(ac_r, fwd_r))
    r_str = f" recent2y ic={m_r['ic']:.4f} icir={m_r['icir']:.3f}" if m_r else ' recent2y NONE'
    flag = 'PASS' if gate_pass(m) else 'FAIL'
    print(f'ar1_t_{w}: full ic={m["ic"]:+.4f} icir={m["icir"]:+.3f} hit={m["ic_hit_ratio"]:.2f} '
          f'n={m["n_ic_dates"]} cov8={cov["coverage_dates_ge8"]:.2f} to={to:.3f} | {flag} |{r_str}')
    if w == 60:
        print_stats(f'ar1_t_{w} detail', m, cov, to, dec)
