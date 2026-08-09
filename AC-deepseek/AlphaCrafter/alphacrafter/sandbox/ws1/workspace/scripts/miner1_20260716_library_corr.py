"""miner_1: compute passing candidates' h=10 metrics + pairwise cross-sectional
rank correlation to plan a diverse, non-redundant factor library."""
import numpy as np
import pandas as pd
from miner1_20260716_lib import (build_panel, factor_values, forward_returns,
                                 daily_ic, summarize)


def make_mom(lookback, skip=5):
    def fn(sym, close, volume):
        return close.shift(skip) / close.shift(skip + lookback) - 1.0
    return fn


def make_trend(win):
    def fn(sym, close, volume):
        ma = close.rolling(win).mean()
        return close / ma - 1.0
    return fn


def risk_adj_trend(win=20):
    def fn(sym, close, volume):
        ret = close.pct_change()
        mu = ret.rolling(win).mean()
        sd = ret.rolling(win).std()
        return (mu / sd).replace([np.inf, -np.inf], np.nan)
    return fn


def vol_of_vol(win=20, sub=60):
    def fn(sym, close, volume):
        sd = close.pct_change().rolling(win).std()
        return sd.rolling(sub).std()
    return fn


def efficiency_ratio(win=60):
    def fn(sym, close, volume):
        net = (close - close.shift(win)).abs()
        path = close.diff().abs().rolling(win).sum()
        return (net / path).replace([np.inf, -np.inf], np.nan)
    return fn


CANDIDATES = [
    ('mom_10d_skip5', make_mom(10)),
    ('mom_20d_skip5', make_mom(20)),
    ('mom_120d_skip5', make_mom(120)),
    ('trend_sma60', make_trend(60)),
    ('trend_sma120', make_trend(120)),
    ('risk_adj_trend20', risk_adj_trend(20)),
    ('vol_of_vol20x60', vol_of_vol()),
    ('efficiency_60d', efficiency_ratio(60)),
]

if __name__ == '__main__':
    panel = build_panel()
    closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
    h = 10
    ret = forward_returns(closes, grid, h)
    frames, metrics = {}, {}
    print('=== H=10 VALIDATION (admission gate |IC|>=0.007, |ICIR|>=0.084) ===')
    for label, fn in CANDIDATES:
        fac = factor_values(closes, volumes, grid, fn)
        frames[label] = fac
        ics = daily_ic(fac, ret, min_valid=8)
        m = summarize(ics, label, h)
        metrics[label] = m

    print('\n=== PAIRWISE CROSS-SECTIONAL RANK CORRELATION (mean over dates) ===')
    labels = [l for l, _ in CANDIDATES]
    corr = pd.DataFrame(index=labels, columns=labels, dtype=float)
    common_dates = None
    for l in labels:
        d = frames[l].dropna(how='all').index
        common_dates = d if common_dates is None else common_dates.intersection(d)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i > j:
                continue
            cs = []
            for t in common_dates:
                fa, fb = frames[a].loc[t], frames[b].loc[t]
                mask = fa.notna() & fb.notna() & np.isfinite(fa) & np.isfinite(fb)
                if mask.sum() >= 8:
                    r = pd.Series(fa[mask]).corr(pd.Series(fb[mask]), method='spearman')
                    if np.isfinite(r):
                        cs.append(r)
            v = float(np.mean(cs)) if cs else np.nan
            corr.loc[a, b] = v
            corr.loc[b, a] = v
    pd.set_option('display.width', 200)
    print(corr.round(3))
    print(f'\ncommon dates used: {len(common_dates)}')
