"""Verify vectorized rank-corr vs per-date loop for a few pairs; print full matrix."""
import numpy as np
import pandas as pd
from miner1_20260716_lib import build_panel, factor_values

panel = build_panel()
closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']

CANDIDATES = {
    'mom_10d_skip5': lambda s, c, v: c.shift(5) / c.shift(15) - 1.0,
    'mom_20d_skip5': lambda s, c, v: c.shift(5) / c.shift(25) - 1.0,
    'mom_120d_skip5': lambda s, c, v: c.shift(5) / c.shift(125) - 1.0,
    'trend_sma60': lambda s, c, v: c / c.rolling(60).mean() - 1.0,
    'trend_sma120': lambda s, c, v: c / c.rolling(120).mean() - 1.0,
    'risk_adj_trend20': lambda s, c, v: (c.pct_change().rolling(20).mean()
                                         / c.pct_change().rolling(20).std()
                                         ).replace([np.inf, -np.inf], np.nan),
    'vol_of_vol20x60': lambda s, c, v: c.pct_change().rolling(20).std().rolling(60).std(),
    'zscore_rev_20d': lambda s, c, v: ((c - c.rolling(20).mean())
                                       / (c.pct_change().rolling(20).std() * c)
                                       ).replace([np.inf, -np.inf], np.nan),
    'inv_vol_60d': lambda s, c, v: (-c.pct_change().rolling(60).std()
                                    ).replace([np.inf, -np.inf], np.nan),
}
frames = {l: factor_values(closes, volumes, grid, fn) for l, fn in CANDIDATES.items()}

# full matrix (vectorized)
labels = list(CANDIDATES.keys())
corr = pd.DataFrame(index=labels, columns=labels, dtype=float)
for a in labels:
    for b in labels:
        if a == b:
            corr.loc[a, b] = 1.0
            continue
        dfa = frames[a].where(frames[a].notna() & frames[b].notna())
        dfb = frames[b].where(frames[a].notna() & frames[b].notna())
        ok = dfa.notna().sum(axis=1) >= 8
        corr.loc[a, b] = dfa[ok].corrwith(dfb[ok], axis=1).mean()
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)
print(corr.round(3))

# per-date loop check on 300 random common dates for (zscore_rev_20d, mom_20d_skip5), (inv_vol_60d, vol_of_vol20x60), (zscore vs risk_adj_trend20)
rng = np.random.default_rng(42)
cd = frames['zscore_rev_20d'].index.intersection(frames['mom_20d_skip5'].index)
dates = pd.DatetimeIndex(rng.choice(cd, size=300, replace=False))
for a, b in [('zscore_rev_20d', 'mom_20d_skip5'), ('zscore_rev_20d', 'risk_adj_trend20'),
             ('inv_vol_60d', 'vol_of_vol20x60'), ('zscore_rev_20d', 'trend_sma60'),
             ('inv_vol_60d', 'zscore_rev_20d')]:
    cs = []
    for t in dates:
        x, y = frames[a].loc[t], frames[b].loc[t]
        m = x.notna() & y.notna()
        if m.sum() >= 8:
            r = pd.Series(x[m]).corr(pd.Series(y[m]), method='spearman')
            if np.isfinite(r):
                cs.append(r)
    print(f'LOOP check {a} vs {b}: mean_spearman={np.mean(cs):.3f} (n={len(cs)})')
