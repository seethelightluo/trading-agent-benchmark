"""Refine passing candidates efficiently: mutual correlation, regime IC, admission metrics."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner2_20260716_common import load_panel, load_index_panel, rank_ic, summarize_ic, forward_returns, MIN_INSTR

panel = load_panel()
ret = panel.pct_change()
idx = load_index_panel()
MP = lambda w: max(10, int(0.7 * w))
vix = idx['VIX']; vix_ret = vix.pct_change()
dxy_ret = idx['DXY'].pct_change()

def rolling_beta(y, x, window=60):
    df = pd.concat([y.rename('y'), x.rename('x')], axis=1).dropna()
    return df['y'].rolling(window, min_periods=MP(window)).cov(df['x']) / df['x'].rolling(window, min_periods=MP(window)).var()

F = {}
F['vix_beta_60d'] = pd.DataFrame({s: rolling_beta(ret[s], vix_ret.reindex(ret.index)) for s in panel.columns}, index=ret.index)
vol20 = ret.rolling(20, min_periods=MP(20)).std()
F['mom_vol_20d'] = (panel / panel.shift(20) - 1.0) / vol20
bond20 = panel['US10Y'] / panel['US10Y'].shift(20) - 1.0
F['vs_bond_20d'] = (panel / panel.shift(20) - 1.0).subtract(bond20, axis=0)
gold20 = panel['XAU'] / panel['XAU'].shift(20) - 1.0
F['vs_gold_20d'] = (panel / panel.shift(20) - 1.0).subtract(gold20, axis=0)
F['neg_volvol_20d'] = -vol20.rolling(20, min_periods=MP(20)).std()
sma60 = panel.rolling(60, min_periods=MP(60)).mean()
F['trend_60d'] = (panel / sma60 - 1.0) / vol20
F['dxy_beta_60d'] = pd.DataFrame({s: rolling_beta(ret[s], dxy_ret.reindex(ret.index)) for s in panel.columns}, index=ret.index)
F['range_pos_20d'] = ((panel / panel.rolling(20, min_periods=MP(20)).max()) - (panel / panel.rolling(20, min_periods=MP(20)).min())).rank(axis=1)
F['mom_20d'] = panel / panel.shift(20) - 1.0
F['accel_5_20'] = (panel / panel.shift(5) - 1.0) - (panel / panel.shift(20) - 1.0)
skew60 = ret.rolling(60, min_periods=MP(60)).skew()
F['neg_skew_60d'] = -skew60
F['shallow_dd_60d'] = -(panel / panel.rolling(60, min_periods=MP(60)).max() - 1.0)

# ---- mutual cross-sectional correlation (mean Spearman across dates), vectorized ranking ----
names = list(F.keys())
ranks = {}
for a in names:
    ra = F[a].rank(axis=1)
    ranks[a] = (ra - ra.mean(axis=1)) / ra.std(axis=1, ddof=0).replace(0, np.nan)

corr = pd.DataFrame(np.nan, index=names, columns=names)
for i, a in enumerate(names):
    for j in range(i, len(names)):
        b = names[j]
        if a == b:
            corr.loc[a, b] = 1.0
            continue
        pair = pd.concat([ranks[a], ranks[b]], axis=1, keys=['a', 'b'])
        pair = pair.replace([np.inf, -np.inf], np.nan)
        # date-wise corr
        pair['prod'] = pair['a'] * pair['b']
        valid = pair[['a', 'b']].notna().sum(axis=1)        # per-date valid count (could allow NaN values counted)
        mask = (pair[['a','b']].notna().sum(axis=1) >= MIN_INSTR)
        p = pair.loc[mask]
        if len(p) == 0:
            continue
        # pearson on ranks ~ spearman
        pm = p.dropna()
        co = pm['a'].corr(pm['b'])
        corr.loc[a, b] = co
        corr.loc[b, a] = co
print('=== mean cross-sectional Spearman corr between candidates ===')
print(corr.round(3).to_string())

# ---- admission metrics at h=10 and h=20 ----
rows = []
for name, f in F.items():
    for h in (10, 20):
        fwd = forward_returns(panel, h)
        ic_series, counts = rank_ic(f, fwd)
        s = summarize_ic(ic_series)
        rows.append({'factor': name, 'h': h, 'ic': s['mean_ic'], 'icir': s['icir'],
                     'hit': s['hit_ratio'], 'n': s['n_dates']})
t = pd.DataFrame(rows)
t['pass'] = (t['ic'].abs() >= 0.007) & (t['icir'].abs() >= 0.084)
pd.set_option('display.width', 200)
print('\n=== admission metrics ===')
print(t.to_string(index=False))
