"""Refine passing candidates: cross-sectional mutual correlation, regime IC, and admission metrics."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner2_20260716_common import load_panel, load_index_panel, evaluate_factor, rank_ic, summarize_ic, forward_returns, MIN_INSTR

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

# ---- mutual cross-sectional correlation (mean Spearman across dates) ----
names = list(F.keys())
corr = pd.DataFrame(np.nan, index=names, columns=names)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j <= i:
            continue
        vals = []
        common = F[a].join(F[b], lsuffix='_a', rsuffix='_b')
        for dt, row in common.iterrows():
            x = row[[c for c in common.columns if c.endswith('_a')]].values
            y = row[[c for c in common.columns if c.endswith('_b')]].values
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < MIN_INSTR:
                continue
            if np.all(x[m] == x[m][0]) or np.all(y[m] == y[m][0]):
                continue
            c = pd.Series(x[m]).corr(pd.Series(y[m]), method='spearman')
            if np.isfinite(c):
                vals.append(c)
        corr.loc[a, b] = np.nanmean(vals) if vals else np.nan
        corr.loc[b, a] = corr.loc[a, b]
corr = corr.copy(); np.fill_diagonal(corr.values, 1.0)
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
