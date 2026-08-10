"""miner_1 batch-2 factor exploration: trend/range, vol, volume, macro-conditional families."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, rank_ic_series, summarize_ic,
                                 coverage_metrics, turnover_rank, decay_profile,
                                 library_signals, max_library_corr, TRADABLE, MACRO)

panels = load_panels()
closes = close_panel(panels)
rets = closes.pct_change()
fwd10 = forward_returns(closes, 10)

# ---- build candidate panels ----
hi = {a: panels[a]['high'].astype(float) for a in TRADABLE}
lo = {a: panels[a]['low'].astype(float) for a in TRADABLE}
vo = {a: panels[a]['volume'].astype(float) for a in TRADABLE}
H = pd.concat(hi, axis=1).sort_index()
L = pd.concat(lo, axis=1).sort_index()
V = pd.concat(vo, axis=1).sort_index()

cands = {}
# range position family
for n in (14, 20, 60):
    hh = H.rolling(n).max(); ll = L.rolling(n).min()
    cands[f'range_pos_{n}'] = (closes - ll) / (hh - ll)
# distance from high
for n in (20, 60, 120):
    cands[f'dist_high_{n}'] = closes / closes.rolling(n).max() - 1.0
# bollinger z
for n in (20, 60):
    sma = closes.rolling(n).mean(); sd = closes.rolling(n).std()
    cands[f'bollz_{n}'] = (closes - sma) / sd
# momentum variants
for n in (20, 60):
    cands[f'mom_{n}d_skip5'] = closes.shift(5) / closes.shift(n + 5) - 1.0
cands['risk_adj_mom_60'] = cands['mom_60d_skip5'] / rets.rolling(20).std()
# vol family
cands['vol_20'] = rets.rolling(20).std()
cands['vol_60'] = rets.rolling(60).std()
cands['vol_ratio_5_60'] = rets.rolling(5).std() / rets.rolling(60).std()
cands['skew_20'] = rets.rolling(20).skew()
cands['kurt_60'] = rets.rolling(60).kurt()
cands['downside_vol_20'] = rets.clip(upper=0).rolling(20).std()
cands['range_amp_20'] = (H.rolling(20).max() - L.rolling(20).min()) / closes
# volume family
cands['vol_trend_20'] = V.rolling(20).mean() / V.rolling(60).mean() - 1.0
cands['vol_z_20'] = (V - V.rolling(20).mean()) / V.rolling(20).std()
vp_corr = {}
for a in closes.columns:
    z = pd.concat([rets[a].rename('r'), V[a].rename('v')], axis=1).dropna()
    vp_corr[a] = z['r'].rolling(20).corr(z['v'])
cands['vol_price_corr_20'] = pd.DataFrame(vp_corr, index=closes.index)

# macro conditional families
def cond_beta(asset_ret, macro_ret, win_beta=60):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename('a'), macro_ret.rename('m')], axis=1).dropna()
        beta[a] = z['a'].rolling(win_beta).cov(z['m']) / z['m'].rolling(win_beta).var()
    return pd.DataFrame(beta, index=asset_ret.index)

for m in MACRO:
    mret = panels[m]['close'].astype(float)
    if m in ('US10Y', 'CN10Y'):
        mret = panels[m]['close'].astype(float)
    mret = mret.pct_change()
    mmom = mret.rolling(20).mean()
    b = cond_beta(rets, mret, 60)
    cands[f'{m.lower()}_beta_cond_60x20'] = -b * mmom * 20.0  # sign: defensive if beta*+macro mom
    cands[f'{m.lower()}_beta_pos_60x20'] = b * mmom * 20.0

# ---- evaluate all ----
lib = library_signals(panels, closes, rets)
rows = []
for name, panel in cands.items():
    ics = rank_ic_series(panel, fwd10, 8)
    if len(ics) < 200:
        continue
    m = summarize_ic(ics, 1)
    m.update(coverage_metrics(panel))
    m['turnover_10d_rank'] = turnover_rank(panel, 10)
    corr, key = max_library_corr(panel, lib)
    m['max_lib_corr'] = corr
    m['max_corr_key'] = key
    m['name'] = name
    rows.append(m)

df = pd.DataFrame(rows).set_index('name')
cols = ['ic', 'icir', 'ic_hit_ratio', 'n_ic_dates', 'coverage_asset_days',
        'coverage_dates_ge8', 'turnover_10d_rank', 'max_lib_corr', 'max_corr_key']
df[cols].sort_values('icir', key=lambda s: s.abs(), ascending=False).to_csv('scripts/_batch2_results.csv')
print(df[cols].sort_values('icir', key=lambda s: s.abs(), ascending=False).to_string())
print()
print('=== PASS gate (|ic|>=0.007 & |icir|>=0.084) ===')
passing = df[(df.ic.abs() >= 0.007) & (df.icir.abs() >= 0.084)]
print(passing[cols].to_string() if len(passing) else 'none')
