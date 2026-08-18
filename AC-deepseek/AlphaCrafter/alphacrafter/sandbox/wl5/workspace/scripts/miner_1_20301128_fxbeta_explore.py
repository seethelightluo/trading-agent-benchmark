# -*- coding: utf-8 -*-
"""miner_1 2030-11-28: explore sibling FX-beta candidates (USDJPY, EURUSD).

Motivation: cny_beta_60 passed admission (IC +0.0223, ICIR 2.88) in the last
cycle. Question: is CNY a unique FX driver, or is there a broader FX-beta
family (USDJPY, EURUSD) with similar or better predictive power? Validate the
same construction (rolling 60d beta of each asset's daily returns to the FX
pair's returns, min_periods=30, 10d forward horizon) for USDJPY and EURUSD,
and check redundancy vs cny_beta_60 / dxy_beta_60.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from miner_1_20301128_common import (
    WATCH, VISIBLE_THROUGH, CURRENT_DATE, ohlcv_panels, macro_panel,
    rank_ic_series, summarize_ic, decay_analysis, turnover_10d, coverage_stats,
    regime_split, roll_beta, library_correlation,
)

C = ohlcv_panels()['close']
R = C.pct_change()
fwd10 = C.shift(-10) / C - 1.0

refs = {
    "USDJPY": macro_panel("USDJPY"),
    "EURUSD": macro_panel("EURUSD"),
}

results = {}
for ref_name, ref in refs.items():
    Rref = ref.pct_change()
    beta = roll_beta(R, Rref, 60, 30)
    last = beta.index[-1]
    print(f"\n===== beta_60 to {ref_name} (as of {last.date()}) =====")
    print(beta.iloc[-1].sort_values(ascending=False).round(4).to_string())

    ic_s = rank_ic_series(beta, fwd10)
    print(f"\nYearly IC (10d horizon, {ref_name}):")
    for yr in range(2020, 2031):
        sub = ic_s[(ic_s.index >= pd.Timestamp(f'{yr}-01-01')) & (ic_s.index <= pd.Timestamp(f'{yr}-12-31'))]
        if len(sub):
            print(f'  {yr}: IC={sub.mean():+.4f} ICIR={sub.mean()/sub.std(ddof=1)*np.sqrt(len(sub)):+.2f} n={len(sub)}')

    m = summarize_ic(ic_s, f'{ref_name.lower()}_beta_60')
    reg = regime_split(ic_s)
    dec = decay_analysis(beta, C)
    cov = coverage_stats(beta)
    to = turnover_10d(beta)
    corrs, max_abs = library_correlation(beta, C,
        {'DXY': macro_panel('DXY'), 'VIX': macro_panel('VIX'), 'USDCNY': macro_panel('USDCNY')})
    print(f"\n=== {ref_name.lower()}_beta_60 summary ===")
    print('IC10=%.4f ICIR10=%.4f hit=%.3f n=%d' % (m['ic'], m['icir'], m['ic_hit_ratio'], m['n_ic_dates']))
    print('regimes:', {k: 'IC=%.4f/ICIR=%.2f/n=%d' % (v['ic'], v['icir'], v['n']) for k, v in reg.items()})
    print('decay:', {k: round(v, 4) for k, v in dec.items()})
    print('coverage_asset_days=%.3f ge8=%.3f turnover10d=%.3f' % (cov['coverage_asset_days'], cov['coverage_dates_ge8'], to))
    print('library_corr:', {k: round(v, 4) for k, v in corrs.items() if np.isfinite(v)})
    print('max_abs_library_correlation=%.4f' % max_abs)
    gate = abs(m['ic']) >= 0.0070 and abs(m['icir']) >= 0.0840
    print('GATE PASS: %s' % gate)
    results[ref_name] = {
        'factor_id': f'{ref_name.lower()}_beta_60', 'visible_through': VISIBLE_THROUGH,
        'metrics': m, 'regimes': reg, 'decay': dec, 'coverage': cov, 'turnover': to,
        'library_corr': corrs, 'max_abs_library_correlation': max_abs, 'gate': gate,
        'latest_cross_section': beta.iloc[-1].sort_values(ascending=False).round(4).to_dict(),
    }

# cross-correlation of the three FX betas themselves
print("\n===== FX-beta family cross-correlation (pooled) =====")
fx_betas = {}
for ref_name, ref in refs.items():
    fx_betas[ref_name.lower()] = roll_beta(R, ref.pct_change(), 60, 30)
fx_betas['cny'] = roll_beta(R, macro_panel('USDCNY').pct_change(), 60, 30)
fx_betas['dxy'] = roll_beta(R, macro_panel('DXY').pct_change(), 60, 30)
names = list(fx_betas.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a = fx_betas[names[i]].stack()
        b = fx_betas[names[j]].stack()
        df = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
        if len(df):
            print(f'  corr({names[i]}, {names[j]}) = {df["a"].corr(df["b"]):+.4f}  (n={len(df)})')

with open('scripts/miner_1_20301128_fxbeta_explore.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print('\nSaved scripts/miner_1_20301128_fxbeta_explore.json')
