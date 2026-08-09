"""Screening batch 2: macro-beta, trend, risk-adjusted and volatility-dynamics factors (miner_2)."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner2_20260716_common import load_panel, load_index_panel, evaluate_factor

panel = load_panel()
ret = panel.pct_change()
idx = load_index_panel()  # DXY, USDCNY, USDJPY, EURUSD, VIX
print(f'panel: {panel.shape[0]} dates; index panel: {idx.shape[0]} dates')

MP = lambda w: max(10, int(0.7 * w))
vix = idx['VIX']
vix_ret = vix.pct_change()
dxy_ret = idx['DXY'].pct_change()
usdjpy_ret = idx['USDJPY'].pct_change()
eurusd_ret = idx['EURUSD'].pct_change()

def rolling_beta(y, x, window=60):
    """rolling beta of y on x, aligned to y's index."""
    df = pd.concat([y.rename('y'), x.rename('x')], axis=1).dropna()
    beta = df['y'].rolling(window, min_periods=MP(window)).cov(df['x']) / df['x'].rolling(window, min_periods=MP(window)).var()
    return beta

factors = {}
# 1. VIX beta 60d (raw beta; sign to be determined empirically)
factors['vix_beta_60d'] = rolling_beta(ret.stack().rename('r'), vix_ret.reindex(ret.index).stack().rename('v')) \
    if False else pd.DataFrame({s: rolling_beta(ret[s], vix_ret.reindex(ret.index)) for s in panel.columns}, index=ret.index)
# 2. DXY beta 60d
factors['dxy_beta_60d'] = pd.DataFrame({s: rolling_beta(ret[s], dxy_ret.reindex(ret.index)) for s in panel.columns}, index=ret.index)
# 3. USDJPY beta 60d
factors['jpy_beta_60d'] = pd.DataFrame({s: rolling_beta(ret[s], usdjpy_ret.reindex(ret.index)) for s in panel.columns}, index=ret.index)
# 4. Risk-adjusted momentum 20d: mom20 / vol20
factors['mom_vol_20d'] = (panel / panel.shift(20) - 1.0) / ret.rolling(20, min_periods=MP(20)).std()
# 5. Vol-of-vol 20d: negative to prefer stable vol regimes
vol20 = ret.rolling(20, min_periods=MP(20)).std()
factors['neg_volvol_20d'] = -vol20.rolling(20, min_periods=MP(20)).std()
# 6. Serial correlation 5d (negative = reversal prone)
factors['neg_autocorr_5d'] = -ret.rolling(10, min_periods=MP(10)).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 3 else np.nan, raw=False)
# 7. Trend: distance of close from 60d SMA (normalized by vol)
sma60 = panel.rolling(60, min_periods=MP(60)).mean()
factors['trend_60d'] = (panel / sma60 - 1.0) / vol20
# 8. Bond-equity rotation: US10Y 20d return minus equity market 20d return, applied as common tilt? -> per-asset: asset 20d return minus bond 20d return (relative to bond)
bond20 = (panel['US10Y'] / panel['US10Y'].shift(20) - 1.0)
factors['vs_bond_20d'] = (panel / panel.shift(20) - 1.0).subtract(bond20, axis=0)
# 9. Vs gold 20d: asset 20d return minus gold 20d return (risk-on/off tilt)
gold20 = (panel['XAU'] / panel['XAU'].shift(20) - 1.0)
factors['vs_gold_20d'] = (panel / panel.shift(20) - 1.0).subtract(gold20, axis=0)
# 10. Downside vol 20d (semideviation, negative)
neg = ret.clip(upper=0)
factors['neg_downsidevol_20d'] = -neg.rolling(20, min_periods=MP(20)).std()
# 11. Amplitude: mean daily |return| over 20d (negative)
factors['neg_amplitude_20d'] = -ret.abs().rolling(20, min_periods=MP(20)).mean()

rows = []
for name, f in factors.items():
    res, to, tc, cov = evaluate_factor(f, panel, horizons=(5, 10, 20, 40))
    def g(h, k):
        return res[h][k] if res[h] else np.nan
    rows.append({
        'factor': name,
        'ic5': g(5, 'mean_ic'), 'icir5': g(5, 'icir'),
        'ic10': g(10, 'mean_ic'), 'icir10': g(10, 'icir'),
        'ic20': g(20, 'mean_ic'), 'icir20': g(20, 'icir'),
        'ic40': g(40, 'mean_ic'), 'icir40': g(40, 'icir'),
        'hit20': g(20, 'hit_ratio'), 'n20': g(20, 'n_dates'),
        'turnover': round(to, 4), 'coverage': round(cov['coverage'], 4),
    })

tbl = pd.DataFrame(rows).sort_values('ic20', key=lambda s: s.abs(), ascending=False)
pd.set_option('display.width', 240)
print(tbl.to_string(index=False))
