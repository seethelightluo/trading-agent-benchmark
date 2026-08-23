"""miner_2 factor exploration: 52-week range position (range_pos_250).
Factor = (close - rolling_min(close,250)) / (rolling_max(close,250) - rolling_min(close,250))
A longer-horizon trend signal distinct from the existing 20d relative momentum.
Cross-sectional IC vs forward 10d returns on the 15-instrument tradable universe.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU',
         'COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load_close_panel(days=5000):
    closes = {}
    for s in WATCH:
        df = get_stock_daily_data(s, days)
        if df is None or len(df) < 300:
            print('WARN insufficient', s, len(df) if df is not None else None)
            continue
        closes[s] = df.set_index('date')['close'].astype(float)
    panel = pd.DataFrame(closes).sort_index()
    panel = panel[~panel.index.duplicated(keep='last')]
    return panel

def range_pos(close, window=250):
    rmin = close.rolling(window, min_periods=window).min()
    rmax = close.rolling(window, min_periods=window).max()
    denom = (rmax - rmin)
    out = (close - rmin) / denom.replace(0, np.nan)
    return out

def fwd_returns(panel, horizon):
    return panel.shift(-horizon) / panel - 1.0

def cross_sectional_ic(factor_df, fwd_df, min_valid=8):
    dates, ics = [], []
    for dt in factor_df.index:
        f = factor_df.loc[dt]; r = fwd_df.loc[dt]
        m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if m.sum() >= min_valid:
            ics.append(f[m].corr(r[m], method='spearman'))
            dates.append(dt)
    return pd.Series(ics, index=dates)

def summarize(ic):
    ic = ic.dropna()
    if len(ic)==0: return {}
    mean = ic.mean(); std = ic.std(ddof=1)
    return {'n': len(ic), 'ic': mean, 'icir': mean/std if std>0 else 0,
            'hit': (ic>0).mean()}

panel = load_close_panel()
print('panel shape', panel.shape, 'dates', panel.index[0].date(), panel.index[-1].date())

factor_df = pd.DataFrame(index=panel.index, columns=WATCH)
for col in panel.columns:
    s = panel[col].dropna()
    factor_df[col] = range_pos(s).reindex(panel.index)

fwd10 = fwd_returns(panel, 10)
ic = cross_sectional_ic(factor_df, fwd10)
res = summarize(ic)
print('=== range_pos_250 ===')
print('n_dates', res.get('n'), 'IC %.4f'%res.get('ic',0), 'ICIR %.3f'%res.get('icir',0), 'hit %.3f'%res.get('hit',0))

ge8 = (factor_df.notna().sum(axis=1) >= 8).mean()
cov = factor_df.notna().sum().sum() / (factor_df.shape[0]*factor_df.shape[1])
print('coverage_ge8 %.3f coverage %.3f'%(ge8, cov))

# turnover 10d rank
ranks = factor_df.rank(axis=1)
r10 = ranks.shift(10)
chg = (ranks - r10).abs().mean(axis=1).dropna()
print('turnover_10d_rank %.3f'%chg.mean())

# decay
print('decay:', end=' ')
for h in [1,2,3,5,10,20]:
    icc = cross_sectional_ic(factor_df, fwd_returns(panel, h))
    print('%dd:%.4f'%(h, icc.mean()), end=' ')
print()

# warmup vs live
warm_end = pd.Timestamp('2026-07-15')
warm = ic[ic.index <= warm_end]; live = ic[ic.index > warm_end]
print('warmup', summarize(warm))
print('live ', summarize(live))
