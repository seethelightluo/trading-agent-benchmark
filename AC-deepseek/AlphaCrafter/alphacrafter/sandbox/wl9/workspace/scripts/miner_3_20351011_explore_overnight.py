"""miner_3 exploration: overnight vs intraday return structure factors.
Universe: 15 tradable cross-asset instruments, full history truncated to visible_through.
"""
import pandas as pd, numpy as np

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
         'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
VIS_THROUGH = '2035-10-10'

def load(sym):
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv', parse_dates=['date'])
    df = df[df['date'] <= VIS_THROUGH].reset_index(drop=True)
    return df

frames = {s: load(s) for s in WATCH}
close = pd.DataFrame({s: frames[s].set_index('date')['close'] for s in WATCH})
openp = pd.DataFrame({s: frames[s].set_index('date')['open'] for s in WATCH})
# common index
idx = close.index.intersection(openp.index)
close = close.loc[idx]; openp = openp.loc[idx]
fwd_ret = close.shift(-10)/close - 1.0  # H=10 forward return

on_ret = openp/close.shift(1) - 1.0      # overnight return
intra_ret = close/openp - 1.0            # intraday return

def report(name, fac, H=10, neg=False):
    fwd = close.shift(-H)/close - 1.0
    fac = fac.replace([np.inf,-np.inf], np.nan)
    ic=[]
    for t in range(len(fac)):
        row = fac.iloc[t]; fw = fwd.iloc[t]
        m = row.notna() & fw.notna()
        if m.sum() >= 8:
            ic.append((t, row[m].rank().corr(fw[m].rank())))
    if len(ic) < 30:
        print(f'{name}: too few ({len(ic)})'); return
    s = pd.Series([x[1] for x in ic])
    icir = s.mean()/s.std()
    print(f'{name:26s} H={H} dates={len(s):4d} IC={s.mean():+.4f} ICIR={icir:+.4f} hit={(s>0).mean():.3f}')

print('=== candidates H=10 ===')
report('overnight_mean_20', on_ret.rolling(20,min_periods=12).mean())
report('intraday_mean_20', intra_ret.rolling(20,min_periods=12).mean())
report('on_minus_id_20', (on_ret-intra_ret).rolling(20,min_periods=12).mean())
print('=== H=5 ===')
report('overnight_mean_20', on_ret.rolling(20,min_periods=12).mean(), H=5)
report('on_minus_id_20', (on_ret-intra_ret).rolling(20,min_periods=12).mean(), H=5)
print('=== H=20 ===')
report('on_minus_id_20', (on_ret-intra_ret).rolling(20,min_periods=12).mean(), H=20)