"""miner_1 2034-07-20 data probe: confirm visible range, artifact shapes, macro data."""
import glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

def load_close():
    out = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=4000)
        if df is None or len(df) < 300:
            print('WARN no data', s)
            continue
        df = df.copy(); df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        out[s] = df['close'].astype(float)
    idx = None
    for s, ser in out.items():
        idx = ser.index if idx is None else idx.union(ser.index)
    idx = idx.sort_values()
    for s in out:
        out[s] = out[s].reindex(idx)
    return pd.DataFrame(out)

C = load_close()
print('grid shape:', C.shape, 'range:', C.index.min().date(), '->', C.index.max().date())
print('last 5 dates:', [str(d.date()) for d in C.index[-5:]])
print('columns:', list(C.columns))
print('coverage:', round(float(1 - C.isna().mean().mean()), 4))
print('flat-feed check last 60 rows (should be zero change):')
for s in ASSETS:
    tail = C[s].dropna().tail(60)
    chg = tail.pct_change().dropna()
    print(f'  {s:10s} n={len(tail):4d} last={tail.iloc[-1]:.4f} nonzero_chg={(chg.abs()>1e-12).sum()}')

print('\nartifact shapes:')
for f in sorted(glob.glob('factors/*.signal.npy'))[:40]:
    arr = np.load(f, allow_pickle=True)
    print(f'  {f.split("/")[-1]:35s} {arr.shape}')

print('\nmacro data (observation-only):')
for s in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    try:
        df = pd.read_csv(f'../persistent/index_data/{s}.csv', parse_dates=['date'])
        df = df[df['date'] <= C.index.max()]
        print(f'  {s:8s} n={len(df):5d} last={str(df["date"].iloc[-1].date())} close={df["close"].iloc[-1]:.4f}')
    except Exception as e:
        print('  ', s, 'ERR', e)
