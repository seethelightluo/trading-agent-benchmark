"""Data QA for miner_3: check tradable asset data through visible_through."""
import pandas as pd
import numpy as np

VISIBLE = '2026-07-29'
ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

for name, folder in [('STOCK', '../persistent/stock_data'), ('OBS', '../persistent/index_data')]:
    files = ASSETS if name == 'STOCK' else OBS
    print(f'==== {name} ====')
    for a in files:
        try:
            df = pd.read_csv(f'{folder}/{a}.csv', parse_dates=['date'])
        except Exception as e:
            print(a, 'ERR', e)
            continue
        df = df[df['date'] <= pd.Timestamp(VISIBLE)].reset_index(drop=True)
        vol = df['volume']
        print(f'{a:10s} rows={len(df):5d} first={df["date"].iloc[0].date()} last={df["date"].iloc[-1].date()} '
              f'vol_nz={(vol>0).mean():.3f} nan_vol={vol.isna().mean():.3f} '
              f'close_nan={df["close"].isna().mean():.3f}')
        if a in ('SPX', 'BTC', 'US10Y', 'CN10Y'):
            print('   close tail:', df['close'].tail(3).round(4).tolist())
            print('   pct tail  :', df['pct_change'].tail(3).round(6).tolist())

# fundamental availability
df = pd.read_csv('../persistent/stock_data/000300.SH.csv', parse_dates=['date'])
df = df[df['date'] <= pd.Timestamp(VISIBLE)]
for c in ['PE', 'PS', 'PB', 'DYR']:
    print(c, 'non-null frac:', df[c].notna().mean())
