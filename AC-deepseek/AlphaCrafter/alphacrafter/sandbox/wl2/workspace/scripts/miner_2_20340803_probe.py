"""miner_2 2034-08-03 data probe: confirm visible range, volume availability, macro data."""
from alphacrafter.sim.utils import get_stock_daily_data
import pandas as pd

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

for s in ASSETS:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None:
        print(f'{s}: NO DATA')
        continue
    print(f'{s}: rows={len(df)} range={df["date"].iloc[0].date()}..{df["date"].iloc[-1].date()} '
          f'vol_nan={(df["volume"].isna().mean() if "volume" in df else 1.0):.3f} '
          f'vol_last={df["volume"].iloc[-1]:.0f}' if "volume" in df else f'{s}: no vol col')

print('--- macro ---')
for m in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    try:
        df = pd.read_csv(f'../persistent/index_data/{m}.csv', parse_dates=['date'])
        df = df.set_index('date').sort_index()
        print(f'{m}: rows={len(df)} range={df.index[0].date()}..{df.index[-1].date()} cols={list(df.columns)}')
    except Exception as e:
        print(f'{m}: ERR {e}')
