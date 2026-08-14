"""miner_1 2034-01-19 data probe: verify live data through previous completed day."""
from alphacrafter.sim.utils import get_stock_daily_data
import pandas as pd

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

for s in ASSETS:
    df = get_stock_daily_data(symbol=s, days=2000)
    if df is None:
        print(f"{s}: NO DATA")
        continue
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    last = df['date'].max()
    n = len(df)
    # recent 20d return
    closes = df.sort_values('date')['close'].astype(float)
    r20 = (closes.iloc[-1] / closes.iloc[-21] - 1) if len(closes) > 21 else None
    print(f"{s:10s} rows={n:5d} last={last.date()} r20={r20:+.4f}" if r20 is not None else f"{s:10s} rows={n:5d} last={last.date()} r20=NA")
