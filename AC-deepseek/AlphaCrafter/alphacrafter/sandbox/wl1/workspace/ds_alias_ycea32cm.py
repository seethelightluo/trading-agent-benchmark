from alphacrafter.sim.utils import get_stock_daily_data
import pandas as pd

assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# block from 2030-12-20 to 2031-01-03; get 30 days data ending at last
print(f"{'asset':10s} {'px@12-19':>10s} {'px@01-03':>10s} {'ret%':>8s}")
rets = {}
for a in assets:
    df = get_stock_daily_data(symbol=a, days=30)
    if df is None or len(df) < 5:
        print(a, 'no data'); continue
    df = df.copy(); df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    # find close on 12-19 (last before decision) and last close
    d0 = df.index[df.index <= pd.Timestamp('2030-12-19')]
    d1 = df.index[-1]
    p0 = df.loc[d0[-1], 'close'] if len(d0) else None
    p1 = df.loc[d1, 'close']
    if p0 is not None:
        r = 100*(p1/p0 - 1)
        rets[a] = r
        print(f"{a:10s} {p0:10.4f} {p1:10.4f} {r:8.2f}")
print()
print('block-end weights (from account):')
