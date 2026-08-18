"""Trader post-step review: reconstruct decision-time regime and proposed target
for the 2029-12-27 -> 2030-01-10 block."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

def get(a, n=300):
    try:
        return get_index_daily_data(a, days=n)
    except Exception:
        return None

panel = {}
for a in WATCH:
    df = get(a)
    if df is not None and len(df) > 30:
        panel[a] = df.set_index('date')['close']
panel = pd.DataFrame(panel).sort_index()
print('panel rows:', len(panel), 'last date:', panel.index[-1].date())

# regime at last decision (block start 2029-12-27) - use data through previous close
rets = panel.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean())
v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
regime = 'bull' if trend > 1.0 else ('bear' if trend < -1.0 else 'sideways')
print(f'decision-date trend t-stat: {trend:.3f} regime={regime}')

# per-asset 10d (block) returns at decision date
blk = panel.iloc[-11:-1]  # 10 completed days before decision (2029-12-13..2029-12-26 approx)
r10 = (blk.iloc[-1] / blk.iloc[0] - 1.0) * 100
print('\n10d returns to decision:')
for a in WATCH:
    print(f'  {a:10s} {r10[a]:+7.2f}%')

# end-of-block account weights and PnL attribution
acct = get_account_dict()
nav = acct['net_assets']
print(f'\nnet_assets={nav:.2f}')
tot = 0.0
for p in sorted(acct['positions'], key=lambda x: -abs(x['profit_loss'])):
    w = p['market_value'] / nav * 100
    print(f"  {p['symbol']:10s} w={w:6.2f}% pnl={p['profit_loss']:+10.2f}")
    tot += p['profit_loss']
print('sum pnl:', round(tot, 2))
