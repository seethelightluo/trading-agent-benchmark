"""Trader block review for 2027-09-27..2027-10-11 cycle (executed 09-27 rebalance)."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

def load(a, n=40):
    df = get_stock_daily_data(a, days=n)
    if df is None or len(df) < 5:
        df = get_index_daily_data(a, days=n)
    return df

acc = get_account_dict()
nav = acc['net_assets']
pos = {p['symbol']: p for p in acc['positions']}

# execution price (2027-09-27) vs last close (2027-10-08)
block_ret = {}
for a in ASSETS:
    df = load(a)
    if df is None or len(df) < 12:
        print(a, 'NO DATA')
        continue
    df = df.sort_values('date').reset_index(drop=True)
    # find rows: execution date 09-27 and last date 10-08
    last = df.iloc[-1]
    mask = df['date'].astype(str).str.startswith('2027-09-27')
    if mask.any():
        ex = df[mask].iloc[-1]
        block_ret[a] = (last['close'] / ex['close'] - 1.0) * 100
    else:
        # fallback: use first row close as base
        base = df.iloc[0]['close']
        block_ret[a] = (last['close'] / base - 1.0) * 100

print(f"{'asset':10s} {'w_now':>7s} {'block_ret%':>9s} {'contrib%':>9s}")
total_contrib = 0.0
for a in ASSETS:
    w = pos[a]['market_value'] / nav if a in pos else 0.0
    r = block_ret.get(a, 0.0)
    contrib = w * r
    total_contrib += contrib
    print(f"{a:10s} {w*100:6.2f} {r:8.2f} {contrib:8.3f}")
print('sum contrib (approx block pnl, %):', round(total_contrib, 3))
print('net_assets:', round(nav, 2))
print('cash:', acc.get('available_cash'))
