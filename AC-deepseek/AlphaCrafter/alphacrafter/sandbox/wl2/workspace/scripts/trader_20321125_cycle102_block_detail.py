"""Cycle102 (11-11 -> 11-25) per-asset block detail for memory log."""
import json
from alphacrafter.sim.utils import get_stock_daily_data

WL = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
      'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

# executed target weights at block start (from account last_executed_target_weights)
ew = json.load(open('../persistent/account.json'))['last_executed_target_weights']

# NAV at end of cycle101 (memory): 1,251,054 at 11-11; current NAV from account
na = json.load(open('../persistent/account.json'))
nav_now = float(na['net_assets'])
nav_start = 1251054.0

print(f"NAV start(11-11)={nav_start:.2f} NAV now(11-25)={nav_now:.2f} "
      f"period ret={(nav_now/nav_start-1)*100:.3f}%")

rows = []
for s in WL:
    df = get_stock_daily_data(s, days=30)
    if df is None or len(df) < 15:
        print(s, 'NO DATA')
        continue
    df = df.sort_values('date')
    dates = [str(x)[:10] for x in df['date']]
    closes = list(df['close'])
    # find close on/after 11-11 and last close
    c_start = None
    for dt, c in zip(dates, closes):
        if dt >= '2032-11-11':
            c_start = c
            break
    c_end = closes[-1]
    if c_start is None or c_start == 0:
        print(s, 'START NOT FOUND', dates[-3:])
        continue
    r = c_end / c_start - 1.0
    w = ew.get(s, 0.0)
    rows.append((s, w, r, w * r, c_start, c_end, dates[-1]))

rows.sort(key=lambda x: -abs(x[3]))
tot_contrib = sum(r[3] for r in rows)
print(f"sum(weight*ret) approx = {tot_contrib*100:.3f}%")
print(f"{'asset':10s} {'wt':>7s} {'ret%':>8s} {'contrib%':>9s}  px_start px_end")
for s, w, r, c, p0, p1, dlast in rows:
    print(f"{s:10s} {w*100:6.2f} {r*100:7.2f} {c*100:8.3f}  {p0:9.2f} {p1:9.2f} last={dlast}")
