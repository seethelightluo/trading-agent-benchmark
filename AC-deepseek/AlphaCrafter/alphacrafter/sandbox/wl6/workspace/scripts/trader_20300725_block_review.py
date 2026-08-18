"""Trader post-step review 2030-07-25: per-asset block returns + contributions.

Read-only diagnostic: computes returns from close prices over the block
2030-07-11 -> 2030-07-25 and cross-checks account state.
"""
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

acc = json.load(open('../persistent/account.json'))
na = acc.get('net_assets')
print('net_assets now: %.2f' % na)

prev_na = 1040638.0
print('block PnL: %.2f  block return: %.4f%%' % (na - prev_na, 100*(na/prev_na - 1)))

assets = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

# Approx executed target from probe (2030-07-11 decision, visible through 07-10)
tgt = {'000300.SH': 0.1200, 'SPX': 0.1200, 'US10Y': 0.1200, 'N225': 0.1153,
       'SX5E': 0.1142, 'XAU': 0.1027, 'NDX': 0.0991, 'WTI': 0.0800,
       'ETH': 0.0800, 'SOX': 0.0122, 'COPPER': 0.0122, 'BTC': 0.0092,
       'HSI': 0.0050, '000688.SH': 0.0050, 'CN10Y': 0.0050}

rows = []
for a in assets:
    df = get_stock_daily_data(a, days=40) or get_index_daily_data(a, days=40)
    if df is None or len(df) < 12:
        print(a, 'no data'); continue
    df = df.sort_values('date').reset_index(drop=True)
    # block: close on 2030-07-10 (decision day last bar) -> close on 2030-07-25
    d10 = df[df['date'] <= '2030-07-10']
    d25 = df[df['date'] <= '2030-07-25']
    if len(d10) == 0 or len(d25) == 0:
        print(a, 'window missing'); continue
    c0 = float(d10['close'].iloc[-1]); c1 = float(d25['close'].iloc[-1])
    r = c1 / c0 - 1.0
    w = tgt.get(a, 0.0)
    contrib = w * r
    rows.append((a, w, r, contrib))

rows.sort(key=lambda x: -x[3])
print('\n%-10s %7s %9s %9s' % ('asset', 'wt', 'ret%', 'contrib%'))
tot_c = 0.0
for a, w, r, c in rows:
    print('%-10s %6.2f%% %8.2f%% %8.2f%%' % (a, 100*w, 100*r, 100*c))
    tot_c += c
print('sum contrib: %.2f%%  (actual block: %.2f%%)' % (100*tot_c, 100*(na/prev_na - 1)))

# current weights vs target to gauge drift
print('\ncurrent weights (drift vs 07-11 target):')
for p in sorted(acc.get('positions', []), key=lambda x: -x['market_value']):
    w = p['market_value'] / na
    t = tgt.get(p['symbol'], 0.0)
    print('  %-10s cur %6.2f%%  tgt %6.2f%%' % (p['symbol'], 100*w, 100*t))
