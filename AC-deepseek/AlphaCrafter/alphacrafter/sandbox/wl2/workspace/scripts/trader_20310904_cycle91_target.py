"""Reconstruct cycle91 executed target at block start 08-21 (visible 08-20)
and estimate one-way turnover vs cycle90 target. Uses end-of-block holdings
and per-asset block returns to back out start-of-block notional."""
import sys
sys.path.insert(0, '.')
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

acc = get_account_dict()
nav_end = acc['total_assets']
nav_start = 1044237.0

# cycle90 target weights (from memory entry 20310821), used as the pre-rebalance reference
cyc90_target = {
    'SOX': 0.123, '000300.SH': 0.087, 'WTI': 0.054, 'ETH': 0.032, 'NDX': 0.075,
    'XAU': 0.027, 'SPX': 0.073, 'N225': 0.065, '000688.SH': 0.095,
    'HSI': 0.0698, 'SX5E': 0.0698, 'BTC': 0.0698, 'US10Y': 0.0698, 'CN10Y': 0.0698,
    'COPPER': 0.016,
}
print(f"{'sym':10s} {'mv_end':>10s} {'ret':>7s} {'notional_start':>13s} {'tgt_w':>7s} {'cyc90_w':>8s} {'delta_w':>7s}")
tgt = {}
for p in acc['positions']:
    sym = p['symbol']
    df = get_stock_daily_data(symbol=sym, days=40)
    df = df.sort_values('date').reset_index(drop=True)
    dates = [str(d.date()) for d in df['date']]
    i0 = next(i for i, dt in enumerate(dates) if dt >= '2031-08-20')
    i1 = next(i for i, dt in enumerate(dates) if dt >= '2031-09-03')
    c0 = df.iloc[i0]['close']; c1 = df.iloc[i1]['close']
    r = c1 / c0 - 1
    mv = p['market_value']
    ns = mv / (1 + r)
    w = ns / nav_start
    tgt[sym] = w
    cw = cyc90_target.get(sym, 0.0)
    print(f"{sym:10s} {mv:10.1f} {r*100:6.2f}% {ns:13.1f} {w:7.4f} {cw:8.4f} {w-cw:7.4f}")

tot = sum(tgt.values())
print(f'\nSum reconstructed target: {tot:.4f}')
oneway = sum(abs(tgt.get(s,0) - cyc90_target.get(s,0)) for s in set(tgt)|set(cyc90_target)) / 2
print(f'One-way turnover (target vs cyc90 target): {oneway*100:.2f}%')
