"""Reconstruct cycle91 (08-21 -> 09-04) block: per-asset returns over the block,
current weights, and attribution. Uses only data visible through 2031-09-03."""
import json, sys
sys.path.insert(0, '.')
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

acc = get_account_dict()
print('NAV total_assets:', round(acc['total_assets'], 2), 'cash:', round(acc['available_cash'], 4))
print('n_positions:', len(acc['positions']), 'n_orders:', len(acc['orders']))

nav_start = 1044237.0   # cycle90 end NAV (block start 08-21) from memory
nav_end = acc['total_assets']
print(f'Block NAV: {nav_start:.0f} -> {nav_end:.2f}  return {(nav_end/nav_start-1)*100:.3f}%')

symbols = [p['symbol'] for p in acc['positions']]
print('\nPer-asset: current mv, weight%, return over block (close[09-03]/close[08-20]-1)')
rows = []
for p in acc['positions']:
    sym = p['symbol']
    df = get_stock_daily_data(symbol=sym, days=40)
    if df is None or len(df) < 20:
        print(sym, 'NO DATA')
        continue
    df = df.sort_values('date').reset_index(drop=True)
    dates = [str(d.date()) for d in df['date']]
    def find(d):
        for i, dt in enumerate(dates):
            if dt >= d:
                return i
        return None
    i0 = find('2031-08-20')
    i1 = find('2031-09-03')
    c0 = df.iloc[i0]['close']; c1 = df.iloc[i1]['close']
    r = c1 / c0 - 1 if c0 else 0
    mv = p['market_value']
    w = mv / nav_end * 100
    rows.append((sym, w, r, mv))
    print(f"{sym:10s} mv={mv:10.1f} w={w:6.2f}%  block_ret={r*100:7.2f}%  ({dates[i0]}->{dates[i1]})")

tot_w = sum(r[1] for r in rows)
print(f'\nSum weights: {tot_w:.3f}%')
attrib = sum(r[1]/100 * r[2] for r in rows)
print(f'Approx attribution (current-wt x block-ret): {attrib*100:.2f}%')
