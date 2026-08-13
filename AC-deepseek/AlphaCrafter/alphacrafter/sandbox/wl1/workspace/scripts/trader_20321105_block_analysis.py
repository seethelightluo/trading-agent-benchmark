"""Trader block analysis: 2032-10-22 -> 2032-11-05."""
import json
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

with open('../persistent/account.json') as f:
    acc = json.load(f)

# executed weights at 2032-10-22 rebalance
exec_w = None
for h in acc.get('rebalance_history', []):
    if str(h.get('date', '')).startswith('2032-10-22') and h.get('executed'):
        exec_w = h['executed_target_weights']
        break

assert exec_w is not None
assets = list(exec_w.keys())

rows = []
for a in assets:
    df = get_stock_daily_data(symbol=a, days=40)
    if df is None or len(df) < 20:
        rows.append({'asset': a, 'ret': float('nan'), 'contrib': float('nan')})
        continue
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    # last close on/before 1022 and last close (1105)
    d1022 = df[df['date'] <= pd.Timestamp('2032-10-22')]
    c0 = float(d1022['close'].iloc[-1]) if len(d1022) else float('nan')
    c1 = float(df['close'].iloc[-1])
    r = c1 / c0 - 1.0 if c0 == c0 else float('nan')
    contrib = exec_w[a] * r * 100.0  # pp contribution
    rows.append({'asset': a, 'w': exec_w[a], 'ret_pct': r * 100.0, 'contrib_pp': contrib})

out = pd.DataFrame(rows).sort_values('contrib_pp', ascending=False)
pd.set_option('display.width', 200)
print(out.to_string(index=False))
print()
print('sum contrib (pp):', round(out['contrib_pp'].sum(), 3))
print('nav start:', round(acc['rebalance_history'][-1]['pre_trade_nav'], 2))
print('cost:', round(acc['rebalance_history'][-1]['cost'], 2))
