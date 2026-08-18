"""Trader block attribution for 2028-05-23 -> 2028-06-06."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

acc = json.load(open('../persistent/account.json'))
pos = {p['symbol']: p for p in acc.get('positions', [])}
total = acc.get('total_assets', 0)
cash = acc.get('available_cash', 0)
orders = acc.get('orders', [])
print(f"total_assets={total:.2f} cash={cash:.2f} n_pos={len(pos)} n_orders={len(orders)}")
for o in orders[:5]:
    print("order:", o)

def series(sym):
    try:
        df = get_stock_daily_data(sym, days=30)
    except Exception:
        df = None
    if df is None or len(df) < 10:
        try:
            df = get_index_daily_data(sym, days=30)
        except Exception:
            return None
    if df is None or len(df) < 10:
        return None
    return df.sort_values('date')

rows = []
for sym, p in pos.items():
    df = series(sym)
    if df is None:
        rows.append((sym, float('nan'), p.get('market_value', 0) / total if total else 0, 0.0))
        continue
    c = df['close'].astype(float).values
    d = df['date'].astype(str).values
    start = None
    end = c[-1]
    for i in range(len(d)):
        if d[i] <= '2028-05-22':
            start = c[i]
    if start is None or end is None or start <= 0:
        continue
    r = end / start - 1.0
    w = p.get('market_value', 0) / total if total else 0
    rows.append((sym, r, w, r * w * 100))

rows.sort(key=lambda x: -(x[3] if x[3] == x[3] else -1e9))
print(f"{'sym':10s} {'block_ret%':>10s} {'wt%':>6s} {'contrib_pp':>10s}")
tot = 0.0
for sym, r, w, c in rows:
    if r != r:
        print(f"{sym:10s} {'NA':>10s} {w*100:6.1f} {'NA':>10s}")
        continue
    print(f"{sym:10s} {r*100:10.2f} {w*100:6.1f} {c:10.3f}")
    tot += c
print(f"sum contrib (approx): {tot:.3f} pp")
