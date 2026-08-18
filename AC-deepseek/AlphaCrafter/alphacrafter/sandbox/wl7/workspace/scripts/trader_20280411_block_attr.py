"""Trader block attribution for 2028-03-28 -> 2028-04-11 (approx)."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

acc = json.load(open('../persistent/account.json'))
pos = {p['symbol']: p for p in acc.get('positions', [])}
total = acc.get('total_assets', 0)

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
        continue
    c = df['close'].astype(float).values
    d = df['date'].astype(str).values
    start = None
    end = c[-1]
    for i in range(len(d)):
        if d[i] <= '2028-03-27':
            start = c[i]
    if start is None or end is None or start <= 0:
        continue
    r = end / start - 1.0
    w = p.get('market_value', 0) / total if total else 0
    rows.append((sym, r, w, r * w * 100))

rows.sort(key=lambda x: -x[3])
print(f"{'sym':10s} {'block_ret%':>10s} {'wt%':>6s} {'contrib_pp':>10s}")
tot = 0.0
for sym, r, w, c in rows:
    print(f"{sym:10s} {r*100:10.2f} {w*100:6.1f} {c:10.3f}")
    tot += c
print(f"sum contrib (approx): {tot:.3f} pp")
