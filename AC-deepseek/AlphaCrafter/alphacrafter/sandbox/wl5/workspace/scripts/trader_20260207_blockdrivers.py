import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

a = get_account_dict()
qty = {p['symbol']: p['quantity'] for p in a['positions']}
total = a['net_assets']


def closes(sym):
    df = None
    try:
        df = get_stock_daily_data(sym, days=30)
    except Exception:
        df = None
    if df is None or 'close' not in df or len(df) < 5:
        try:
            df = get_index_daily_data(sym, days=30)
        except Exception:
            df = None
    if df is None or 'close' not in df or len(df) < 5:
        return None
    s = df[['date', 'close']].copy()
    s['date'] = pd.to_datetime(s['date'])
    return s.set_index('date')['close'].astype(float)


rows = []
for sym, q in qty.items():
    c = closes(sym)
    if c is None or len(c) < 5:
        rows.append((sym, float('nan'), float('nan'), q, 0.0))
        continue
    p0 = float(c.iloc[-11])  # close ~02-06 (block start, decision day)
    p1 = float(c.iloc[-1])   # close ~02-21 (block end)
    if p0 <= 0:
        continue
    ret = p1 / p0 - 1.0
    mv0 = q * p0
    contrib = (mv0 / total) * ret
    rows.append((sym, ret, contrib, q, p1))

rows.sort(key=lambda r: -r[1])
print(f"{'sym':10s} {'block_ret%':>10s} {'weight0%':>8s} {'contrib_pp':>10s} {'end_px':>10s}")
tot = 0.0
for sym, ret, contrib, q, p1 in rows:
    w0 = q * (p1 / (1 + ret)) / total
    print(f"{sym:10s} {ret*100:10.2f} {w0*100:8.2f} {contrib*100:10.2f} {p1:10.2f}")
    tot += contrib
print(f"sum contrib: {tot*100:.2f} pp | net return target: +0.67%")
