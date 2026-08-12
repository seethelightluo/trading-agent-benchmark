from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd

acc = get_account_dict()
watch = acc.get('watch_list', [])
pos = {p['symbol']: p for p in acc.get('positions', [])}
total = acc.get('total_assets', 0)

rows = []
for s in watch:
    df = get_stock_daily_data(symbol=s, days=40)
    if df is None or len(df) == 0:
        df = get_index_daily_data(symbol=s, days=40)
    if df is None or len(df) == 0:
        rows.append((s, None, None, None, None))
        continue
    df = df.sort_values('date').reset_index(drop=True)
    dates = df['date'].astype(str)
    # find block start 2029-03-08 (visible through prev day) and end 2029-03-22
    d_start = None; d_end = None
    for d in dates:
        if d <= '2029-03-08':
            d_start = d
    d_end = dates.iloc[-1]  # 03-22
    c_start = df.loc[dates == d_start, 'close'].iloc[0] if d_start else None
    c_end = df.loc[dates == d_end, 'close'].iloc[0]
    ret = c_end / c_start - 1 if c_start else None
    p = pos.get(s)
    mv = p.get('market_value', 0) if p else 0
    w_now = mv / total if total else 0
    w_start = w_now / (1 + ret) if ret is not None else None
    rows.append((s, d_start, round(ret*100,2) if ret is not None else None, round(w_now*100,2), round(w_start*100,2) if w_start is not None else None))

print(f"{'sym':9} {'d_start':11} {'ret%':>7} {'w_now%':>7} {'w_start%':>7}")
for r in rows:
    print(f"{r[0]:9} {str(r[1]):11} {str(r[2]):>7} {str(r[3]):>7} {str(r[4]):>7}")
print('sum w_now', round(sum(r[3] for r in rows if r[3] is not None),3))
