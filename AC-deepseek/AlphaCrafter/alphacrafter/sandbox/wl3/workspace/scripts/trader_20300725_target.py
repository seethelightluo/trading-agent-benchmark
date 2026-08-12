from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
pos = {p['symbol']: p for p in acc.get('positions', [])}
total = 0.0
vals = {}
for sym in acc.get('watch_list', []):
    p = pos[sym]
    qty = p['quantity']
    df = get_stock_daily_data(symbol=sym, days=40)
    if df is None or len(df) < 2:
        df = get_index_daily_data(symbol=sym, days=40)
    px0710 = df.iloc[-1]['close'] if df is not None else None
    # find 07-10 row: cost basis date
    row0710 = df[df['date'] == '2030-07-10']
    px = float(row0710['close'].iloc[0]) if len(row0710) else px0710
    v = qty * px
    vals[sym] = v
    total += v
print(f"implied total at 07-10 close: {total:.0f}")
for sym in sorted(vals, key=lambda s: -vals[s]):
    print(f"  {sym}: {vals[sym]/total*100:.2f}%")
