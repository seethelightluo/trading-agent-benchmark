from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
pos = {p['symbol']: p for p in acc.get('positions', [])}
for sym in ['SPX', 'XAU', 'WTI', 'COPPER', 'N225', '000688.SH']:
    p = pos[sym]
    cost = p['cost_price']
    df = get_stock_daily_data(symbol=sym, days=40)
    if df is None or len(df) < 2:
        df = get_index_daily_data(symbol=sym, days=40)
    best = None
    for i, row in df.iterrows():
        diff = abs(row['close'] - cost) / cost
        if best is None or diff < best[1]:
            best = (row['date'].date(), diff)
    print(f"{sym}: cost={cost:.4f} closest_close={best[0]} rel_diff={best[1]*100:.3f}%")
