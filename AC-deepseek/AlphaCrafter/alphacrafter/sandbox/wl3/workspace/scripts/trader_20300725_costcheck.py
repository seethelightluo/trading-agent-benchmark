from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import json

acc = get_account_dict()
pos = {p['symbol']: p for p in acc.get('positions', [])}
watch = acc.get('watch_list', [])
print("watch_list:", watch)
print()
for sym in watch:
    p = pos.get(sym)
    if p is None:
        print(sym, "NOT HELD"); continue
    df = get_stock_daily_data(symbol=sym, days=12)
    if df is None or len(df) < 2:
        df = get_index_daily_data(symbol=sym, days=12)
    if df is None or len(df) < 2:
        print(sym, "no data"); continue
    last = df.iloc[-1]
    prev = df.iloc[-2]
    cost = p['cost_price']
    print(f"{sym}: cost={cost:.4f} last_close={last['close']:.4f} ({last['date'].date()}) prev_close={prev['close']:.4f} ({prev['date'].date()}) | cost~last? {abs(cost-last['close'])/last['close']<0.001} cost~prev? {abs(cost-prev['close'])/prev['close']<0.001}")
