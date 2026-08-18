import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("total_assets", round(acct['total_assets'],2))
print("cash", acct['available_cash'])
print("gross", acct['gross_position_rate'])
print("n_orders", len(acct.get('orders', [])))

pos = {p['symbol']: p for p in acct['positions']}
watch = acct['watch_list']
total = acct['total_assets']

def get(sym):
    try:
        df = get_stock_daily_data(sym, days=30)
        if df is None or len(df) < 15:
            df = get_index_daily_data(sym, days=30)
        return df
    except Exception:
        try:
            return get_index_daily_data(sym, days=30)
        except Exception:
            return None

rows = []
for s in watch:
    df = get(s)
    if df is None or len(df) < 15:
        rows.append((s, None, None, None, None))
        continue
    df = df.sort_values('date')
    last = df.iloc[-1]
    prev = df.iloc[-11] if len(df) >= 11 else df.iloc[0]
    block_ret = last['close']/prev['close'] - 1.0 if prev['close'] else None
    w = pos.get(s, {}).get('market_value', 0)/total if total else 0
    rows.append((s, round(block_ret*100,2) if block_ret is not None else None, round(w*100,2), round((block_ret or 0)*w*100,2), pos.get(s,{}).get('quantity',0)))

rows.sort(key=lambda r: -(r[3] if r[3] is not None else 0))
print(f"{'sym':<10}{'block_ret%':>10}{'wt%':>8}{'attr_pp':>9}{'qty':>14}")
for s, br, w, attr, q in rows:
    print(f"{s:<10}{str(br):>10}{str(w):>8}{str(attr):>9}{str(q):>14}")

for m in ['VIX','DXY','USDCNY','USDJPY','EURUSD']:
    try:
        df = get_index_daily_data(m, days=30)
        if df is not None and len(df):
            df = df.sort_values('date')
            c20 = round((df.iloc[-1]['close']/df.iloc[-21]['close']-1)*100,2) if len(df)>=21 else None
            print(m, "last", round(df.iloc[-1]['close'],2), "chg20d%", c20)
    except Exception as e:
        print(m, "err", e)
