"""Block PnL attribution 2027-07-20 -> 2027-08-03 (10 trading days)."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

acct = get_account_dict()
assets = acct.get('watch_list', [])
pos = {p['symbol']: p for p in acct.get('positions', [])}

def series(sym):
    try:
        df = get_stock_daily_data(sym, days=30)
    except Exception:
        df = None
    if df is None or len(df) < 5:
        try:
            df = get_index_daily_data(sym, days=30)
        except Exception:
            df = None
    if df is None or len(df) < 5:
        return None
    s = pd.Series(df['close'].astype(float), index=pd.to_datetime(df['date']))
    return s

print(f"{'asset':10s} {'qty':>12s} {'mv':>12s} {'blk_ret%':>9s} {'blk_pnl':>12s}")
tot = 0.0
for a in assets:
    s = series(a)
    if s is None or len(s) < 11:
        print(f"{a:10s} INSUFFICIENT DATA")
        continue
    r = float(s.iloc[-1]/s.iloc[-11]-1.0)
    p = pos.get(a)
    pnl = p['market_value'] * r / (1+r) if p else 0.0
    tot += pnl
    print(f"{a:10s} {p['quantity'] if p else 0:12.4f} {p['market_value'] if p else 0:12.2f} {r*100:9.2f} {pnl:12.2f}")
print(f"{'TOTAL':10s} {'':12s} {'':12s} {'':9s} {tot:12.2f}")
print("net_assets:", acct.get('net_assets'))
