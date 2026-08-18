"""Trader block review 2028-11-30 -> 2028-12-14 (read-only; no step/backtest)."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

assets = get_account_dict()["watch_list"]

def series(a):
    try:
        return get_stock_daily_data(a, days=400)
    except Exception:
        return None

def iseries(a):
    try:
        return get_index_daily_data(a, days=400)
    except Exception:
        return None

closes = {}
for a in assets:
    f = series(a)
    if f is None or len(f) < 60:
        f = iseries(a)
    if f is not None and len(f):
        closes[a] = f.set_index("date")["close"].astype(float)

def px(a, d):
    c = closes.get(a)
    if c is None:
        return None
    s = c[c.index <= d]
    return float(s.iloc[-1]) if len(s) else None

for d in ["2028-11-16", "2028-11-29", "2028-11-30", "2028-12-13", "2028-12-14"]:
    row = " ".join(f"{a}={px(a,d):.2f}" for a in ["BTC","SPX","N225","ETH","WTI","NDX"])
    print(d, row)

print("\n--- Block returns 11-29 close -> 12-13 close ---")
acc = json.load(open('../persistent/account.json'))
mv = {p['symbol']: p['market_value'] for p in acc['positions']}
tot = sum(mv.values())
total_ret = 0.0
for a in assets:
    p0, p1 = px(a, "2028-11-29"), px(a, "2028-12-13")
    r = (p1/p0 - 1.0) if p0 and p1 else None
    w = mv.get(a, 0.0)/tot
    if r is not None:
        total_ret += w*r
    print(f"  {a:10s} w={w*100:6.2f}%  r={r*100:7.2f}%  contrib={w*r*100:7.3f}%")

print("\n--- Cost basis check ---")
print("cost BTC=299389.58  px11-16=", px("BTC","2028-11-16"), " px11-30=", px("BTC","2028-11-30"))
print("cost SPX=6832.85  px11-16=", px("SPX","2028-11-16"), " px11-30=", px("SPX","2028-11-30"))
print("cost N225=50494.27 px11-16=", px("N225","2028-11-16"), " px11-30=", px("N225","2028-11-30"))
print("cost ETH=2990.67 px11-16=", px("ETH","2028-11-16"), " px11-30=", px("ETH","2028-11-30"))
print("tot_assets", tot, "net_assets", acc['net_assets'], "cash", acc['available_cash'])
print("orders_pending", len(acc.get('orders', [])))
