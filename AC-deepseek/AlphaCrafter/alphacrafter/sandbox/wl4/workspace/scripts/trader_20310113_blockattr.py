"""Block attribution for 12-16..12-30 and 12-30..01-13 blocks (trader review 2031-01-13)."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = get_account_dict()["watch_list"]

def get(a, n=40):
    try:
        return get_stock_daily_data(a, days=n)
    except Exception:
        try:
            return get_index_daily_data(a, days=n)
        except Exception:
            return None

# executed weights at 12-16 and 12-30 from account rebalance history
acc = json.load(open('../persistent/account.json'))
rh = acc['rebalance_history']
w16 = [r for r in rh if r['date'] == '2030-12-16'][0]['executed_target_weights']
w30 = [r for r in rh if r['date'] == '2030-12-30'][0]['executed_target_weights']

data = {}
for a in assets:
    df = get(a)
    if df is None or len(df) < 35:
        print(a, "NO DATA"); continue
    df = df.sort_values("date")
    closes = [float(x) for x in df["close"]]
    data[a] = closes

# Need closes: for 12-16 close, 12-30 close, 01-13 close
# Locate by index from the end: last close = 01-13 (current). 10 trading days before = 12-30, 20 before = 12-16
print(f"{'asset':10s} {'w16':>6s} {'r16_30%':>8s} {'c16':>8s} {'w30':>6s} {'r30_13%':>8s} {'c30':>8s} {'r13(10d)%':>9s}")
tot16, tot30 = 0.0, 0.0
for a in assets:
    if a not in data:
        continue
    cl = data[a]
    c13 = cl[-1]
    c30 = cl[-11] if len(cl) >= 11 else None   # close 12-30
    c16 = cl[-21] if len(cl) >= 21 else None   # close 12-16
    r16_30 = (c30 / c16 - 1) * 100 if c16 and c30 else float('nan')
    r30_13 = (c13 / c30 - 1) * 100 if c30 else float('nan')
    c16c = w16.get(a, 0) * r16_30
    c30c = w30.get(a, 0) * r30_13
    tot16 += c16c; tot30 += c30c
    print(f"{a:10s} {w16.get(a,0):6.3f} {r16_30:8.2f} {c16c:8.2f} {w30.get(a,0):6.3f} {r30_13:8.2f} {c30c:8.2f} {r30_13:9.2f}")
print(f"{'TOTAL':10s} {'':>6s} {'':>8s} {tot16:8.2f} {'':>6s} {'':>8s} {tot30:8.2f}")
