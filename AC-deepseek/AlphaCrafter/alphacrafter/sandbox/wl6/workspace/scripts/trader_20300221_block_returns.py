import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = get_account_dict()["watch_list"]


def get_data(a, days):
    f = get_stock_daily_data(a, days=days)
    if f is None:
        f = get_index_daily_data(a, days=days)
    return f


rows = {}
for a in assets:
    f = get_data(a, 15)
    if f is None or len(f) < 12:
        continue
    f = f.sort_values("date")
    prev = f.iloc[-11]["close"]   # close 10 trading days ago (block start reference)
    last = f.iloc[-1]["close"]
    rows[a] = (prev, last, last / prev - 1.0)

# regime at block start: cross-asset 20d drift using data up to 2030-02-20
panel = {}
for a in assets:
    f = get_data(a, 40)
    if f is not None:
        panel[a] = f.set_index("date")["close"]
p = pd.DataFrame(panel).sort_index()
rets = p.pct_change().dropna()
mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean())
v20 = float(mkt.tail(20).std())
trend = r20 / v20 * (20 ** 0.5) if v20 and v20 > 1e-12 else 0.0
regime = "bull" if trend > 1.0 else ("bear" if trend < -1.0 else "sideways")

vix = get_index_daily_data("VIX", days=25)
vix_last = float(vix.sort_values("date").iloc[-1]["close"]) if vix is not None else None

print(f"block_start_regime={regime} trend={trend:.3f} vix_last={vix_last:.1f}")
print("asset block returns (prev_close -> last_close):")
for a in sorted(rows, key=lambda x: -rows[x][2]):
    prev, last, r = rows[a]
    print(f"  {a}: {r*100:+.2f}%  ({prev:.2f} -> {last:.2f})")
