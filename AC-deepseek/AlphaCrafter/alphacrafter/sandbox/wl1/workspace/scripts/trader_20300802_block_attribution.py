"""Trader attribution for block 2030-07-19 -> 2030-08-02."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

acc = get_account_dict()
assets = acc.get("watch_list", [])
start = "2030-07-18"  # decision data cutoff (last completed day before rebalance)
end = "2030-08-02"

print("=== Per-asset returns and block attribution ===")
rows = []
for a in assets:
    df = get_stock_daily_data(symbol=a, days=40)
    if df is None or len(df) < 20:
        continue
    df = df.sort_values("date")
    d0 = df[df["date"] <= start]
    if d0.empty:
        continue
    p0 = float(d0.iloc[-1]["close"])
    p1 = float(df.iloc[-1]["close"])
    r = p1 / p0 - 1.0
    rows.append((a, r, p0, p1))

# block-end weights from account
tot = acc.get("total_assets", 1)
w_end = {p["symbol"]: p.get("market_value", 0) / tot for p in acc.get("positions", [])}
for a, r, p0, p1 in sorted(rows, key=lambda x: -x[1]):
    w = w_end.get(a, 0)
    print(f"{a:10s} ret {r*100:7.2f}%  end_w {w*100:5.2f}%  contrib~ {r*w*100:6.2f}%")
print()
print("Total assets:", round(acc.get("total_assets", 0), 2))
print("Last rebalance date:", acc.get("last_rebalance_date"))
