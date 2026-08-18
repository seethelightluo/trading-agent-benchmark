"""Trader cycle review: account state + block drivers for 2029-09-06 -> 2029-09-20."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
print("total_assets:", round(acc.get("total_assets", 0), 2))
print("net_assets:", round(acc.get("net_assets", 0), 2))
print("available_cash:", round(acc.get("available_cash", 0), 2))
print("market_value:", round(acc.get("market_value", 0), 2))
print("gross_position_rate:", round(acc.get("gross_position_rate", 0), 4))
print("watch_list:", acc.get("watch_list", []))

positions = {p["symbol"]: p for p in acc.get("positions", [])}
total_mv = sum(p.get("market_value", 0) for p in positions.values()) or 1.0
print("\n--- POSITIONS ---")
for sym in acc.get("watch_list", []):
    p = positions.get(sym)
    if p is None:
        print(f"{sym}: NO POSITION")
        continue
    qty = p.get("quantity", 0)
    mv = p.get("market_value", 0)
    pl = p.get("profit_loss", 0)
    plr = p.get("profit_loss_rate", 0)
    cost = p.get("cost_price", 0)
    print(f"{sym}: qty={qty:.4f} mv={mv:,.0f} ({mv/total_mv*100:.2f}%) pl={pl:,.0f} plr={plr*100:.2f}% cost={cost}")

print("\n--- BLOCK RETURNS (close 09-05 -> close 09-19, last 2 obs) ---")
def get_closes(sym):
    df = get_stock_daily_data(sym, days=5)
    if df is None or len(df) < 2:
        df = get_index_daily_data(sym, days=5)
    if df is None or len(df) < 2:
        return None
    return df[["date", "close"]].values

for sym in acc.get("watch_list", []):
    arr = get_closes(sym)
    if arr is None:
        print(f"{sym}: no data")
        continue
    p0 = float(arr[-2][1]); p1 = float(arr[-1][1])
    if p0 <= 0:
        print(f"{sym}: bad price {p0}")
        continue
    print(f"{sym}: {arr[-2][0]} close={p0:.4f} -> {arr[-1][0]} close={p1:.4f}  ret={(p1/p0-1)*100:+.2f}%")

print("\norders:", len(acc.get("orders", [])))
