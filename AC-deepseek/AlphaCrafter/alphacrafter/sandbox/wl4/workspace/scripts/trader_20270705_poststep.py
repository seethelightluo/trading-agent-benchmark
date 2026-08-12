"""Post-step probe 2027-07-05: account state + block asset returns for summary."""
import json
from pathlib import Path
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
print("net_assets:", round(acc.get("net_assets", 0), 2))
print("cash:", acc.get("available_cash"), "gross_pos_rate:", acc.get("gross_position_rate"))
print("positions:")
tot = 0
for p in acc.get("positions", []):
    mv = p.get("market_value", 0)
    tot += mv
    print(f"  {p['symbol']:10s} qty={p.get('quantity',0):12.4f} mv={mv:12.2f} pnl={p.get('profit_loss',0):10.2f} ({p.get('profit_loss_rate',0)*100:+.2f}%)")
print("sum mv:", round(tot, 2))
print("orders:", acc.get("orders", []))

# block asset returns: 10 trading days back from last close
assets = acc.get("watch_list", [])
print("\n--- block returns (10d) ---")
for a in assets:
    df = get_stock_daily_data(a, days=12)
    if df is None or len(df) < 11:
        print(f"  {a}: no data")
        continue
    df = df.sort_values("date")
    c = df["close"].astype(float)
    r = float(c.iloc[-1] / c.iloc[-11] - 1)
    print(f"  {a:10s} {r*100:+.2f}%")
