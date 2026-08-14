import json, math
from datetime import date as _date
from pathlib import Path
from alphacrafter.sim.utils import get_account_dict

BASE = Path(".")
DATE_PATH = Path("../persistent/date.json")
date_state = json.loads(DATE_PATH.read_text())
current = date_state["current_date"]
trading_days = date_state["trading_days"]
weekdays = [x for x in trading_days if _date.fromisoformat(x).weekday() < 5]
ONLINE_START = "2026-07-16"
k = weekdays.index(current) - weekdays.index(ONLINE_START)
print("current_date:", current, "| visible_through:", date_state.get("visible_through"))
print("block index k:", k, "| k % 10:", k % 10)
print("is block start:", k % 10 == 0)

acc = get_account_dict()
print("\nnet_assets:", acc.get("net_assets"), "| cash:", acc.get("available_cash"))
print("watch_list:", acc.get("watch_list"))
pos = {p["symbol"]: p for p in acc.get("positions", [])}
for s in acc.get("watch_list", []):
    p = pos.get(s)
    if p:
        print(f"  {s}: qty={p.get('quantity'):.4f} mv={p.get('market_value'):,.0f} price={p.get('current_price'):.4f} pnl={p.get('profit_loss'):,.0f}")
    else:
        print(f"  {s}: NO POSITION")
print("orders:", acc.get("orders"))
