"""Trader inspection (read-only): account state + ensemble match check."""
import json
from pathlib import Path
from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("net_assets:", acc.get("net_assets"))
print("total_assets:", acc.get("total_assets"))
print("available_cash:", acc.get("available_cash"))
print("market_value:", acc.get("market_value"))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("watch_list:", acc.get("watch_list"))
print("positions:")
for p in acc.get("positions", []):
    print("  ", p["symbol"], p.get("direction"), "qty", round(p.get("quantity", 0), 4),
          "mv", round(p.get("market_value", 0), 2),
          "w", round(p.get("market_value", 0) / max(acc.get("net_assets", 1), 1e-9), 4),
          "pnl", round(p.get("profit_loss", 0), 2))
print("orders:", acc.get("orders", [])[:5])

ens_path = Path("factor_ensemble.json")
ens = json.loads(ens_path.read_text())
print("ROOT ensemble factors:", [(f["factor_id"], f.get("weight"), f.get("direction")) for f in ens.get("selected_factors", [])])
ens2_path = Path("factors/factor_ensemble.json")
if ens2_path.exists():
    ens2 = json.loads(ens2_path.read_text())
    print("FACTORS/ ensemble factors:", [(f["factor_id"], f.get("weight"), f.get("direction")) for f in ens2.get("selected_factors", [])])
