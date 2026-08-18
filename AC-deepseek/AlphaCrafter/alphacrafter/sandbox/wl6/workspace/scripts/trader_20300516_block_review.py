import json
from pathlib import Path

acc = json.loads(Path("../persistent/account.json").read_text())
hist = acc.get("rebalance_history", [])
print("rebalance_history entries:", len(hist))
for h in hist[-4:]:
    print(json.dumps({k: h.get(k) for k in ("date", "executed", "gross_edge_bp", "one_way_turnover", "cost_bp", "factor_ids", "gross_edge", "turnover") if k in h}, default=str)[:700])
print("---last entry full keys---", list(hist[-1].keys()) if hist else None)

print("total_assets", acc.get("total_assets"), "net_assets", acc.get("net_assets"),
      "cash", acc.get("available_cash"), "market_value", acc.get("market_value"))
print("positions:")
for p in acc.get("positions", []):
    print(" ", p["symbol"], p["direction"], "qty", round(p["quantity"], 4), "mv", round(p.get("market_value", 0), 1),
          "cost", round(p.get("cost_price", 0), 3), "px", round(p.get("current_price", 0), 3),
          "pnl", round(p.get("profit_loss", 0), 1), "pnl%", round(p.get("profit_loss_rate", 0), 3))
print("orders:", len(acc.get("orders", [])))
