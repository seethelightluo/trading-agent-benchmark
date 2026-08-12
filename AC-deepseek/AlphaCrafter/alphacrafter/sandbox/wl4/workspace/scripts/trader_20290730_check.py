import sys
sys.path.insert(0, ".")
import strategy
from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("watch_list:", acc.get("watch_list"))
print("total_assets:", acc.get("total_assets"), "cash:", acc.get("available_cash"))
print("positions:", [(p["symbol"], round(p["quantity"], 4)) for p in acc.get("positions", [])])
print("orders:", len(acc.get("orders", [])))
w, f, ids, info = strategy.compute_target(list(acc["watch_list"]))
print("factor_ids:", ids)
print("stale:", info.get("stale"))
print("weights sum:", round(sum(w.values()), 6))
for a in acc["watch_list"]:
    print(f"  {a:10s} w={w[a]:.4f} f={f[a]:+.5f}")
