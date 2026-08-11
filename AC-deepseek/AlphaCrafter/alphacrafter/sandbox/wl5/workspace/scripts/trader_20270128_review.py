from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets:", round(acc.get("total_assets", 0), 2))
print("net_assets:", round(acc.get("net_assets", 0), 2))
print("available_cash:", round(acc.get("available_cash", 0), 2))
print("market_value:", round(acc.get("market_value", 0), 2))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("net_position_rate:", acc.get("net_position_rate"))
print("orders:", len(acc.get("orders", [])))
pos = acc.get("positions", [])
pos_sorted = sorted(pos, key=lambda p: p.get("profit_loss_rate", 0), reverse=True)
for p in pos_sorted:
    print(f"{p['symbol']:>8s} qty={p['quantity']:>12.4f} mktval={p['market_value']:>12.2f} pl={p['profit_loss']:>12.2f} plr={p['profit_loss_rate']*100:>7.2f}%")
w = {}
for p in pos:
    w[p['symbol']] = p['market_value'] / acc.get("net_assets", 1)
print("\nWeights (sum=%.4f):" % sum(w.values()))
for s, v in sorted(w.items(), key=lambda kv: -kv[1]):
    print(f"  {s:>8s} {v*100:6.2f}%")
