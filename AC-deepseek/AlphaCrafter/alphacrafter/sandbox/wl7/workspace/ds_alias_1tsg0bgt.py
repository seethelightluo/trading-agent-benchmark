import json
d=json.load(open("../persistent/account.json"))
print("net_assets",d["net_assets"],"gross",d["gross_position_rate"],"cash",d["available_cash"])
print("last_rebalance",d["last_rebalance_date"],"cum_cost",round(d["cumulative_transaction_cost"],2))
print("orders",d["orders"])
print("last_exec_target:",{k:round(v,3) for k,v in d["last_executed_target_weights"].items()})
print("last_proposed:",{k:round(v,3) for k,v in d["last_proposed_target_weights"].items()})