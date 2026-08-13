from alphacrafter.sim.utils import get_account_dict
import json

acct = get_account_dict()
print("last_rebalance_date:", acct.get("last_rebalance_date"))
print("cumulative_transaction_cost:", round(acct.get("cumulative_transaction_cost", 0), 2))

print("\n-- last_target_weights --")
print(json.dumps(acct.get("last_target_weights"), indent=0)[:800])
print("\n-- last_executed_target_weights --")
print(json.dumps(acct.get("last_executed_target_weights"), indent=0)[:800])
print("\n-- last_proposed_target_weights --")
print(json.dumps(acct.get("last_proposed_target_weights"), indent=0)[:800])

print("\n-- rebalance_history (tail) --")
rh = acct.get("rebalance_history", [])
print("n entries:", len(rh))
for e in rh[-4:]:
    print(json.dumps(e)[:900])
    print("---")

print("\n-- decision_history (tail) --")
dh = acct.get("decision_history", [])
print("n entries:", len(dh))
for e in dh[-4:]:
    print(json.dumps(e)[:900])
    print("---")
