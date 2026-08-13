
import json
acct = json.load(open('../persistent/account.json'))
hist = acct.get('decision_history', [])
print("num decisions:", len(hist))
for d in hist[-2:]:
    print("date:", d.get("date"), "executed:", d.get("executed"), "skip_reason:", d.get("skip_reason"))
    print("one_way_turnover:", d.get("one_way_turnover"), "gross_edge_bps:", d.get("gross_edge_bps"), "threshold_bps:", d.get("decision_edge_threshold_bps"))
    print("proposed:", {k: round(v,4) for k,v in d.get("proposed_target_weights",{}).items()})
    print("executed:", {k: round(v,4) for k,v in d.get("executed_target_weights",{}).items()})
    print("---")
print("last_proposed:", {k: round(v,4) for k,v in acct.get("last_proposed_target_weights",{}).items()})
print("last_executed:", {k: round(v,4) for k,v in acct.get("last_executed_target_weights",{}).items()})
