"""Dry-run of strategy_hook at 2030-03-12 decision - patch rebalance_to_weights to capture proposal (no state mutation)."""
import sys, json
sys.path.insert(0, ".")
import strategy

captured = {}

def fake_rebalance(weights, **kwargs):
    captured["weights"] = dict(weights)
    captured["kwargs"] = kwargs
    return {"executed": False, "skip_reason": "dry-run"}

strategy.rebalance_to_weights = fake_rebalance
strategy.strategy_hook()

w = captured.get("weights")
if not w:
    print("NO PROPOSAL CAPTURED")
else:
    tot = sum(w.values())
    print("n_assets:", len(w), "sum:", round(tot, 6))
    for a, v in sorted(w.items(), key=lambda x: -x[1]):
        print(f"  {a:10s} {v:7.4f}")
    print("forecast sample:", {k: round(v, 5) for k, v in list(captured["kwargs"].get("forecast_returns", {}).items())[:4]})
    print("factor_ids:", captured["kwargs"].get("factor_ids"))
