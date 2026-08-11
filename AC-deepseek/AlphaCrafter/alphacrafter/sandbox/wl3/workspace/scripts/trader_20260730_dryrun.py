"""Dry-run trader strategy on 2026-07-30 without touching the account."""
import json
import numpy as np
import strategy

calls = []
strategy.rebalance_to_weights = lambda weights, **kw: calls.append((weights, kw))

try:
    strategy.strategy_hook()
except Exception as e:
    import traceback
    traceback.print_exc()
    raise SystemExit(1)

if not calls:
    print("NO REBALANCE CALL (gate/block check failed)")
    raise SystemExit(0)

weights, kw = calls[0]
print("factor_ids:", kw.get("factor_ids"))
print("horizon_days:", kw.get("horizon_days"))
print("forecast sample:", {k: round(v, 5) for k, v in list(kw["forecast_returns"].items())[:5]})
wsum = sum(weights.values())
print("n assets:", len(weights), "sum:", round(wsum, 10))
for a, w in sorted(weights.items(), key=lambda kv: -kv[1]):
    print(f"  {a:<10} {w:>8.4f}")
neg = [a for a, w in weights.items() if w < 0]
print("negative weights:", neg)
