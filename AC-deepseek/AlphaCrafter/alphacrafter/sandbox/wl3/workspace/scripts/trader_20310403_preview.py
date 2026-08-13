"""Preview-only: capture what strategy_hook would submit at 2031-04-03.

Patches rebalance_to_weights so NO orders are placed and NO account state
changes. Prints proposed weights, regime flags, and current positions.
"""
import json
import strategy as st

captured = {}


def capture(weights, forecast_returns=None, factor_ids=None, horizon_days=None):
    captured["weights"] = dict(weights)
    captured["forecast"] = dict(forecast_returns or {})
    captured["factor_ids"] = list(factor_ids or [])
    captured["horizon"] = horizon_days


st.rebalance_to_weights = capture

st.strategy_hook()

if not captured:
    print("NO_PROPOSAL (not a block start or data missing)")
else:
    w = captured["weights"]
    print("PROPOSED WEIGHTS (sum=%.6f):" % sum(w.values()))
    for a, v in sorted(w.items(), key=lambda kv: -kv[1]):
        print(f"  {a:10s} {v*100:6.2f}%")
    print("factor_ids:", captured["factor_ids"])
    # current positions
    acct = st.get_account_dict()
    print("nav=%.2f" % acct.get("net_assets", 0.0))
    for p in acct.get("positions", []):
        print(f"  pos {p['symbol']:10s} qty={p.get('quantity',0):.4f} "
              f"mv={p.get('market_value',0):.2f} price={p.get('current_price',0):.4f}")
    print("pending orders:", len(acct.get("orders", [])))
