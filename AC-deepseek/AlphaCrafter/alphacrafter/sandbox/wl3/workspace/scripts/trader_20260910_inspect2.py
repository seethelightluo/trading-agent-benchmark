# Trader: raw data horizon check + rebalance helper internals + account history
import json, os, sys, inspect
sys.path.insert(0, ".")
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, rebalance_to_weights

print("=== 1. raw dataset files ===")
for root in ["../persistent"]:
    if os.path.isdir(root):
        for name in sorted(os.listdir(root))[:30]:
            print(" ", root, name)

print("\n=== 2. HSI / US10Y via stock vs index api (last rows) ===")
for sym in ["HSI", "US10Y", "BTC"]:
    for fn in [get_stock_daily_data, get_index_daily_data]:
        try:
            df = fn(sym, days=300)
            if df is not None and len(df):
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else last
                print(f"  {sym:7s} {fn.__name__:22s} rows={len(df):4d} last={str(last['date'])[:10]} close={last['close']:.4f} prev_close={prev['close']:.4f}")
            else:
                print(f"  {sym:7s} {fn.__name__:22s} None")
        except Exception as e:
            print(f"  {sym:7s} {fn.__name__:22s} ERR {e}")

print("\n=== 3. account.json keys and history ===")
acc = json.load(open("../persistent/account.json"))
print("keys:", sorted(acc.keys()))
for k in ["initial_capital", "total_assets", "net_assets", "last_rebalance_date",
          "cumulative_transaction_cost", "portfolio_initialized"]:
    print(f"  {k}: {acc.get(k)}")
print("  last_executed_target_weights:", {k: round(v, 4) for k, v in (acc.get("last_executed_target_weights") or {}).items()})
print("  last_proposed_target_weights:", {k: round(v, 4) for k, v in (acc.get("last_proposed_target_weights") or {}).items()})

dec = acc.get("decision_history", [])
print(f"\n  decision_history len={len(dec)}")
for r in dec[-6:]:
    print("   ", r.get("date"), "executed=", r.get("executed"), "skip=", r.get("skip_reason"),
          "turnover=", round(r.get("one_way_turnover") or 0, 4),
          "edge_bps=", round(r.get("gross_edge_bps") or 0, 3),
          "thr_bps=", round(r.get("decision_edge_threshold_bps") or 0, 3))

rh = acc.get("rebalance_history", [])
print(f"\n  rebalance_history len={len(rh)}")
for r in rh[-6:]:
    print("   ", r.get("date"), "cost=", round(r.get("cost") or 0, 2),
          "turnover=", round(r.get("one_way_turnover") or 0, 4),
          "edge_bps=", round(r.get("gross_edge_bps") or 0, 3))

print("\n=== 4. rebalance_to_weights middle (decision/gate logic) ===")
src = inspect.getsource(rebalance_to_weights)
lines = src.split("\n")
for i, ln in enumerate(lines):
    if any(k in ln for k in ["def decision", "gross_edge", "edge_threshold", "skip", "initial_allocation", "cost_bps", "applied_cost"]):
        print(f"{i:3d}: {ln}")
