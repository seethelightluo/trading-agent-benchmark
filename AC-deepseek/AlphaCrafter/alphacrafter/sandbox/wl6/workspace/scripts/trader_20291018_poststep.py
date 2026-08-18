"""Post-step diagnostic for 2029-10-04 -> 2029-10-18 block."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
print("=== ACCOUNT ===")
print("total_assets", acc.get("total_assets"))
print("net_assets", acc.get("net_assets"))
print("available_cash", acc.get("available_cash"))
print("market_value", acc.get("market_value"))
print("total_profit_loss", acc.get("total_profit_loss"))
print("gross_position_rate", acc.get("gross_position_rate"))
print("watch_list", acc.get("watch_list"))
print("pending orders:", len(acc.get("orders", [])))
for o in acc.get("orders", [])[:10]:
    print("  order:", o)

print("\n=== POSITIONS ===")
pos = {p["symbol"]: p for p in acc.get("positions", [])}
tot = 0.0
for sym, p in sorted(pos.items()):
    tot += p.get("market_value", 0.0)
    print(f"{sym:10s} qty={p.get('quantity',0):12.4f} mv={p.get('market_value',0):12.2f} "
          f"pnl={p.get('profit_loss',0):10.2f} ({p.get('profit_loss_rate',0)*100:6.2f}%) "
          f"cost={p.get('cost_price',0):10.4f} cur={p.get('current_price',0):10.4f}")
print("sum mv", tot)

print("\n=== 20d returns per asset (block attribution) ===")
for sym in acc.get("watch_list", []):
    df = get_stock_daily_data(sym, days=30)
    if df is None or len(df) < 2:
        print(f"{sym:10s} no data")
        continue
    r20 = df["close"].iloc[-1] / df["close"].iloc[-11] - 1.0 if len(df) > 10 else float("nan")
    r10 = df["close"].iloc[-1] / df["close"].iloc[-6] - 1.0 if len(df) > 5 else float("nan")
    w = pos.get(sym, {}).get("market_value", 0.0) / max(acc.get("net_assets", 1.0), 1e-9)
    pnl = pos.get(sym, {}).get("profit_loss", 0.0)
    print(f"{sym:10s} r10={r10*100:7.2f}% r20={r20*100:7.2f}% w_now={w*100:6.2f}% pnl={pnl:10.2f}")
