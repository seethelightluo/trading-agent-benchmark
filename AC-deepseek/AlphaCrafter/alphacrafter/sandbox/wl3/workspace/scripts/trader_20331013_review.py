"""Trader cycle review: capture account state after 2033-09-29 -> 2033-10-13 block."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
import json

acct = get_account_dict()
print("date_check net_assets=", round(acct.get("net_assets", 0), 2))
print("total_assets=", round(acct.get("total_assets", 0), 2))
print("available_cash=", round(acct.get("available_cash", 0), 2))
print("market_value=", round(acct.get("market_value", 0), 2))
print("gross_position_rate=", round(acct.get("gross_position_rate", 0), 4))
print("net_position_rate=", round(acct.get("net_position_rate", 0), 4))
print("total_pnl=", round(acct.get("total_profit_loss", 0), 2))
print("total_pnl_rate=", round(acct.get("total_profit_loss_rate", 0), 4))
print("n_positions=", len(acct.get("positions", [])))
print("n_orders=", len(acct.get("orders", [])))
print("watch_list=", acct.get("watch_list", []))

pos = acct.get("positions", [])
pos_sorted = sorted(pos, key=lambda p: p.get("market_value", 0), reverse=True)
for p in pos_sorted:
    print(f"POS {p['symbol']:10s} qty={p.get('quantity',0):>12.4f} mv={p.get('market_value',0):>12.2f} "
          f"px={p.get('current_price',0):>10.4f} cost={p.get('cost_price',0):>10.4f} "
          f"pnl={p.get('profit_loss',0):>12.2f} pnl_pct={p.get('profit_loss_rate',0)*100:>7.2f}%")

# per-asset block return using daily data (last 11 rows covers 09-29..10-13)
print("\n--- block returns (close-to-close 09-29 -> 10-13) ---")
for sym in acct.get("watch_list", []):
    df = get_stock_daily_data(symbol=sym, days=12)
    if df is None or len(df) < 2:
        print(f"{sym}: no data")
        continue
    df = df.sort_values("date")
    c0 = df.iloc[0]["close"]
    c1 = df.iloc[-1]["close"]
    print(f"{sym:10s} {df.iloc[0]['date'].date()}->{df.iloc[-1]['date'].date()} ret={(c1/c0-1)*100:+.2f}%")
