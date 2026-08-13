"""Post-step analysis for the 2032-09-20 -> 2032-10-04 block (v46).
Prints account state, position weights, block returns per asset."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("total_assets:", round(acct.get("total_assets", 0), 2))
print("net_assets:", round(acct.get("net_assets", 0), 2))
print("available_cash:", round(acct.get("available_cash", 0), 2))
print("market_value:", round(acct.get("market_value", 0), 2))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("net_position_rate:", acct.get("net_position_rate"))
print("open orders:", len(acct.get("orders", [])))
print("watch_list:", acct.get("watch_list", []))

positions = acct.get("positions", [])
tot = acct.get("total_assets", 0) or 1.0
print("\npositions:")
for p in sorted(positions, key=lambda x: -x.get("market_value", 0)):
    sym = p["symbol"]
    qty = p.get("quantity", 0)
    mv = p.get("market_value", 0)
    cp = p.get("cost_price", 0)
    px = p.get("current_price", 0)
    pnl = p.get("profit_loss", 0)
    print(f"  {sym:10s} qty={qty:12.4f} cost={cp:10.4f} px={px:10.4f} mv={mv:12.2f} "
          f"w={mv/tot:.4f} pnl={pnl:10.2f} ({p.get('profit_loss_rate',0)*100:+.2f}%)")

# block returns: fetch 15 daily bars ending at last close
print("\nblock asset returns (09-20 -> 10-04):")
for sym in acct.get("watch_list", []):
    try:
        df = get_stock_daily_data(symbol=sym, days=15)
    except Exception:
        df = None
    if df is None or len(df) < 2:
        try:
            df = get_index_daily_data(symbol=sym, days=15)
        except Exception:
            df = None
    if df is None or len(df) < 2:
        print(f"  {sym:10s} no data")
        continue
    df = df.sort_values("date")
    first, last = float(df.iloc[0]["close"]), float(df.iloc[-1]["close"])
    ret = (last / first - 1.0) * 100.0
    print(f"  {sym:10s} {first:10.4f} -> {last:10.4f}  {ret:+7.2f}%")
