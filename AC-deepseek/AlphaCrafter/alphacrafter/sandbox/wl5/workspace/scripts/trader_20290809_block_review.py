"""Trader block review: account state + per-symbol block returns for 2029-07-26 -> 2029-08-09."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("TOTAL_ASSETS", round(acct.get("total_assets", 0), 2))
print("NET_ASSETS", round(acct.get("net_assets", 0), 2))
print("CASH", round(acct.get("available_cash", 0), 2))
print("GROSS_POS", round(acct.get("gross_position_rate", 0), 4))
print("N_POSITIONS", len(acct.get("positions", [])))
print("N_ORDERS", len(acct.get("orders", [])))

positions = {p["symbol"]: p for p in acct.get("positions", [])}
watch = acct.get("watch_list", [])

# Block returns using the last ~12 trading days of data
import pandas as pd
rows = []
for sym in watch:
    df = get_stock_daily_data(symbol=sym, days=20)
    if df is None or len(df) < 12:
        df = get_index_daily_data(symbol=sym, days=20)
    if df is None or len(df) < 12:
        rows.append((sym, None, None, None, None))
        continue
    df = df.sort_values("date").reset_index(drop=True)
    block_ret = df.iloc[-1]["close"] / df.iloc[-11]["close"] - 1.0
    pos = positions.get(sym)
    qty = pos.get("quantity", 0) if pos else 0
    plr = pos.get("profit_loss_rate", 0) if pos else None
    w = (pos.get("market_value", 0) / acct.get("total_assets", 1)) if pos else 0.0
    rows.append((sym, block_ret, w, qty, plr))

print("\n=== BLOCK DRIVERS (2029-07-26 -> 2029-08-09, close-to-close) ===")
for sym, r, w, qty, plr in sorted(rows, key=lambda x: -(x[1] if x[1] is not None else -9)):
    if r is None:
        print(f"{sym:12s} no-data")
    else:
        qty_s = f"{qty:,.4f}" if qty is not None else "-"
        plr_s = f"{plr*100:.1f}%" if plr is not None else "-"
        print(f"{sym:12s} ret {r*100:8.2f}%  w {w*100:6.2f}%  qty {qty_s}  plr {plr_s}")

# Contribution approx: w_prev * ret. Use current weight as proxy for report.
print("\n=== POSITION DETAIL ===")
for p in sorted(acct.get("positions", []), key=lambda x: -x.get("market_value", 0)):
    print(f"{p['symbol']:12s} qty {p.get('quantity',0):>14.4f} mv {p.get('market_value',0):>12.2f} "
          f"pl {p.get('profit_loss',0):>10.2f} plr {p.get('profit_loss_rate',0)*100:7.2f}% "
          f"cost {p.get('cost_price',0):>12.4f} px {p.get('current_price',0):>12.4f}")
