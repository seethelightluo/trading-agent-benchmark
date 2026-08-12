"""Trader block review: 2029-01-15..2029-01-29 cycle.
Prints account state and per-asset 10d returns to attribute block PnL.
"""
import json
from alphacrafter.sim.utils import (
    get_account_dict,
    get_stock_daily_data,
    get_index_daily_data,
)

acc = get_account_dict()
print("== ACCOUNT ==")
print("total_assets:", round(acc.get("total_assets", 0), 2))
print("net_assets:", round(acc.get("net_assets", 0), 2))
print("available_cash:", round(acc.get("available_cash", 0), 2))
print("market_value:", round(acc.get("market_value", 0), 2))
print("gross_position_rate:", round(acc.get("gross_position_rate", 0), 4))
print("net_position_rate:", round(acc.get("net_position_rate", 0), 4))
print("total_profit_loss:", round(acc.get("total_profit_loss", 0), 2))
print("orders:", len(acc.get("orders", [])))

pos = {p["symbol"]: p for p in acc.get("positions", [])}
print("\n== POSITIONS ==")
tot_mv = 0.0
for s, p in sorted(pos.items()):
    mv = p.get("market_value", 0)
    tot_mv += mv
    print(f"{s:10s} qty={p.get('quantity',0):12.4f} mv={mv:12.2f} "
          f"cost={p.get('cost_price',0):12.4f} cur={p.get('current_price',0):12.4f} "
          f"pnl={p.get('profit_loss',0):10.2f} pnl%={p.get('profit_loss_rate',0)*100:7.2f}")
print("sum mv:", round(tot_mv, 2))

print("\n== 10d BLOCK RETURNS (2029-01-15 close .. 2029-01-29 close) ==")
for s in sorted(pos.keys()):
    try:
        df = get_stock_daily_data(symbol=s, days=15)
    except Exception:
        df = None
    if df is None or len(df) < 2:
        try:
            df = get_index_daily_data(symbol=s, days=15)
        except Exception:
            df = None
    if df is None or len(df) < 2:
        print(f"{s:10s} NO DATA")
        continue
    c = df["close"].astype(float)
    ret = c.iloc[-1] / c.iloc[-10] - 1.0 if len(c) >= 11 else c.iloc[-1] / c.iloc[0] - 1.0
    w = pos[s]["market_value"] / tot_mv if tot_mv else 0.0
    contrib = w * ret
    print(f"{s:10s} ret={ret*100:+7.2f}% w~{w*100:5.2f}% contrib~{contrib*100:+6.3f}%")
