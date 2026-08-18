"""Trader cycle dump: account state + per-symbol block PnL for memory logging."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

acc = get_account_dict()
print("total_assets:", round(acc.get("total_assets", 0), 2))
print("available_cash:", round(acc.get("available_cash", 0), 2))
print("market_value:", round(acc.get("market_value", 0), 2))
print("gross_position_rate:", acc.get("gross_position_rate"))
print("orders:", len(acc.get("orders", [])))

positions = {p["symbol"]: p for p in acc.get("positions", [])}
print("n_positions:", len(positions))

def get_px(sym, days=40):
    df = get_stock_daily_data(sym, days=days)
    if df is None or len(df) == 0:
        df = get_index_daily_data(sym, days=days)
    return df

# per-position info
for sym, p in sorted(positions.items()):
    qty = p.get("quantity", 0)
    mv = p.get("market_value", 0)
    pl = p.get("profit_loss", 0)
    plr = p.get("profit_loss_rate", 0)
    cost = p.get("cost_price", 0)
    cur = p.get("current_price", 0)
    # weight
    w = mv / acc.get("total_assets", 1) if acc.get("total_assets") else 0
    print(f"{sym}: qty={qty:.4f} cost={cost:.4f} cur={cur:.4f} mv={mv:.2f} w={w:.4f} pl={pl:.2f} plr={plr*100:.2f}%")

# block price moves (last 11 trading days visible before today)
print("\n--- block 10d price moves ---")
for sym in sorted(positions.keys()):
    df = get_px(sym, days=30)
    if df is None or len(df) < 12:
        print(sym, "no data")
        continue
    df = df.sort_values("date")
    p0 = float(df.iloc[-11]["close"])
    p1 = float(df.iloc[-1]["close"])
    print(f"{sym}: {(p1/p0-1)*100:+.2f}%")
