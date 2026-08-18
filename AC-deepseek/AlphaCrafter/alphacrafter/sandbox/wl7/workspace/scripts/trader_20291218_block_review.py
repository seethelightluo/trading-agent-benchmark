"""Post-step account + block attribution check (2029-12-04 -> 2029-12-18)."""
import json

import pandas as pd

from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
print("ACCOUNT:", json.dumps({
    "total_assets": round(acc.get("total_assets", 0), 2),
    "net_assets": round(acc.get("net_assets", 0), 2),
    "cash": round(acc.get("available_cash", 0), 2),
    "gross_position_rate": round(acc.get("gross_position_rate", 0), 4),
    "n_positions": len(acc.get("positions", [])),
    "n_orders": len(acc.get("orders", [])),
}, indent=1))

pos = {p["symbol"]: p for p in acc.get("positions", [])}
print("\nPOSITIONS (qty, mv, plr):")
for sym, p in sorted(pos.items()):
    print(f"  {sym}: qty={p['quantity']:.4f} mv={p['market_value']:.0f} plr={p['profit_loss_rate']*100:.2f}%")

# block attribution: last data day 12-17 vs 12-03 close (10 trading days)
print("\n--- block returns (12-03 -> 12-17) per asset ---")
def ret_block(sym):
    df = get_stock_daily_data(sym, days=30)
    if df is None or len(df) == 0:
        return None
    df = df.sort_values("date")
    closes = df["close"].astype(float)
    return closes.iloc[-1] / closes.iloc[0] - 1.0

tot = 0.0
for sym, p in sorted(pos.items()):
    rb = ret_block(sym)
    if rb is None:
        continue
    mv_start = p["market_value"] / (1 + rb) if (1 + rb) != 0 else p["market_value"]
    contrib = mv_start * rb
    tot += contrib
    print(f"  {sym}: block_ret={rb*100:+.2f}%  wt_now={p['market_value']/acc['total_assets']*100:.1f}%  contrib={contrib:.0f}")
print(f"  TOTAL est. block PnL: {tot:.0f}")

# observation signals at block end
print("\n--- signals at block end (12-17) ---")
for s in ["VIX", "DXY"]:
    df = get_index_daily_data(s, days=30)
    if df is None or len(df) == 0:
        continue
    df = df.sort_values("date")
    c = df["close"].astype(float)
    print(f"  {s}: close={c.iloc[-1]:.2f} ret20={(c.iloc[-1]/c.iloc[-21]-1)*100:+.2f}% (date {str(df['date'].iloc[-1])[:10]})")
