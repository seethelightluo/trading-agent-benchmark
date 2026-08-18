"""Analyze previous block (2027-10-21 -> 2027-11-04) per-asset PnL for memory log."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
print("=== ACCOUNT ===")
print("total_assets", round(acc.get("total_assets", 0), 2))
print("net_assets", round(acc.get("net_assets", 0), 2))
print("available_cash", round(acc.get("available_cash", 0), 2))
print("gross_position_rate", round(acc.get("gross_position_rate", 0), 4))
print("watch_list", acc.get("watch_list", []))

pos = {p["symbol"]: p for p in acc.get("positions", [])}
print("\n=== POSITIONS (current) ===")
for s, p in sorted(pos.items(), key=lambda x: -x[1].get("market_value", 0)):
    print(f"{s}: qty={p.get('quantity',0):.4f} mv={p.get('market_value',0):.2f} "
          f"pnl_rate={p.get('profit_loss_rate',0)*100:.2f}%")

# Per-asset returns over the block 2027-10-21 -> 2027-11-04
# Strategy made decisions on 2027-10-21; holdings through 2027-11-04.
# Approximate: asset close return over last N days (find 2027-10-21 bar).
import pandas as pd

print("\n=== PER-ASSET BLOCK RETURNS (approx from holdings perspective) ===")
for s in acc.get("watch_list", []):
    df = get_stock_daily_data(symbol=s, days=40)
    if df is None or len(df) < 2:
        print(s, "no data")
        continue
    df = df.sort_values("date").reset_index(drop=True)
    # last date
    last_date = df.iloc[-1]["date"]
    # find the bar closest to 2027-10-21
    target = pd.Timestamp("2027-10-21")
    idx = df.index[df["date"] <= target]
    if len(idx) == 0:
        print(s, "no bar before block start")
        continue
    i0 = idx[-1]
    c0 = df.iloc[i0]["close"]
    c1 = df.iloc[-1]["close"]
    ret = (c1 / c0 - 1.0) * 100
    w = pos.get(s, {}).get("market_value", 0) / acc.get("net_assets", 1)
    print(f"{s}: ret={ret:+.2f}% weight_now={w*100:.2f}% last_date={last_date.date()}")
