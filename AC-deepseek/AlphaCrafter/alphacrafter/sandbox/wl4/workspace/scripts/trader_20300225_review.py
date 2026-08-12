"""Trader post-cycle review for 2030-02-25..03-11 block (analysis only, no sim advance)."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("date:", acct.get("date"))
print("total_assets:", acct.get("total_assets"))
print("net_assets:", acct.get("net_assets"))
print("available_cash:", acct.get("available_cash"))
print("gross_position_rate:", acct.get("gross_position_rate"))
print("net_position_rate:", acct.get("net_position_rate"))
print("orders:", acct.get("orders"))

pos = {p["symbol"]: p for p in acct.get("positions", [])}
total = acct.get("net_assets", 0) or acct.get("total_assets", 0)
print("\npositions (%d):" % len(pos))
wsum = 0.0
for sym, p in sorted(pos.items()):
    w = p.get("market_value", 0) / total if total else 0
    wsum += w
    print(f"  {sym}: qty={p.get('quantity'):.4f} mktval={p.get('market_value'):,.0f} w={w:.4f} pnl={p.get('profit_loss'):,.0f} pnl%={p.get('profit_loss_rate')*100:.2f}%")
print("weights sum:", round(wsum, 6))

# Block returns 02-25 -> 03-11
print("\nblock returns (close-to-close over last 11 sessions):")
for sym in sorted(pos.keys()):
    df = get_stock_daily_data(symbol=sym, days=15)
    if df is None or len(df) < 11:
        df = get_index_daily_data(symbol=sym, days=15)
    if df is None or len(df) < 11:
        print(f"  {sym}: no data")
        continue
    df = df.sort_values("date")
    d0 = df.iloc[-11]["date"].date()
    d1 = df.iloc[-1]["date"].date()
    r = df.iloc[-1]["close"] / df.iloc[-11]["close"] - 1
    w = pos.get(sym, {}).get("market_value", 0) / total if total else 0
    print(f"  {sym}: {d0}->{d1} ret={r*100:+.2f}% w~{w:.3f} contrib~{r*w*100:+.2f}%")
