import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("TOTAL_ASSETS", acct["total_assets"])
print("CASH", acct["available_cash"])
print("GROSS_POS", acct["gross_position_rate"], "NET_POS", acct["net_position_rate"])
positions = {p["symbol"]: p for p in acct.get("positions", [])}
print("N_POSITIONS", len(positions))
for s, p in sorted(positions.items()):
    print(f"  {s:10s} qty={p['quantity']:12.4f} mv={p['market_value']:12.2f} pnl={p['profit_loss']:12.2f} pnl%={p['profit_loss_rate']:8.3f}")

wl = acct.get("watch_list", [])
print("WATCHLIST", wl)

print("\nBlock return proxy (close ~15 trading days ago -> latest close):")
for s in wl:
    df = None
    try:
        df = get_stock_daily_data(s, days=30)
    except Exception:
        df = None
    if df is None or len(df) < 15:
        try:
            df = get_index_daily_data(s, days=30)
        except Exception:
            df = None
    if df is None or len(df) < 15:
        print(f"  {s:10s} NO DATA")
        continue
    df = df.sort_values("date")
    last = float(df.iloc[-1]["close"])
    prev = float(df.iloc[-15]["close"])
    ret = (last - prev) / prev
    print(f"  {s:10s} prev={prev:12.4f} last={last:12.4f} ret={ret*100:8.2f}%  ({df.iloc[-1]['date'].date()})")
