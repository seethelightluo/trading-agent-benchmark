"""Trader post-block review: account state + per-asset block return attribution."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acc = get_account_dict()
print("total_assets", round(acc.get("total_assets", 0), 2))
print("net_assets", round(acc.get("net_assets", 0), 2))
print("available_cash", round(acc.get("available_cash", 0), 2))
print("gross_position_rate", acc.get("gross_position_rate"))
print("watch_list", acc.get("watch_list"))
print("---positions---")
for p in acc.get("positions", []):
    print(p["symbol"], "qty", round(p.get("quantity", 0), 4), "mv", round(p.get("market_value", 0), 2),
          "px", round(p.get("current_price", 0), 4), "pnl%", round(p.get("profit_loss_rate", 0), 4))
print("---orders---", acc.get("orders"))

# Block return attribution: 2032-02-19 -> 2032-03-04 (last 10 trading days)
print("---block asset returns (last 11 closes)---")
for sym in acc.get("watch_list", []):
    df = None
    try:
        df = get_stock_daily_data(symbol=sym, days=15)
    except Exception:
        pass
    if df is None or len(df) < 2:
        try:
            df = get_index_daily_data(symbol=sym, days=15)
        except Exception:
            pass
    if df is None or len(df) < 2:
        print(sym, "no data")
        continue
    df = df.sort_values("date")
    c0 = df.iloc[-11]["close"] if len(df) >= 11 else df.iloc[0]["close"]
    c1 = df.iloc[-1]["close"]
    print(sym, "ret%", round((c1 / c0 - 1) * 100, 2), "last_close", round(c1, 4))
