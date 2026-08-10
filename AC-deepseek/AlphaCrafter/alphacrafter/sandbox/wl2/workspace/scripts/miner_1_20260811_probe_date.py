"""Probe current simulation date via the trading API."""
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

for sym, fn in [("SPX", get_index_daily_data), ("BTC", get_stock_daily_data)]:
    df = fn(symbol=sym, days=8)
    print(sym, "last dates:", list(df["date"].dt.strftime("%Y-%m-%d"))[-3:] if df is not None else None)
acct = get_account_dict()
print("account:", {k: acct[k] for k in ("total_assets", "available_cash", "watch_list") if k in acct})
