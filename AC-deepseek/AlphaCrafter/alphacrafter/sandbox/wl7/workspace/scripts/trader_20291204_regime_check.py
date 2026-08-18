"""Trader regime check at 2029-12-04 decision point (data thru prev day)."""
import json

from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

acc = get_account_dict()
print("ACCOUNT:", json.dumps({
    "total_assets": round(acc.get("total_assets", 0), 2),
    "net_assets": round(acc.get("net_assets", 0), 2),
    "cash": round(acc.get("available_cash", 0), 2),
    "gross_position_rate": round(acc.get("gross_position_rate", 0), 4),
    "n_positions": len(acc.get("positions", [])),
    "n_orders": len(acc.get("orders", [])),
}, indent=1))

assets = acc.get("watch_list", [])
print("WATCHLIST:", assets)

def last_close(sym, fn):
    df = fn(sym, days=70)
    if df is None or len(df) == 0:
        return None
    df = df.sort_values("date")
    ret20 = (df["close"].iloc[-1] / df["close"].iloc[-21] - 1) if len(df) > 21 else None
    ret60 = (df["close"].iloc[-1] / df["close"].iloc[-61] - 1) if len(df) > 61 else None
    return {"date": str(df["date"].iloc[-1])[:10], "close": round(float(df["close"].iloc[-1]), 3),
            "ret20": round(float(ret20) * 100, 2) if ret20 is not None else None,
            "ret60": round(float(ret60) * 100, 2) if ret60 is not None else None}

print("\n--- observation-only signals (index_data) ---")
for s in ["VIX", "DXY", "USDJPY", "EURUSD", "USDCNY"]:
    print(s, last_close(s, get_index_daily_data))

print("\n--- tradable assets (stock_data) 20d/60d momentum ---")
for a in assets:
    print(a, last_close(a, get_stock_daily_data))
