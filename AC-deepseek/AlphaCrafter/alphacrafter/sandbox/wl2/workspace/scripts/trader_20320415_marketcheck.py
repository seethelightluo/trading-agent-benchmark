"""Trader pre-step market check for 2032-04-15 block (visible through 04-14)."""
import json
from datetime import date
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WL = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
      "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

print("=== 15-asset recent returns (visible 04-14) ===")
for a in WL:
    try:
        df = get_stock_daily_data(a, days=200)
    except Exception:
        df = None
    if df is None or len(df) < 30:
        print(f"{a:10s} no data")
        continue
    c = df["close"].astype(float)
    def r(n):
        return (c.iloc[-1] / c.iloc[-1 - n] - 1.0) * 100.0 if len(c) > n else float("nan")
    print(f"{a:10s} 5d {r(5):7.2f}%  10d {r(10):7.2f}%  20d {r(20):7.2f}%  60d {r(60):7.2f}%  180d {r(180):7.2f}%")

print("\n=== macro observation signals ===")
for a in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    try:
        df = get_index_daily_data(a, days=80)
    except Exception:
        df = None
    if df is None or len(df) < 10:
        print(f"{a:8s} no data")
        continue
    c = df["close"].astype(float)
    print(f"{a:8s} last {c.iloc[-1]:9.2f}  5d {(c.iloc[-1]/c.iloc[-6]-1)*100:7.2f}%  20d {(c.iloc[-1]/c.iloc[-21]-1)*100:7.2f}%")

print("\n=== account ===")
acct = json.load(open("../persistent/account.json"))
print("NAV", round(acct["net_assets"], 2), "cash", round(acct["available_cash"], 2),
      "gross", acct["gross_position_rate"])
