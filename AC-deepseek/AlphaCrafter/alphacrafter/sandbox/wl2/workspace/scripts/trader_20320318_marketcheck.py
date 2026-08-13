"""Trader cycle94 market check: recent returns + live factor snapshot visible through 2032-03-17."""
import json
from pathlib import Path
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

closes = {}
for a in ASSETS:
    try:
        df = get_stock_daily_data(a, days=300)
        if df is not None and "close" in df and len(df) >= 130:
            closes[a] = df["close"].astype(float).reset_index(drop=True)
    except Exception:
        pass

print(f"{'asset':10s} {'px':>10s} {'r20':>8s} {'r60':>8s} {'r120':>8s}")
for a in ASSETS:
    c = closes.get(a)
    if c is None:
        print(f"{a:10s}  NO DATA")
        continue
    px = c.iloc[-1]
    r20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) >= 21 else float("nan")
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) >= 61 else float("nan")
    r120 = c.iloc[-1] / c.iloc[-121] - 1 if len(c) >= 121 else float("nan")
    print(f"{a:10s} {px:10.3f} {r20*100:7.2f}% {r60*100:7.2f}% {r120*100:7.2f}%")

# macro observations
for s in ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]:
    try:
        df = get_index_daily_data(s, days=60)
        if df is not None and len(df) >= 21:
            c = df["close"].astype(float)
            print(f"{s:7s} last={c.iloc[-1]:.2f} r5={ (c.iloc[-1]/c.iloc[-6]-1)*100:+.2f}% r20={(c.iloc[-1]/c.iloc[-21]-1)*100:+.2f}%")
    except Exception as e:
        print(s, "ERR", e)

# current account weights
acc = json.load(open("../persistent/account.json"))
na = acc.get("net_assets", 1.0)
print("\ncurrent weights:")
for p in acc.get("positions", []):
    print(f"  {p['symbol']:10s} {p['market_value']/na*100:6.2f}%")
