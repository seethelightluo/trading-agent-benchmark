"""Trader pre-step market check for block start 2033-11-10 (visible 2033-11-09)."""
import json
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

acc = get_account_dict()
print("NAV:", round(acc["net_assets"], 2), "cash:", acc["available_cash"],
      "gross:", acc["gross_position_rate"])

closes = {}
for a in WATCH:
    try:
        df = get_stock_daily_data(a, days=300)
    except Exception:
        df = None
    if df is not None and "close" in df and len(df) >= 130:
        closes[a] = df["close"].astype(float)
    else:
        print("NO DATA:", a)

print("\n=== 20d / 60d / 180d returns (visible 2033-11-09) ===")
rows = []
for a, c in closes.items():
    def r(n):
        if len(c) > n:
            return c.iloc[-1] / c.iloc[-1 - n] - 1.0
        return float("nan")
    rows.append((a, r(20), r(60), r(180)))
for a, r20, r60, r180 in sorted(rows, key=lambda x: -x[1]):
    print(f"{a:10s} r20={r20*100:7.2f}%  r60={r60*100:7.2f}%  r180={r180*100:7.2f}%")

# VIX level
try:
    vf = get_index_daily_data("VIX", days=40)
    if vf is not None and "close" in vf and len(vf) >= 2:
        print("\nVIX last:", round(float(vf["close"].iloc[-1]), 2),
              "5d ago:", round(float(vf["close"].iloc[-6]), 2) if len(vf) > 5 else None)
except Exception as e:
    print("VIX err", e)

# current weights from account
na = acc["net_assets"]
print("\n=== current weights ===")
for p in sorted(acc["positions"], key=lambda x: -x["market_value"]):
    print(f"{p['symbol']:10s} wt={p['market_value']/na*100:6.2f}%  pl={p['profit_loss_rate']*100:6.2f}%")
