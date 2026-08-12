"""Cycle 50 review: compute block P&L attribution for 2029-03-22 -> 2029-04-05."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

TARGET = {
    "000300.SH": 0.0490, "SPX": 0.1400, "HSI": 0.0390, "N225": 0.0233,
    "SX5E": 0.0606, "000688.SH": 0.0901, "SOX": 0.1400, "NDX": 0.0227,
    "XAU": 0.0295, "COPPER": 0.0669, "WTI": 0.0380, "BTC": 0.0606,
    "ETH": 0.0750, "US10Y": 0.0827, "CN10Y": 0.0827,
}

START, END = "2029-03-22", "2029-04-05"

def get(a):
    try:
        return get_stock_daily_data(a, days=60)
    except Exception:
        return None

print(f"{'asset':>10} {'wt':>7} {'ret%':>8} {'contrib%':>9}")
total = 0.0
for a in ASSETS:
    df = get(a)
    if df is None or len(df) == 0:
        print(f"{a:>10} no data")
        continue
    df = df.sort_values("date")
    dts = [str(d)[:10] for d in df["date"]]
    try:
        p0 = float(df.loc[dts.index(START), "close"])
        p1 = float(df.loc[dts.index(END), "close"])
    except ValueError:
        # fall back to nearest available
        avail = [d for d in dts if START <= d <= END]
        if not avail:
            print(f"{a:>10} no dates in block")
            continue
        sub = df[(df["date"].astype(str).str[:10] >= START) & (df["date"].astype(str).str[:10] <= END)]
        if len(sub) < 2:
            print(f"{a:>10} insufficient block data")
            continue
        p0 = float(sub["close"].iloc[0]); p1 = float(sub["close"].iloc[-1])
    r = p1 / p0 - 1.0
    w = TARGET.get(a, 0.0)
    c = w * r
    total += c
    print(f"{a:>10} {w:7.2%} {r*100:8.2f} {c*100:9.3f}")

print(f"\nSum of approx contributions (pre-cost): {total*100:.3f}%")

# regime snapshot at block end
try:
    vf = get_index_daily_data("VIX", days=40)
    if vf is not None and len(vf):
        print("VIX at block end:", float(vf["close"].iloc[-1]))
except Exception as e:
    print("VIX err", e)

try:
    spx = get("SPX")
    if spx is not None and len(spx) > 21:
        c = spx.sort_values("date")["close"].astype(float)
        print(f"SPX 20d ret at end: {(c.iloc[-1]/c.iloc[-21]-1)*100:.2f}%")
except Exception as e:
    print("SPX err", e)
