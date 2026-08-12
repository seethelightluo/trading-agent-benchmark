"""Trader block analysis: 2030-10-25 -> 2030-11-08 (data window 10-24 -> 11-07)."""
import json
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
RECON = pd.Timestamp("2030-10-24")   # recon close used for 1025 execution
LAST = pd.Timestamp("2030-11-07")    # last completed trading day (marking)

acc = json.load(open("../persistent/account.json"))
nav_end = acc["net_assets"]
nav_start = 912881.20  # end NAV of prior block (1011 cycle)
pos_qty = {p["symbol"]: p["quantity"] for p in acc["positions"]}

rows = []
for a in ASSETS:
    df = get_stock_daily_data(symbol=a, days=40)
    if df is None or len(df) == 0:
        rows.append((a, None, None, None, None, None))
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    closes = df["close"].astype(float)
    c0 = closes[closes.index <= RECON]
    c1 = closes[closes.index <= LAST]
    if len(c0) == 0 or len(c1) == 0:
        rows.append((a, None, None, None, None, None))
        continue
    p0 = float(c0.iloc[-1]); p1 = float(c1.iloc[-1])
    if p0 <= 0:
        rows.append((a, p0, p1, None, None, None))
        continue
    ret = p1 / p0 - 1.0
    qty = pos_qty.get(a, 0.0)
    mv0 = qty * p0
    w0 = mv0 / nav_start
    contrib = w0 * ret * 100.0
    rows.append((a, p0, p1, ret * 100.0, w0 * 100.0, contrib))

print(f"{'asset':9s} {'px0':>10s} {'px1':>10s} {'ret%':>8s} {'w0%':>7s} {'contrib%':>9s}")
tot = 0.0
for a, c0, c1, ret, w0, contrib in sorted(rows, key=lambda r: -(r[5] if r[5] is not None else -9e9)):
    if ret is None:
        print(f"{a:9s}  NO DATA / flat artifact")
        continue
    print(f"{a:9s} {c0:10.4f} {c1:10.4f} {ret:8.2f} {w0:7.2f} {contrib:9.2f}")
    tot += contrib
print(f"\nSum contrib: {tot:.2f}%  | actual block PnL: {(nav_end/nav_start-1)*100:.2f}%  NAV {nav_start:.2f} -> {nav_end:.2f}")

# regime snapshot at recon (visible through 10-24)
print("\n--- regime at recon (20d mean daily ret, % of assets above MA20) ---")
rets = []
above = 0
n = 0
for a in ASSETS:
    df = get_stock_daily_data(symbol=a, days=60)
    if df is None or len(df) < 30:
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    close = df["close"].astype(float)
    c = close[close.index <= RECON]
    if len(c) < 25:
        continue
    rets.append(float(c.pct_change().tail(20).mean()))
    ma20 = float(c.rolling(20).mean().iloc[-1])
    if float(c.iloc[-1]) > ma20:
        above += 1
    n += 1
import numpy as np
print(f"mean 20d daily ret: {np.mean(rets)*100:.4f}%  above-MA20: {above}/{n}")
