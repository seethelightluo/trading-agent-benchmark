"""Trader regime snapshot at 2035-08-24 decision (data through 2035-08-23)."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

WATCH = get_account_dict().get("watch_list", [])
DAYS = 170
frames = {}
for a in WATCH:
    try:
        df = get_stock_daily_data(symbol=a, days=DAYS)
        if df is None or len(df) < 60:
            frames[a] = None
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        frames[a] = df.set_index("date").sort_index()
    except Exception as e:
        frames[a] = None

rets = {}
ma20_up = []
for a in WATCH:
    df = frames.get(a)
    if df is None or len(df) < 30:
        continue
    c = df["close"].astype(float)
    rets[a] = c.iloc[-1] / c.iloc[-21] - 1.0
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    ma20_up.append((a, c.iloc[-1] > ma20, c.iloc[-1] > ma60, (c.iloc[-1] / c.iloc[-21] - 1.0)))

# cross-sectional daily dispersion over last 20 days
px = pd.DataFrame({a: df["close"].astype(float) for a, df in frames.items() if df is not None and len(df) > 25})
disp = px.pct_change().tail(20).std(axis=1).mean() if len(px) > 5 else float("nan")

# VIX
try:
    vix = pd.read_csv("../persistent/index_data/VIX.csv")
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix[vix["date"] <= pd.Timestamp("2035-08-23")].sort_values("date")
    vix_c = vix["close"].astype(float)
    vix_now = float(vix_c.iloc[-1])
    vix_20 = float(vix_c.iloc[-21]) if len(vix_c) > 20 else float("nan")
    vix_60 = float(vix_c.iloc[-61]) if len(vix_c) > 60 else float("nan")
except Exception:
    vix_now = vix_20 = vix_60 = float("nan")

order = sorted(rets.items(), key=lambda kv: kv[1], reverse=True)
print("20d returns (sorted):")
for a, r in order:
    print(f"  {a:10s} {r*100:7.2f}%")
breadth20 = sum(1 for _, b, _, _ in ma20_up if b)
breadth60 = sum(1 for _, _, b, _ in ma20_up if b)
print(f"breadth MA20: {breadth20}/{len(ma20_up)}  MA60: {breadth60}/{len(ma20_up)}")
print(f"20d x-sect daily dispersion: {disp*100:.2f}%")
print(f"VIX now {vix_now:.1f}  20d ago {vix_20:.1f}  60d ago {vix_60:.1f}")

acct = get_account_dict()
print("NAV:", round(acct.get("net_assets", 0), 2), "gross:", acct.get("gross_position_rate"))
