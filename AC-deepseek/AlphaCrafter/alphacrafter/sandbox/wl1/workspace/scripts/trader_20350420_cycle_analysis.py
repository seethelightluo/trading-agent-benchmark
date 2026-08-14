"""Trader cycle analysis for block 0406 -> 0420 (decision 0406, gate PASSED).
Regime context at decision (0405 close) + block performance attribution.
Macro signals read from persistent/index_data (observation-only).
"""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = {"DXY": "DXY", "USDCNY": "USDCNY", "USDJPY": "USDJPY", "EURUSD": "EURUSD", "VIX": "VIX"}

def fetch(sym, days=260):
    df = get_stock_daily_data(symbol=sym, days=days)
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def fetch_macro(name, days=260):
    path = f"../persistent/index_data/{name}.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    close_col = "close" if "close" in df.columns else df.columns[1]
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "close"]].dropna()

def last_close_on_or_before(df, dstr):
    d = pd.Timestamp(dstr)
    sub = df[df["date"] <= d]
    if len(sub) == 0:
        return None
    return float(sub.iloc[-1]["close"])

def ret_between(df, d0, d1):
    c0 = last_close_on_or_before(df, d0)
    c1 = last_close_on_or_before(df, d1)
    if c0 is None or c1 is None or c0 == 0:
        return None
    return c1 / c0 - 1.0

acc = json.load(open("../persistent/account.json"))
last = acc["rebalance_history"][-1]
cur_nav = acc["net_assets"]
print("last rebalance:", last["date"], "pre_trade_nav", round(last["pre_trade_nav"], 2),
      "post_trade_nav", round(last["post_trade_nav"], 2), "cost", round(last["cost"], 2))
print("current net_assets:", round(cur_nav, 2))
print("block PnL (post-trade basis):", f"{(cur_nav/last['post_trade_nav']-1)*100:.2f}%")

# ---- regime context at decision: data visible through 0405 ----
print("\n--- Regime at decision (through 0405 close) ---")
frames = {s: fetch(s) for s in WATCH}
for m in MACRO:
    frames[m] = fetch_macro(m)

rets20 = []
for s in WATCH:
    r = ret_between(frames[s], "2035-03-06", "2035-04-05")
    if r is not None:
        rets20.append(r)
eqw20 = np.mean(rets20)
print("20d eqw cum:", f"{eqw20*100:.2f}%", f"(mean daily {eqw20/20*100:.3f}%)")

def pct_above_ma(df, span):
    d = pd.Timestamp("2035-04-05")
    sub = df[df["date"] <= d]
    if len(sub) < span + 5:
        return None
    close = sub["close"].values
    ma = pd.Series(close).rolling(span).mean().values[-1]
    return close[-1] > ma

b20 = sum(1 for s in WATCH if pct_above_ma(frames[s], 20) is True)
b60 = sum(1 for s in WATCH if pct_above_ma(frames[s], 60) is True)
print("breadth above MA20:", b20, "/15  above MA60:", b60, "/15")

vix = frames["VIX"]
vix_now = last_close_on_or_before(vix, "2035-04-05")
vix_10 = last_close_on_or_before(vix, "2035-03-23")
vix_20 = last_close_on_or_before(vix, "2035-03-06")
print("VIX now/10d/20d:", round(vix_now,1) if vix_now else None,
      round(vix_10,1) if vix_10 else None, round(vix_20,1) if vix_20 else None)

dates = frames[WATCH[0]]["date"].values
dd = pd.Timestamp("2035-04-05")
valid_dates = [x for x in dates if x <= dd][-22:]
disp = []
for i in range(1, len(valid_dates)):
    rs = []
    for s in WATCH:
        df = frames[s]
        c0 = last_close_on_or_before(df, str(valid_dates[i-1])[:10])
        c1 = last_close_on_or_before(df, str(valid_dates[i])[:10])
        if c0 and c1 and c0 > 0:
            rs.append(c1/c0 - 1.0)
    if len(rs) >= 10:
        disp.append(np.std(rs))
print("20d mean daily x-sect stdev:", f"{np.mean(disp)*100:.2f}%")

print("\n--- Macro ---")
for m in MACRO:
    df = frames[m]
    c0 = last_close_on_or_before(df, "2035-03-06")
    c1 = last_close_on_or_before(df, "2035-04-05")
    if c0 and c1:
        print(f"{m:8s} now={c1:.2f} 20d_chg={((c1/c0-1)*100):.2f}%")

print("\n--- 20d returns at decision (through 0405) ---")
rr = {}
for s in WATCH:
    r = ret_between(frames[s], "2035-03-06", "2035-04-05")
    if r is not None:
        rr[s] = r
for s, r in sorted(rr.items(), key=lambda x: -x[1]):
    print(f"{s:12s} {r*100:7.2f}%")

print("\n--- Block returns 0406 -> 0419 ---")
for s in WATCH:
    r = ret_between(frames[s], "2035-04-06", "2035-04-19")
    print(f"{s:12s} {r*100:7.2f}%" if r is not None else f"{s:12s}  n/a")
