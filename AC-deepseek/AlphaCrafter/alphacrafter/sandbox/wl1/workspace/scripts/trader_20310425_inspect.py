"""Trader 2031-04-25: inspect China-name (000300/000688/HSI) trend state and
recent regime data to decide on the Screener-recommended China de-rank guard."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

acc = get_account_dict()
print("net_assets:", acc.get("net_assets"))
print("cash:", acc.get("available_cash"))
print("last_rebalance_date:", acc.get("last_rebalance_date"))
print("positions:")
for p in acc.get("positions", []):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} mv={p['market_value']:.0f} pnl%={p['profit_loss_rate']*100:.2f}")
print("pending orders:", len(acc.get("orders", [])))

assets = acc.get("watch_list", [])
print("\nwatch_list:", assets)

for a in ["000300.SH", "000688.SH", "HSI", "CN10Y"]:
    df = get_stock_daily_data(symbol=a, days=170)
    if df is None or len(df) < 30:
        print(f"\n{a}: NO DATA (len={0 if df is None else len(df)})")
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    c = df["close"].astype(float)
    ma20 = c.rolling(20).mean().iloc[-1]
    last = c.iloc[-1]
    mom20 = c.iloc[-1] / c.iloc[-21] - 1
    mom120 = c.iloc[-1] / c.iloc[-121] - 1 if len(c) > 121 else np.nan
    ret_10d = c.iloc[-1] / c.iloc[-11] - 1
    print(f"\n{a}: last={last:.2f} ma20={ma20:.2f} below_ma20={last < ma20}")
    print(f"  mom20={mom20*100:.2f}% mom120={mom120*100:.2f}% ret10d={ret_10d*100:.2f}%")

# Cross-asset 20d mean daily return for regime
rets = []
for a in assets:
    df = get_stock_daily_data(symbol=a, days=30)
    if df is not None and len(df) >= 25:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        rets.append(float(df["close"].pct_change().tail(20).mean()))
print(f"\nregime 20d mean daily (15 assets): {np.mean(rets)*100:.3f}%  n={len(rets)}")
print(f"assets above MA20: ", end="")
for a in assets:
    df = get_stock_daily_data(symbol=a, days=30)
    if df is not None and len(df) >= 25:
        df = df.copy(); df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        c = df["close"].astype(float)
        if c.iloc[-1] > c.rolling(20).mean().iloc[-1]:
            print(a, end=" ")
print()
