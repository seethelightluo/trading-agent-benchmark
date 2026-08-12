"""Trader audit 2027-11-18: verify frozen (zero-movement) assets since online start."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

OBS_ONLY = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
ONLINE_START = "2026-07-16"

account = get_account_dict()
watch = account.get("watch_list", [])
print("watch_list:", watch)

def get_df(symbol, days=800):
    try:
        if symbol in OBS_ONLY:
            return get_index_daily_data(symbol, days=days)
        return get_stock_daily_data(symbol, days=days)
    except Exception as e:
        print(symbol, "ERR", e)
        return None

rows = []
for s in watch:
    df = get_df(s)
    if df is None or len(df) < 100:
        rows.append((s, "NO_DATA", None, None, None, None))
        continue
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    post = df[df["date"] >= ONLINE_START]
    if len(post) == 0:
        rows.append((s, "NO_POST", None, None, None, None))
        continue
    c = post["close"].astype(float)
    n = len(c)
    n_unique = c.nunique()
    pct_move = (c.iloc[-1] / c.iloc[0] - 1.0) * 100.0
    std = c.pct_change().std()
    rows.append((s, "OK", n, n_unique, pct_move, std))

print(f"{'symbol':10s} {'status':8s} {'days':>6s} {'unique':>7s} {'tot_pct':>9s} {'std':>8s}")
for r in rows:
    print(f"{r[0]:10s} {r[1]:8s} {str(r[2]):>6s} {str(r[3]):>7s} {('%.3f'%r[4]) if r[4] is not None else '-':>9s} {('%.5f'%r[5]) if r[5] is not None else '-':>8s}")

frozen = [r[0] for r in rows if r[2] is not None and r[3] is not None and r[3] <= 2 and abs(r[4]) < 1e-9]
print("\nFROZEN (<=2 unique closes since online start):", frozen)
