"""miner_2 regime probe as of 2026-08-13 (visible through 2026-08-12)."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

DATE_PATH = "../persistent/date.json"
date_state = json.load(open(DATE_PATH))
TRADING_DAYS = date_state["trading_days"]
VISIBLE = date_state["visible_through"]
ROW0 = TRADING_DAYS.index("2020-01-01")
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
print(f"grid rows: {len(GRID)}  {GRID[0]}..{GRID[-1]}")

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))
print("assets:", len(ASSETS), ASSETS)

def load_asset(sym):
    df = get_stock_daily_data(sym, days=2100)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

DATA = {s: load_asset(s) for s in ASSETS}
for s, df in DATA.items():
    print(f"  {s:10s} rows={0 if df is None else len(df)} last={0 if df is None else df.index[-1]}")

# regime stats over last 20/60 trading days of the grid
W = GRID[-60:]
W20 = GRID[-20:]
def ret_over(win):
    out = {}
    for s, df in DATA.items():
        if df is None:
            continue
        sub = df.loc[df.index.intersection(win)]
        if len(sub) < 2:
            continue
        r = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1
        out[s] = r
    return out

r20 = ret_over(W20)
r60 = ret_over(W)
print("\n=== 20d returns ===")
for s, r in sorted(r20.items(), key=lambda kv: -kv[1]):
    print(f"  {s:10s} {r:+.2%}")
print("\n=== 60d returns ===")
for s, r in sorted(r60.items(), key=lambda kv: -kv[1]):
    print(f"  {s:10s} {r:+.2%}")

# VIX level
vix = pd.read_csv("../persistent/index_data/VIX.csv", parse_dates=["date"])
vix["date"] = pd.to_datetime(vix["date"]).dt.strftime("%Y-%m-%d")
vix = vix.set_index("date")
vix_last = vix.loc[vix.index.intersection(GRID)]
print("\nVIX last:", vix_last["close"].iloc[-1], "20d ago:", vix_last["close"].iloc[-21] if len(vix_last) > 21 else "na")

# dispersion: cross-sectional std of 20d returns
import statistics
rs = list(r20.values())
print(f"cross-sectional dispersion 20d: mean={np.mean(rs):+.2%} std={np.std(rs):.2%}")
