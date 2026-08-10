"""miner_3 probe: check grid, data coverage, library artifact alignment (2026-08-13)."""
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
print("assets:", ASSETS)

def load_asset(sym):
    df = get_stock_daily_data(sym, days=2200)
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
    if df is None:
        print(f"  {s:10s} NO DATA")
        continue
    vol_nonnull = int(df["volume"].notna().sum())
    vol_gt0 = int((df["volume"] > 0).sum())
    print(f"  {s:10s} rows={len(df)} last={df.index[-1]} vol_nonnull={vol_nonnull} vol>0={vol_gt0}")

# library artifacts
import glob, os
for f in sorted(glob.glob("factors/*.signal.npy"))[:30]:
    m = np.load(f, allow_pickle=True)
    print(f"  artifact {os.path.basename(f):45s} shape={m.shape}")
