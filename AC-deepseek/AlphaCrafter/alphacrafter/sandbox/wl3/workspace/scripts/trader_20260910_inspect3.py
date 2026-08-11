# Trader: raw CSV horizon + full rebalance_to_weights + Decision class source
import json, os, sys, inspect
sys.path.insert(0, ".")
import pandas as pd

print("=== raw CSVs for frozen symbols ===")
base = "../persistent/stock_data"
files = [f for f in os.listdir(base) if any(s in f.upper() for s in ["HSI", "SX5E", "BTC", "US10Y", "CN10Y", "SPX", "XAU"])]
print("matching files:", files[:20])
for f in sorted(files)[:8]:
    path = os.path.join(base, f)
    try:
        df = pd.read_csv(path)
        print(f"\n{f}: rows={len(df)} cols={list(df.columns)[:8]}")
        if "date" in df.columns:
            print("  date range:", str(df['date'].iloc[0])[:10], "->", str(df['date'].iloc[-1])[:10])
        cc = df.columns[1] if len(df.columns) > 1 else None
        if cc:
            tail = df[cc].tail(12).tolist()
            print("  last close vals:", [round(float(x), 4) for x in tail])
    except Exception as e:
        print(f, "ERR", e)

print("\n=== rebalance_to_weights full source (first 150 lines) ===")
from alphacrafter.sim.utils import rebalance_to_weights
src = inspect.getsource(rebalance_to_weights)
for i, ln in enumerate(src.split("\n")[:150]):
    print(f"{i:3d}| {ln}")
