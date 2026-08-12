import sys
sys.path.insert(0, "scripts")
from miner_3_20260813_lib import GRID, ASSETS, load_asset
print("grid rows:", len(GRID), "first:", GRID[0], "last:", GRID[-1])
print("assets:", ASSETS)
for s in ASSETS:
    df = load_asset(s)
    print(s, None if df is None else len(df), None if df is None else str(df.index[-1]))
