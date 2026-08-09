"""Check data depth: how far back can we fetch, and what does index_data cover?"""
import os
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

# Try fetching more days
for days in (500, 800, 1200, 2000):
    try:
        df = get_stock_daily_data("SPX", days=days)
        print(f"SPX days={days}: rows={len(df)} first={df.iloc[0]['date']} last={df.iloc[-1]['date']}")
    except Exception as e:
        print(f"SPX days={days}: ERR {e}")

print("\n--- index_data date ranges ---")
for f in sorted(os.listdir("../persistent/index_data")):
    p = os.path.join("../persistent/index_data", f)
    df = pd.read_csv(p)
    print(f, "rows=", len(df), "first=", df.iloc[0]['date'], "last=", df.iloc[-1]['date'])

# Check whether there are more persistent subfolders
for root, dirs, files in os.walk("../persistent"):
    depth = root.count(os.sep)
    if depth <= 2:
        print("DIR:", root, "->", dirs[:10], "| files:", len(files))