"""Probe data availability up to current date (2034-05-11)."""
import sys
sys.path.insert(0, 'scripts')
import factor_common as fc
import pandas as pd
import numpy as np

prices = fc.load_prices(days=4000)
print("tradable symbols:", len(prices))
for s, df in prices.items():
    print(f"{s:10s} rows={len(df):5d} first={df.index.min().date()} last={df.index.max().date()}")

max_date = max(d.index.max() for d in prices.values())
print("\nmax visible date:", max_date.date())

for sig in fc.INDEX_SIGNALS:
    df = fc.load_index(sig, days=4000, prices=prices)
    if df is not None:
        print(f"{sig:8s} rows={len(df):5d} last={df.index.max().date()}")
    else:
        print(f"{sig:8s} None")
