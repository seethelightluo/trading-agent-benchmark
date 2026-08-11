"""Debug why rets/vol-based factors show tiny n while close-based factors show n~800."""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import load_panels, close_panel, TRADABLE

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
print(f"closes shape {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | load {time.time()-t0:.1f}s")

print("\n== closes non-NaN count per asset ==")
print(closes.notna().sum().to_string())

print("\n== rets non-NaN count per asset ==")
print(rets.notna().sum().to_string())

print("\n== closes last 3 rows ==")
print(closes.tail(3).to_string())

print("\n== rets last 3 rows ==")
print(rets.tail(3).to_string())

# check volume/ohlc presence
print("\n== panel columns and volume/open/high/low non-null counts ==")
for a in TRADABLE:
    df = panels.get(a)
    if df is None:
        print(f"{a}: MISSING")
        continue
    cols = list(df.columns)
    v = df.get("volume")
    o = df.get("open")
    h = df.get("high")
    lo = df.get("low")
    print(f"{a}: n={len(df)} cols={cols[:6]} vol_nonnull={int(v.notna().sum()) if v is not None else 'NA'} "
          f"open_nonnull={int(o.notna().sum()) if o is not None else 'NA'} "
          f"high_nonnull={int(h.notna().sum()) if h is not None else 'NA'} "
          f"low_nonnull={int(lo.notna().sum()) if lo is not None else 'NA'} "
          f"close_zero={(df['close']==0).sum() if 'close' in df else 'NA'}")

# where do NaN rows cluster in rets?
r_valid_per_date = rets.notna().sum(axis=1)
print("\n== rets valid-per-date distribution ==")
print(r_valid_per_date.describe().to_string())
print("dates with >=8 valid:", int((r_valid_per_date >= 8).sum()), "of", len(r_valid_per_date))
print("first 10 dates with >=8 valid:", [d.date() for d in r_valid_per_date[r_valid_per_date >= 8].index[:10]])
print("last 10 dates with >=8 valid:", [d.date() for d in r_valid_per_date[r_valid_per_date >= 8].index[-10:]])

# check a couple of 60d rolling std coverage
s60 = rets.rolling(60).std()
print("\n== rolling60 std non-NaN per asset ==")
print(s60.notna().sum().to_string())
print("dates with >=8 valid (std60):", int((s60.notna().sum(axis=1) >= 8).sum()))
