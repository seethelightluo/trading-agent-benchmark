"""Debug the calendar/NaN issue: union index has weekends (crypto trades daily)."""
import sys
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from factor_research_lib import load_panels, close_panel

panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()

idx = closes.index
print("total rows:", len(idx), "weekend rows:", int((idx.dayofweek >= 5).sum()),
      "weekday rows:", int((idx.dayofweek < 5).sum()))

spx = rets["SPX"]
print("\nSPX rets valid:", int(spx.notna().sum()), "of", len(spx))
print("SPX close valid:", int(closes["SPX"].notna().sum()), "of", len(closes))

wd = spx.index.dayofweek[spx.notna().values]
print("valid rets by weekday:", pd.Series(wd).value_counts().sort_index().to_dict())

s60 = rets["SPX"].rolling(60).std()
print("rolling60 std valid (default minp=60):", int(s60.notna().sum()))
s60b = rets["SPX"].rolling(60, min_periods=40).std()
print("rolling60 std valid (minp40):", int(s60b.notna().sum()))

# own-calendar returns
s = closes["SPX"].dropna()
own = s.pct_change()
print("\nSPX own-calendar ret valid:", int(own.notna().sum()))
print("own rolling60 std valid (default):", int(own.rolling(60).std().notna().sum()))
print("own rolling20 std valid (default):", int(own.rolling(20).std().notna().sum()))
print("own rolling20 std valid (minp12):", int(own.rolling(20, min_periods=12).std().notna().sum()))

# align own-calendar rets back to union index -> how many dates have >=8 valid?
own_rets = pd.DataFrame({a: closes[a].dropna().pct_change() for a in closes.columns}).reindex(closes.index)
print("\nown-calendar aligned rets valid-per-date: dates>=8 =",
      int((own_rets.notna().sum(axis=1) >= 8).sum()), "of", len(own_rets))

# rolling20 std on own-calendar aligned (minp=12)
std20 = own_rets.rolling(20, min_periods=12).std()
print("std20(minp12) valid-per-date: dates>=8 =", int((std20.notna().sum(axis=1) >= 8).sum()), "of", len(std20))
print("std20 per-asset valid:", std20.notna().sum().to_dict())

# what about 2020-01-01 start: does HSI/ETH frozen matter for fwd returns?
print("\nHSI/ETH flat since 2026-10-14 -> forward returns collapse?")
for a in ["HSI", "ETH"]:
    s2 = closes[a].dropna()
    tail = s2.tail(30)
    flat = int((tail.diff() == 0).sum())
    print(f"{a}: last30 flat={flat}")
EOF