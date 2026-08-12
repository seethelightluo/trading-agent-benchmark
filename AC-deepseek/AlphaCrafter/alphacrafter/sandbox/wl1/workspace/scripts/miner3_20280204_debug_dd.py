import pandas as pd, numpy as np, pickle
panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"]
ret = C.pct_change()

for w in (60, 120, 250):
    fac = C / C.rolling(w).max() - 1.0
    fwd10 = C.shift(-10) / C - 1.0
    n_valid = (fac.notna() & fwd10.notna()).sum(axis=1)
    dates = n_valid[n_valid >= 8].index
    print(f"dd_{w}d: valid dates={len(dates)}, span {dates.min().date()} -> {dates.max().date()}")
    # count valid per year
    yrs = dates.to_period('Y').value_counts().sort_index()
    print("  by year:", dict(yrs))
    # how many dates in last 12m
    cut = dates.max() - pd.Timedelta(days=365)
    print("  last12m dates:", int((dates >= cut).sum()))
