"""Trader diag 2030-01-15: check date alignment across asset series."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
import pandas as pd

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
series = {}
for a in assets:
    df = get_stock_daily_data(a, days=200)
    if df is None or len(df) == 0:
        print(a, "NO DATA")
        continue
    s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
    series[a] = s
    print(f"{a}: n={len(s)} first={s.index[0]} last={s.index[-1]} dtype={s.index.dtype}")

# check overlaps
if series:
    idx = [set(s.index) for s in series.values()]
    inter = set.intersection(*idx)
    print("\ncommon index size:", len(inter))
    if inter:
        print("sample common:", sorted(inter)[:3], "...", sorted(inter)[-3:])
    # pairwise
    names = list(series.keys())
    for i in range(min(3, len(names))):
        a = names[i]
        for b in names[i+1:]:
            ia, ib = set(series[a].index), set(series[b].index)
            ov = len(ia & ib)
            print(f"{a} vs {b}: overlap={ov} a_only={len(ia-ib)} b_only={len(ib-ia)}")
    # check for duplicates or non-unique index
    for a, s in series.items():
        if s.index.has_duplicates:
            print(f"{a}: DUPLICATE index entries!")
        if not s.index.is_monotonic_increasing:
            print(f"{a}: NON-MONOTONIC index")
