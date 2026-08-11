"""Regime assessment v2 - per-asset own calendar, no NaN artifacts."""
import pandas as pd
import numpy as np

VISIBLE = "2026-07-29"
UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

px = {}
for s in UNIVERSE:
    d = pd.read_csv(f"../persistent/stock_data/{s}.csv")
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= VISIBLE].set_index("date").sort_index()
    px[s] = d["close"].dropna()

print("=" * 96)
print(f"REGIME ASSESSMENT v2 (visible through {VISIBLE})")
print("=" * 96)

print("\n[1] TREND - cumulative returns by horizon (%) [own calendar]")
hors = [5, 21, 63, 126, 252]
for h in hors:
    row = []
    for s in UNIVERSE:
        c = px[s]
        if len(c) > h:
            row.append(f"{s}:{(c.iloc[-1]/c.iloc[-1-h]-1)*100:+7.1f}")
        else:
            row.append(f"{s}:  n/a")
    print(f"  h={h:>4}: " + "  ".join(row))

# Common-date EW basket (only dates where >= 12 assets have data)
pxdf = pd.DataFrame(px)
common = pxdf.dropna(thresh=12)
ew = common.mean(axis=1)
ew_ret = ew.pct_change().dropna()
print("\n  EW-basket (common dates, >=12 assets) cumulative:")
for h in hors:
    if len(ew) > h:
        print(f"    h={h:>4}: {(ew.iloc[-1]/ew.iloc[-1-h]-1)*100:+7.2f}%")

print("\n[2] RISK - realized vol (ann. %) [own calendar]")
for s in UNIVERSE:
    r = px[s].pct_change().dropna()
    v20 = r.iloc[-20:].std() * np.sqrt(252) * 100
    v60 = r.iloc[-60:].std() * np.sqrt(252) * 100
    print(f"  {s:<10} vol20={v20:6.1f}%  vol60={v60:6.1f}%")

print("\n[3] CORRELATION (60d, common dates)")
r60 = ew_ret.iloc[-60:]
print(f"  EW-basket vol20={ew_ret.iloc[-20:].std()*np.sqrt(252)*100:.1f}%  vol60={r60.std()*np.sqrt(252)*100:.1f}%")

# Pairwise corr on common dates
rall = common.pct_change()
rc = rall.iloc[-60:]
corr = rc.corr()
mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
print(f"  60d avg |corr|={corr.where(mask).abs().stack().mean():.3f}  avg corr={corr.where(mask).stack().mean():.3f}")
print(f"  60d cross-sectional daily std (dispersion) mean={rc.std(axis=1).mean()*100:.2f}%/day")

print("\n[4] MOMENTUM SPREAD (cross-sectional, 60d return dispersion)")
for h in [21, 63]:
    vals = {}
    for s in UNIVERSE:
        c = px[s]
        if len(c) > h:
            vals[s] = c.iloc[-1] / c.iloc[-1 - h] - 1
    v = pd.Series(vals) * 100
    print(f"  h={h}: best={v.idxmax()}({v.max():+.1f}%) worst={v.idxmin()}({v.min():+.1f}%) "
          f"std={v.std():.1f}pp  p90-p10={v.quantile(.9)-v.quantile(.1):.1f}pp")

print("\n[5] RECENT 20d PRICE ACTION SUMMARY")
for s in UNIVERSE:
    c = px[s]
    r = c.pct_change().dropna()
    last5 = (c.iloc[-1] / c.iloc[-6] - 1) * 100 if len(c) > 6 else np.nan
    print(f"  {s:<10} 20d={ (c.iloc[-1]/c.iloc[-21]-1)*100:+7.2f}%  5d={last5:+6.2f}%  "
          f"last_daily={r.iloc[-1]*100:+6.2f}%")
