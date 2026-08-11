"""miner_3: debug vix_beta_cond_60x20 (no data in revalidation) + check mom at admission horizon h=10."""
import numpy as np
import pandas as pd
import pickle

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"]; M = panel["macro"]

vix = M["VIX"]
vix_ret = vix.pct_change()
cov60 = C.pct_change().rolling(60).cov(vix_ret)
var60 = vix_ret.rolling(60).var()
beta_vix = cov60 / var60
cond = (vix > vix.rolling(20).mean()).astype(float)
f = beta_vix * cond

print("beta_vix valid counts per asset (last):", beta_vix.notna().sum().to_dict())
print("factor valid last 5 rows sample:")
print(f.tail(3).T.iloc[:, :3])
print("factor global notna count:", int(f.notna().sum().sum()))
print("cond last value:", cond.dropna().iloc[-1] if cond.notna().any() else None)
print("cond notna:", int(cond.notna().sum()))

# check date-level valid count
valid_per_date = f.notna().sum(axis=1)
print("dates with >=8 valid:", int((valid_per_date >= 8).sum()), "of", len(f))

def daily_ic_series(fac, h):
    fwd = C.shift(-h) / C - 1.0
    ics = []
    for dt in fac.index:
        ff, rr = fac.loc[dt], fwd.loc[dt]
        m = ff.notna() & rr.notna()
        if m.sum() < 8:
            continue
        ic = ff[m].rank().corr(rr[m].rank())
        if np.isfinite(ic):
            ics.append((dt, ic))
    return pd.Series({d: v for d, v in ics})

for h in (1, 5, 10):
    s = daily_ic_series(f, h)
    if len(s):
        ic = s.mean(); icir = ic / s.std(ddof=1) if s.std(ddof=1) > 0 else 0
        print(f"vix_beta_cond h={h}: IC={ic:+.5f} ICIR={icir:+.5f} n={len(s)}")
    else:
        print(f"vix_beta_cond h={h}: no data")

# mom at admission horizon h=10
mom = C.shift(5) / C.shift(125) - 1.0
for h in (10, 5):
    s = daily_ic_series(mom, h)
    if len(s):
        ic = s.mean(); icir = ic / s.std(ddof=1) if s.std(ddof=1) > 0 else 0
        cut = s.index.max() - pd.Timedelta(days=365)
        s12 = s[s.index >= cut]
        ic12 = s12.mean(); icir12 = ic12 / s12.std(ddof=1) if s12.std(ddof=1) > 0 else 0
        print(f"mom_120d_skip5 h={h}: IC={ic:+.5f} ICIR={icir:+.5f} n={len(s)} | 12m IC={ic12:+.5f} ICIR12={icir12:+.5f}")
