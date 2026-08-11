"""miner_3: horizon detail for borderline library factors (2027-05-13 panel)."""
import numpy as np
import pandas as pd
import pickle

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"]
lr = C.pct_change()
VIX = panel["macro"]["VIX"]
vix_ret = VIX.pct_change()

vol20 = lr.rolling(20).std()
beta60 = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
for c in C.columns:
    beta60[c] = (C[c].pct_change().rolling(60, min_periods=30).cov(vix_ret)
                 / vix_ret.rolling(60, min_periods=30).var())
vix_move20 = VIX / VIX.shift(20) - 1.0

factors = {
    "mom_120d_skip5": C.shift(5) / C.shift(125) - 1.0,
    "vol_of_vol20x60": vol20.rolling(60).std(),
    "vix_beta_cond_60x20": -beta60.mul(vix_move20, axis=0),
}

def daily_ic_series(f, h):
    fwd = C.shift(-h) / C - 1.0
    out = []
    for dt in f.index:
        ff, rr = f.loc[dt], fwd.loc[dt]
        m = ff.notna() & rr.notna()
        if m.sum() < 8:
            continue
        ic = ff[m].rank().corr(rr[m].rank())
        if np.isfinite(ic):
            out.append((dt, ic))
    return pd.Series({d: v for d, v in out})

for name, f in factors.items():
    print(f"== {name} ==")
    for h in (1, 2, 3, 5, 10):
        s = daily_ic_series(f, h)
        if len(s) == 0:
            continue
        ic = s.mean(); icir = ic / s.std(ddof=1) if s.std(ddof=1) > 0 else 0
        cut = s.index.max() - pd.Timedelta(days=365)
        s12 = s[s.index >= cut]
        ic12 = s12.mean() if len(s12) else np.nan
        icir12 = ic12 / s12.std(ddof=1) if len(s12) > 2 and s12.std(ddof=1) > 0 else np.nan
        cut6 = s.index.max() - pd.Timedelta(days=183)
        s6 = s[s.index >= cut6]
        ic6 = s6.mean() if len(s6) else np.nan
        icir6 = ic6 / s6.std(ddof=1) if len(s6) > 2 and s6.std(ddof=1) > 0 else np.nan
        print(f"  h={h:2d} IC={ic:+.5f} ICIR={icir:+.5f} n={len(s):4d} | 12m IC={ic12:+.5f} ICIR={icir12:+.5f} | 6m IC={ic6:+.5f} ICIR={icir6:+.5f}")
