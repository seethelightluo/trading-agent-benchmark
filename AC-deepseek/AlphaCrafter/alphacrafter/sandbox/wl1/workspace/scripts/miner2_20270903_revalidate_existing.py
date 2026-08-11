"""miner_2: re-validate existing factor library on panel through 2027-09-02.
Computes daily rank IC vs 1d forward return for all miner_2 factors.
Full-sample + last-252d window for drift monitoring.
Gate: |IC1| >= 0.0070 and |ICIR1| >= 0.0840 (daily paper IC, non-annualized mean/std).
"""
import pandas as pd
import numpy as np
import pickle, json, sys

with open("scripts/panel_cache.pkl", "rb") as fh:
    P = pickle.load(fh)
C, O, H, L, V, R = P["close"], P["open"], P["high"], P["low"], P["vol"], P["ret"]

def rank_ic_series(factor, fwd=1):
    """Daily cross-sectional Spearman IC between factor (t) and forward return (t+1)."""
    fwd_ret = C.shift(-fwd) / C - 1.0
    dates, ics = [], []
    for dt in factor.index:
        fv = factor.loc[dt]
        fr = fwd_ret.loc[dt]
        m = fv.notna() & fr.notna()
        if m.sum() >= 8:
            ics.append(fv[m].rank().corr(fr[m].rank()))
            dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def metrics(ic_series):
    ic = ic_series.mean()
    icir = ic / ic_series.std() if ic_series.std() > 0 else 0.0
    hit = (ic_series > 0).mean()
    return ic, icir, hit, len(ic_series)

FACTORS = {
    "nclv_1d":   lambda: -(C - L) / (H - L),
    "nclv_2d":   lambda: -(C - L.rolling(2).min()) / (H.rolling(2).max() - L.rolling(2).min()),
    "nclv_3d":   lambda: -(C - L.rolling(3).min()) / (H.rolling(3).max() - L.rolling(3).min()),
    "nclv_5d":   lambda: -(C - L.rolling(5).min()) / (H.rolling(5).max() - L.rolling(5).min()),
    "rev_1d":    lambda: -(np.log(C) - np.log(C.shift(1))),
    "rev_2d":    lambda: -(np.log(C) - np.log(C.shift(2))),
    "rev_3d":    lambda: -(np.log(C) - np.log(C.shift(3))),
    "rev_5d":    lambda: -(np.log(C) - np.log(C.shift(5))),
    "rev_1d_vs": lambda: -(np.log(C) - np.log(C.shift(1))) / (V.rolling(5).mean() / V.rolling(5).mean()),
    "id_rev_1d": lambda: -(C / O - 1.0),
    "nbody_1d":  lambda: -(C - O) / (H - L),
}

print(f"{'factor':<12} {'full_IC1':>8} {'full_ICIR':>8} {'full_hit':>7} {'full_N':>6} | {'rec_IC1':>8} {'rec_ICIR':>8} {'rec_N':>5} {'rec_pass':>8}")
results = {}
for name, fn in FACTORS.items():
    try:
        F = fn()
    except Exception as e:
        print(name, "ERR", e); continue
    full = rank_ic_series(F)
    ic, icir, hit, n = metrics(full)
    rec = full[full.index >= full.index[-1] - pd.Timedelta(days=400)]  # ~252 trading days
    ric, ricir, rhit, rn = metrics(rec)
    rec_pass = (abs(ric) >= 0.0070 and abs(ricir) >= 0.0840)
    print(f"{name:<12} {ic:>8.4f} {icir:>8.3f} {hit:>7.3f} {n:>6d} | {ric:>8.4f} {ricir:>8.3f} {rn:>5d} {'PASS' if rec_pass else 'fail':>8}")
    results[name] = {"full": dict(ic=ic, icir=icir, hit=hit, n=n),
                     "recent": dict(ic=ric, icir=ricir, hit=rhit, n=rn)}

with open("scripts/miner2_20270903_reval_results.json", "w") as fh:
    json.dump(results, fh, indent=2, default=float)
print("\nsaved scripts/miner2_20270903_reval_results.json")
