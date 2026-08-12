"""miner_3 2029-04-27: re-validate all library factors on panel through 2029-04-26.
Admission gate: abs IC >= 0.0070, abs ICIR >= 0.0840 (daily, h=1).
Optimized: precompute rank ICs with scipy rankdata + numpy corrcoef.
"""
import numpy as np
import pandas as pd
import pickle, json
from scipy.stats import rankdata

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"]
O = panel["open"]; H = panel["high"]; L = panel["low"]
R = C.pct_change()
dates = C.index
T, N = C.shape

gate_ic, gate_icir = 0.0070, 0.0840

factors = {}
factors["mom_120d_skip5"] = C.shift(5) / C.shift(125) - 1.0
for nd in (1, 2, 3, 5):
    factors[f"rev_{nd}d"] = -(C.shift(nd) / C - 1.0)
factors["rev_1d_vs"] = -(C.shift(1) / C - 1.0) * np.sign(C - C.shift(1))
for nd in (1, 2, 3, 5):
    hi = C.rolling(nd).max(); lo = C.rolling(nd).min()
    factors[f"nclv_{nd}d"] = (C - lo) / (hi - lo) - 0.5
factors["id_rev_1d"] = -(C / O - 1.0)
rng = (H - L).replace(0, np.nan)
factors["nbody_1d"] = (C - L) / rng - 0.5
factors["vol_of_vol20x60"] = R.rolling(20).std().rolling(60).std()
M = panel["macro"]
vix = M["VIX"].reindex(C.index).ffill()
vix_ret = vix.pct_change()
cov60 = R.rolling(60).cov(vix_ret)
var60 = vix_ret.rolling(60).var()
beta_vix = cov60.div(var60, axis=0)
cond = (vix > vix.rolling(20).mean()).astype(float)
factors["vix_beta_cond_60x20"] = beta_vix * cond

fmat = {k: v.values for k, v in factors.items()}
fwd = {h: (C.shift(-h) / C - 1.0).values for h in (1, 2, 3, 5)}


def ic_series(farr, fwdarr):
    out = []
    for t in range(T):
        a, b = farr[t], fwdarr[t]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 8:
            continue
        ra = rankdata(a[m]); rb = rankdata(b[m])
        ic = np.corrcoef(ra, rb)[0, 1]
        if np.isfinite(ic):
            out.append((dates[t], ic))
    return pd.Series({d: v for d, v in out})


print(f"{'factor':22s} {'h1 ic':>9s} {'icir':>7s} {'hit':>6s} {'n':>5s} | {'365d ic':>9s} {'icir':>7s} {'n':>5s} | {'120d ic':>9s} {'icir':>7s} {'n':>5s} | gate")
out = {}
for name, farr in fmat.items():
    row = {}
    for h in (1, 2, 3, 5):
        s = ic_series(farr, fwd[h])
        row[h] = dict(n=len(s), ic=float(s.mean()) if len(s) else np.nan,
                      icir=float(s.mean() / s.std()) if len(s) > 1 else np.nan,
                      hit=float((s > 0).mean()) if len(s) else np.nan,
                      last=(s.index.max().date() if len(s) else None))
    out[name] = row
    ic1 = row[1]
    n365 = n120 = 0; ic365 = icir365 = ic120 = icir120 = np.nan
    if ic1['n']:
        s = ic_series(fmat[name], fwd[1])  # already have row1 but need series for windows; recompute cheap
        last365 = s[s.index >= s.index.max() - pd.Timedelta(days=365)]
        last120 = s[s.index >= s.index.max() - pd.Timedelta(days=120)]
        n365 = len(last365); n120 = len(last120)
        ic365 = last365.mean() if n365 else np.nan
        icir365 = last365.mean() / last365.std() if n365 > 1 else np.nan
        ic120 = last120.mean() if n120 else np.nan
        icir120 = last120.mean() / last120.std() if n120 > 1 else np.nan
    ok = (abs(ic1['ic']) >= gate_ic) and (abs(ic1['icir']) >= gate_icir) and ic1['n'] > 30
    print(f"{name:22s} {ic1['ic']:+9.5f} {ic1['icir']:+7.4f} {ic1['hit']:6.3f} {ic1['n']:5d} | "
          f"{ic365:+9.5f} {icir365:+7.4f} {n365:5d} | {ic120:+9.5f} {icir120:+7.4f} {n120:5d} | {ok}")

json.dump(out, open("scripts/miner3_20290427_revalidate_results.json", "w"), indent=1, default=str)
print("\nsaved scripts/miner3_20290427_revalidate_results.json")
