"""miner_2: re-validate all existing effective factors through 2028-06-22.
Panel cache rebuilt (close through 2028-06-22). Same methodology as prior runs.
Admission gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 (1d horizon, full window 2021-01-01+).
Also reports recent-400d IC/ICIR and max |corr| vs the rest of the library.
"""
import pandas as pd
import numpy as np
import json
import pickle

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C, O, H, L, V = panel["close"], panel["open"], panel["high"], panel["low"], panel["vol"]
M = panel["macro"]
R = C.pct_change()

def realized_vol(s, w):
    return s.pct_change().rolling(w).std()

LIB = {}
LIB["mom_120d_skip5"] = (C / C.shift(120).shift(5) - 1.0)
LIB["vol_of_vol20x60"] = realized_vol(C, 20) / realized_vol(C, 60)
LIB["nclv_1d"] = -(C - L) / (H - L)
LIB["nclv_2d"] = -(C - L.rolling(2).min()) / (H.rolling(2).max() - L.rolling(2).min())
LIB["nclv_3d"] = -(C - L.rolling(3).min()) / (H.rolling(3).max() - L.rolling(3).min())
LIB["nclv_5d"] = -(C - L.rolling(5).min()) / (H.rolling(5).max() - L.rolling(5).min())
LIB["rev_1d"] = -(np.log(C) - np.log(C.shift(1)))
LIB["rev_2d"] = -(np.log(C) - np.log(C.shift(2)))
LIB["rev_3d"] = -(np.log(C) - np.log(C.shift(3)))
LIB["rev_5d"] = -(np.log(C) - np.log(C.shift(5)))
LIB["rev_1d_vs"] = -(np.log(C) - np.log(C.shift(1))) * (V / V.rolling(20).mean())
LIB["id_rev_1d"] = -(C / O - 1.0)
LIB["nbody_1d"] = -(C - O) / (H - L)
vix = M["VIX"]
vix_ret = vix.pct_change()
vix_var = vix_ret.rolling(60).var()
vix_beta = pd.DataFrame(index=R.index, columns=C.columns, dtype=float)
for col in C.columns:
    vix_beta[col] = R[col].rolling(60).cov(vix_ret) / vix_var
cond = (vix.pct_change(20) > 0).to_numpy()[:, None]
LIB["vix_beta_cond_60x20"] = pd.DataFrame(np.where(cond, vix_beta.to_numpy(), 0.0),
                                          index=R.index, columns=C.columns)

def rank_ic_series(F, fwd, min_valid=8):
    rF = F.rank(axis=1)
    rR = fwd.rank(axis=1)
    mask = F.notna() & fwd.notna()
    n = mask.sum(axis=1)
    def zscore(df, m):
        out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        out[m] = df[m]
        mu = out.mean(axis=1)
        sd = out.std(axis=1)
        return (out.sub(mu, axis=0)).div(sd, axis=0)
    zF = zscore(rF, mask)
    zR = zscore(rR, mask)
    ic = (zF * zR).sum(axis=1) / (n - 1).clip(lower=1)
    ic = ic.where(n >= min_valid)
    return ic.dropna()

def fwd_ret(h):
    return C.shift(-h) / C - 1.0

FULL0 = pd.Timestamp("2021-01-01")

def metrics(s):
    if len(s) < 50:
        return np.nan, np.nan, np.nan, len(s)
    return s.mean(), (s.mean() / s.std() if s.std() > 0 else 0.0), (s > 0).mean(), len(s)

def turnover_10d(F):
    r = F.rank(axis=1)
    d = r.diff(10).abs().mean(axis=1)
    return d.median()

lib_flat = {k: v.stack().dropna() for k, v in LIB.items()}

print(f"{'factor':<22} {'IC1':>7} {'ICIR1':>7} {'hit':>6} {'N':>5} | {'recIC':>7} {'recICIR':>7} | {'maxLib':>6} | {'cov':>5} {'turn10':>6}")
rows = {}
for name, F in LIB.items():
    Ff = F.loc[F.index >= FULL0]
    s = rank_ic_series(Ff, fwd_ret(1).loc[Ff.index])
    ic, icir, hit, n = metrics(s)
    rec = s[s.index >= s.index[-1] - pd.Timedelta(days=400)]
    ric, ricir, rhit, rn = metrics(rec)
    fflat = F.stack().dropna()
    maxc = 0.0
    for k, lf in lib_flat.items():
        if k == name:
            continue
        j = fflat.index.intersection(lf.index)
        if len(j) > 200:
            c = np.corrcoef(fflat.loc[j].values, lf.loc[j].values)[0, 1]
            maxc = max(maxc, abs(c))
    cov = Ff.notna().mean().mean()
    t10 = turnover_10d(Ff)
    flag = "  <-- PASS" if (abs(ic) >= 0.007 and abs(icir) >= 0.084) else ""
    print(f"{name:<22} {ic:>7.4f} {icir:>7.3f} {hit:>6.3f} {n:>5d} | {ric:>7.4f} {ricir:>7.3f} | {maxc:>6.3f} | {cov:>5.2f} {t10:>6.3f}{flag}")
    rows[name] = dict(ic=ic, icir=icir, hit=hit, n=n, ric=ric, ricir=ricir, rn=rn,
                      maxlib=maxc, coverage=cov, turnover10=t10,
                      last_date=str(s.index[-1].date()))

with open("scripts/miner2_20280623_reval_results.json", "w") as fh:
    json.dump(rows, fh, indent=2, default=float)
print("\nsaved scripts/miner2_20280623_reval_results.json")
