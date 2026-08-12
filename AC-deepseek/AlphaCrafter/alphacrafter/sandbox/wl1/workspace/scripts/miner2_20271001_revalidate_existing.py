"""miner_2: re-validate existing effective factors on panel through 2027-09-30.
Current date 2027-10-01. Full-sample + recent-400d daily rank IC / ICIR vs 1d fwd return.
Gate: |IC1| >= 0.0070 and |ICIR1| >= 0.0840 (full-sample admission; recent for drift).
"""
import pandas as pd
import numpy as np
import pickle, json

with open("scripts/panel_cache.pkl", "rb") as fh:
    P = pickle.load(fh)
C, O, H, L, V, R = P["close"], P["open"], P["high"], P["low"], P["vol"], P["ret"]
M = P["macro"]

def realized_vol(px, w):
    return px.pct_change().rolling(w).std()

# ---------- library signals (existing effective factors) ----------
LIB = {}
LIB["mom_120d_skip5"] = (C / C.shift(120).shift(5) - 1.0)
LIB["vol_of_vol20x60"] = realized_vol(C, 20) / realized_vol(C, 60)
LIB["nclv_1d"] = -(C - L) / (H - L)
LIB["rev_1d"] = -(np.log(C) - np.log(C.shift(1)))
LIB["rev_2d"] = -(np.log(C) - np.log(C.shift(2)))
LIB["id_rev_1d"] = -(C / O - 1.0)
LIB["nbody_1d"] = -(C - O) / (H - L)
vix = M["VIX"]
vix_ret = vix.pct_change()
vix_beta = R.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
cond = (vix.pct_change(20) > 0).to_numpy()[:, None]
LIB["vix_beta_cond_60x20"] = pd.DataFrame(np.where(cond, vix_beta.to_numpy(), 0.0),
                                          index=R.index, columns=C.columns)

# ---------- vectorized daily rank IC ----------
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

fwd1 = C.shift(-1) / C - 1.0
FULL0 = pd.Timestamp("2021-01-01")

def metrics(s):
    if len(s) < 50:
        return np.nan, np.nan, np.nan, len(s)
    return s.mean(), (s.mean() / s.std() if s.std() > 0 else 0.0), (s > 0).mean(), len(s)

lib_flat = {k: v.stack().dropna() for k, v in LIB.items()}

print(f"{'factor':<26} {'IC1':>7} {'ICIR1':>7} {'hit':>6} {'N':>5} | {'recIC1':>7} {'recICIR':>7} {'recN':>5} | {'maxLibCorr':>9}")
rows = {}
for name, F in LIB.items():
    Ff = F.loc[F.index >= FULL0]
    s = rank_ic_series(Ff, fwd1.loc[Ff.index])
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
    print(f"{name:<26} {ic:>7.4f} {icir:>7.3f} {hit:>6.3f} {n:>5d} | {ric:>7.4f} {ricir:>7.3f} {rn:>5d} | {maxc:>9.3f}")
    rows[name] = dict(ic=ic, icir=icir, hit=hit, n=n, ric=ric, ricir=ricir, rn=rn, maxlib=maxc)

with open("scripts/miner2_20271001_reval_results.json", "w") as fh:
    json.dump(rows, fh, indent=2, default=float)
print("\nsaved scripts/miner2_20271001_reval_results.json")
