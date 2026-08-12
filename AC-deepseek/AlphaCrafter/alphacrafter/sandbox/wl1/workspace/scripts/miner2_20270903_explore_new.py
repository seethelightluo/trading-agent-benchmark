"""miner_2: explore new factor ideas (screen 12 candidates) on panel through 2027-09-02.
Metrics: full-sample and recent-252d daily rank IC/ICIR vs 1d forward return.
Also max abs correlation of flattened signal with existing library signals.
Gate: |IC1| >= 0.0070 and |ICIR1| >= 0.0840.
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

# ---------- library signals for correlation ----------
LIB = {}
LIB["mom_120d_skip5"] = (C / C.shift(120).shift(5) - 1.0)  # rough proxy: 120d momentum skip 5
LIB["vol_of_vol20x60"] = realized_vol(C, 20) / realized_vol(C, 60)
LIB["nclv_1d"] = -(C - L) / (H - L)
LIB["rev_1d"] = -(np.log(C) - np.log(C.shift(1)))
LIB["rev_2d"] = -(np.log(C) - np.log(C.shift(2)))
LIB["id_rev_1d"] = -(C / O - 1.0)
LIB["nbody_1d"] = -(C - O) / (H - L)

# vix_beta_cond proxy: rolling beta of asset return on VIX chg, conditioned on VIX rising
vix = M["VIX"]
vix_ret = vix.pct_change()
vix_beta = R.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
cond = (vix.pct_change(20) > 0).to_numpy()[:, None]
LIB["vix_beta_cond_60x20"] = pd.DataFrame(np.where(cond, vix_beta.to_numpy(), 0.0), index=R.index, columns=C.columns)

# ---------- candidates ----------
rv20 = realized_vol(C, 20)
rv60 = realized_vol(C, 60)
sma20 = C.rolling(20).mean()
sma50 = C.rolling(50).mean()
ema12 = C.ewm(span=12, adjust=False).mean()
ema26 = C.ewm(span=26, adjust=False).mean()
atr14 = (H - L).rolling(14).mean()
mom20 = np.log(C) - np.log(C.shift(20))
mom60 = np.log(C) - np.log(C.shift(60))
rev1 = -(np.log(C) - np.log(C.shift(1)))

CAND = {}
CAND["vol_scaled_mom20"] = mom20 / rv20
CAND["vol_scaled_mom60"] = mom60 / rv60
CAND["trend_consistency20"] = mom20 * (C > sma20).rolling(20).mean()
CAND["ma20_dist"] = C / sma20 - 1.0
CAND["ma50_dist"] = C / sma50 - 1.0
CAND["macd_atr14"] = (ema12 - ema26) / atr14
CAND["vol_adj_rev1"] = rev1 / rv20
delta = C.diff()
up = delta.clip(lower=0).rolling(14).mean()
dn = (-delta.clip(upper=0)).rolling(14).mean()
CAND["rsi14"] = 100 - 100 / (1 + up / dn)
CAND["vol_zscore20"] = (rv20 - rv20.rolling(120).mean()) / rv20.rolling(120).std()
CAND["upper_shadow_1d"] = -(H - np.maximum(O, C)) / (H - L)
CAND["lower_shadow_1d"] = (np.minimum(O, C) - L) / (H - L)
CAND["rel_strength20"] = mom20 - mom20.mean(axis=1)
CAND["amihud_20d"] = -(np.abs(R) / V).rolling(20).mean()  # neg illiquidity
CAND["hilo_range_norm"] = (H.rolling(5).max() - L.rolling(5).min()) / C

def rank_ic_series(factor, fwd=1):
    fwd_ret = C.shift(-fwd) / C - 1.0
    dates, ics = [], []
    for dt in factor.index:
        fv, fr = factor.loc[dt], fwd_ret.loc[dt]
        m = fv.notna() & fr.notna()
        if m.sum() >= 8:
            ics.append(fv[m].rank().corr(fr[m].rank())); dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def metrics(s):
    return s.mean(), (s.mean() / s.std() if s.std() > 0 else 0.0), (s > 0).mean(), len(s)

# flattened library signals for corr
lib_flat = {k: v.stack().dropna() for k, v in LIB.items() if v is not None}

print(f"{'candidate':<20} {'IC1':>7} {'ICIR1':>7} {'hit':>6} {'N':>5} | {'recIC1':>7} {'recICIR':>7} {'recN':>5} | {'maxLibCorr':>9}")
rows = []
for name, F in CAND.items():
    full = rank_ic_series(F)
    ic, icir, hit, n = metrics(full)
    rec = full[full.index >= full.index[-1] - pd.Timedelta(days=400)]
    ric, ricir, rhit, rn = metrics(rec)
    fflat = F.stack().dropna()
    maxc = 0.0
    for k, lf in lib_flat.items():
        j = fflat.index.intersection(lf.index)
        if len(j) > 200:
            c = np.corrcoef(fflat.loc[j].values, lf.loc[j].values)[0, 1]
            maxc = max(maxc, abs(c))
    print(f"{name:<20} {ic:>7.4f} {icir:>7.3f} {hit:>6.3f} {n:>5d} | {ric:>7.4f} {ricir:>7.3f} {rn:>5d} | {maxc:>9.3f}")
    rows.append(dict(name=name, ic=ic, icir=icir, hit=hit, n=n, ric=ric, ricir=ricir, rn=rn, maxlib=maxc))

with open("scripts/miner2_20270903_explore_results.json", "w") as fh:
    json.dump(rows, fh, indent=2, default=float)
print("\nsaved scripts/miner2_20270903_explore_results.json")
