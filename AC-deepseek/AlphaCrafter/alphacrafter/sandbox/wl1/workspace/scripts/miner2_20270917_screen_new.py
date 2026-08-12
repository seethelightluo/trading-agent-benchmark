"""miner_2: comprehensive new-factor screen on panel through 2027-09-16.
Candidates: prior-cycle set (rerun) + new ideas motivated by memory feedback
(momentum whipsaw guard, volume-confirmed reversal, defensive/crash-risk, vol dynamics).
Metrics: full-sample (2021-01-01..) and recent-400d daily rank IC / ICIR vs 1d fwd return.
Gate: |IC1| >= 0.0070 and |ICIR1| >= 0.0840. Also max abs corr vs library signals.
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

# ---------- library signals (existing effective factors) for correlation audit ----------
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

# ---------- candidate factors ----------
rv5 = realized_vol(C, 5)
rv20 = realized_vol(C, 20)
rv60 = realized_vol(C, 60)
sma5 = C.rolling(5).mean()
sma20 = C.rolling(20).mean()
sma50 = C.rolling(50).mean()
ema12 = C.ewm(span=12, adjust=False).mean()
ema26 = C.ewm(span=26, adjust=False).mean()
atr14 = (H - L).rolling(14).mean()
mom20 = np.log(C) - np.log(C.shift(20))
mom60 = np.log(C) - np.log(C.shift(60))
rev1 = -(np.log(C) - np.log(C.shift(1)))
rev3 = -(np.log(C) - np.log(C.shift(3)))
ret1 = C.pct_change()
ret5 = C.pct_change(5)

CAND = {}
# -- prior cycle set --
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
CAND["amihud_20d"] = -(np.abs(R) / V).rolling(20).mean()
CAND["hilo_range_norm"] = (H.rolling(5).max() - L.rolling(5).min()) / C
# -- new: overnight / intraday decomposition --
overnight = O / C.shift(1) - 1.0
intraday = C / O - 1.0
CAND["overnight_rev_5d"] = -overnight.rolling(5).sum()
CAND["intraday_rev_5d"] = -intraday.rolling(5).sum()
# -- new: volume-confirmed reversal & volume momentum --
vma20 = V.rolling(20).mean()
CAND["vol_surge_rev1"] = rev1 * (V / vma20 > 1.2)
obv = (np.sign(R) * V).cumsum()
CAND["obv_slope_20d"] = (obv - obv.shift(20)) / (V.rolling(20).std() + 1e-9)
# -- new: drawdown / extension --
CAND["dd_depth_60d"] = C / C.rolling(60).max() - 1.0
CAND["dist_20d_high"] = C / H.rolling(20).max() - 1.0
# -- new: crash risk / skew --
CAND["skew_20d"] = -R.rolling(20).skew()
# -- new: defensive beta to SPX --
spx_ret = R["SPX"]
beta_spx = R.rolling(60).cov(spx_ret) / spx_ret.rolling(60).var()
CAND["neg_beta_spx_60d"] = -beta_spx
# -- new: short RSI (fast reversal) --
up2 = delta.clip(lower=0).rolling(2).mean()
dn2 = (-delta.clip(upper=0)).rolling(2).mean()
CAND["rsi2"] = -(100 - 100 / (1 + up2 / dn2))
# -- new: vol-of-vol short horizon --
CAND["vol_of_vol_5x20"] = rv5 / rv20
# -- new: range squeeze --
rng = (H - L) / C
CAND["range_squeeze_10d"] = rng.rolling(10).mean() / rng.rolling(60).mean()
# -- new: VIX-regime conditional reversal --
vix_hi = (vix > vix.rolling(120).median()).to_numpy()[:, None]
CAND["vix_cond_rev1"] = pd.DataFrame(np.where(vix_hi, rev1.to_numpy(), 0.0),
                                     index=R.index, columns=C.columns)
# -- new: momentum gated by trend alignment (whipsaw guard) --
CAND["mom120_gated_ma20"] = (C / C.shift(120) - 1.0) * (C > sma20)
CAND["sma5_20_spread"] = sma5 / sma20 - 1.0
# -- new: momentum skip recent (avoid reversal contamination) --
CAND["mom20_skip10"] = np.log(C.shift(10)) - np.log(C.shift(20))
# -- new: cross-sectional z-scored reversal --
cs_std = rev1.std(axis=1)
CAND["rev1_cs_z"] = rev1.sub(rev1.mean(axis=1), axis=0).div(cs_std, axis=0)

# ---------- fast vectorized daily rank IC ----------
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

print(f"{'candidate':<22} {'IC1':>7} {'ICIR1':>7} {'hit':>6} {'N':>5} | {'recIC1':>7} {'recICIR':>7} {'recN':>5} | {'maxLibCorr':>9}")
rows = []
for name, F in CAND.items():
    Ff = F.loc[F.index >= FULL0]
    s = rank_ic_series(Ff, fwd1.loc[Ff.index])
    ic, icir, hit, n = metrics(s)
    rec = s[s.index >= s.index[-1] - pd.Timedelta(days=400)]
    ric, ricir, rhit, rn = metrics(rec)
    fflat = F.stack().dropna()
    maxc = 0.0
    for k, lf in lib_flat.items():
        j = fflat.index.intersection(lf.index)
        if len(j) > 200:
            c = np.corrcoef(fflat.loc[j].values, lf.loc[j].values)[0, 1]
            maxc = max(maxc, abs(c))
    print(f"{name:<22} {ic:>7.4f} {icir:>7.3f} {hit:>6.3f} {n:>5d} | {ric:>7.4f} {ricir:>7.3f} {rn:>5d} | {maxc:>9.3f}")
    rows.append(dict(name=name, ic=ic, icir=icir, hit=hit, n=n,
                     ric=ric, ricir=ricir, rn=rn, maxlib=maxc))

with open("scripts/miner2_20270917_screen_results.json", "w") as fh:
    json.dump(rows, fh, indent=2, default=float)
print("\nsaved scripts/miner2_20270917_screen_results.json")
