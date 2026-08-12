"""miner_2: screen candidate new factors through 2028-06-22.
Explorations for 2028-06-23 cycle:
  A. vol-scaled momentum (mom / realized vol) - addresses momentum whipsaw
  B. trend-strength / consistency (fraction of up days, directional efficiency)
  C. 10d/20d close-location (NCLV longer horizons)
  D. RSI-style 14d mean reversion
  E. cross-asset complex momentum (commodity, equity, crypto blocs)
  F. DXY-beta conditional (USD direction regime)
  G. VIX-level regime conditional (risk-on/off asset beta)
  H. bond-equity relative momentum (US10Y-return led)
  I. volume-adjusted momentum (return * vol-ratio)
  J. rolling Sharpe (mean/std of returns over 60d)
Report full-window IC/ICIR (gate |IC|>=0.007, |ICIR|>=0.084), recent-400d, max lib corr.
"""
import pandas as pd
import numpy as np
import pickle

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C, O, H, L, V = panel["close"], panel["open"], panel["high"], panel["low"], panel["vol"]
M = panel["macro"]
R = C.pct_change()

def rv(s, w):
    return s.pct_change().rolling(w).std()

# ---- existing library (for correlation) ----
LIB = {}
LIB["mom_120d_skip5"] = (C / C.shift(120).shift(5) - 1.0)
LIB["vol_of_vol20x60"] = rv(C, 20) / rv(C, 60)
LIB["nclv_1d"] = -(C - L) / (H - L)
LIB["rev_1d"] = -(np.log(C) - np.log(C.shift(1)))
LIB["rev_2d"] = -(np.log(C) - np.log(C.shift(2)))
LIB["id_rev_1d"] = -(C / O - 1.0)
LIB["nbody_1d"] = -(C - O) / (H - L)
LIB["rev_5d"] = -(np.log(C) - np.log(C.shift(5)))
LIB["nclv_2d"] = -(C - L.rolling(2).min()) / (H.rolling(2).max() - L.rolling(2).min())
LIB["nclv_3d"] = -(C - L.rolling(3).min()) / (H.rolling(3).max() - L.rolling(3).min())
vix = M["VIX"]; vix_ret = vix.pct_change(); vix_var = vix_ret.rolling(60).var()
vix_beta = pd.DataFrame(index=R.index, columns=C.columns, dtype=float)
for col in C.columns:
    vix_beta[col] = R[col].rolling(60).cov(vix_ret) / vix_var
cond = (vix.pct_change(20) > 0).to_numpy()[:, None]
LIB["vix_beta_cond_60x20"] = pd.DataFrame(np.where(cond, vix_beta.to_numpy(), 0.0),
                                          index=R.index, columns=C.columns)

# ---- candidate factors ----
CAN = {}
mom120 = C / C.shift(120).shift(5) - 1.0
mom60 = C / C.shift(60).shift(5) - 1.0
rv20 = rv(C, 20); rv60 = rv(C, 60)

# A. vol-scaled momentum
CAN["mom120_vs20"] = mom120 / rv20
CAN["mom60_vs20"] = mom60 / rv20
# B. trend strength / consistency
CAN["trend_frac_up60"] = (R.rolling(60).apply(lambda x: (x > 0).mean(), raw=True))
CAN["trend_eff60"] = (C - C.shift(60)) / (rv60 * np.sqrt(60))  # directional efficiency
# C. longer close-location
CAN["nclv_10d"] = -(C - L.rolling(10).min()) / (H.rolling(10).max() - L.rolling(10).min())
CAN["nclv_20d"] = -(C - L.rolling(20).min()) / (H.rolling(20).max() - L.rolling(20).min())
# D. RSI 14d (negative => oversold buys)
def rsi14(x):
    d = np.diff(x)
    up = np.where(d > 0, d, 0.0).mean(); dn = np.where(d < 0, -d, 0.0).mean()
    if up + dn == 0:
        return 50.0
    rs = up / dn
    return 100 - 100 / (1 + rs)
CAN["rsi14_neg"] = -C.rolling(15).apply(lambda x: rsi14(x.values), raw=False)
# E. cross-asset complex momentum
def bloc_mom(syms, w):
    sub = C[syms].pct_change(w).mean(axis=1)
    return sub
comm = bloc_mom(["XAU", "COPPER", "WTI"], 20)
eq = bloc_mom(["SPX", "NDX", "SOX"], 20)
cr = bloc_mom(["BTC", "ETH"], 20)
CAN["comm_mom20_rel"] = comm.sub(comm.mean(axis=1), axis=0)
CAN["eq_mom20_rel"] = eq.sub(eq.mean(axis=1), axis=0)
CAN["cr_mom20_rel"] = cr.sub(cr.mean(axis=1), axis=0)
# F. DXY-beta conditional (USD up/down regime)
dxy = M["DXY"]; dxy_ret = dxy.pct_change()
dxy_var = dxy_ret.rolling(60).var()
dxy_beta = pd.DataFrame(index=R.index, columns=C.columns, dtype=float)
for col in C.columns:
    dxy_beta[col] = R[col].rolling(60).cov(dxy_ret) / dxy_var
cond_dxy = (dxy.pct_change(20) > 0).to_numpy()[:, None]
CAN["dxy_beta_cond_60x20"] = pd.DataFrame(np.where(cond_dxy, dxy_beta.to_numpy(), 0.0),
                                          index=R.index, columns=C.columns)
# G. VIX level regime conditional (VIX above median => flight-to-safety beta)
vix_lvl = vix.rolling(60).median()
cond_vlvl = (vix > vix_lvl).to_numpy()[:, None]
CAN["vix_lvl_cond_60"] = pd.DataFrame(np.where(cond_vlvl, vix_beta.to_numpy(), 0.0),
                                      index=R.index, columns=C.columns)
# H. bond-equity relative momentum (US10Y return momentum, cross-sectional)
us10y = C["US10Y"]
us10y_mom = us10y.pct_change(20)
CAN["us10y_mom_rel"] = us10y_mom.sub(us10y_mom.mean(), axis=0)
# I. volume-adjusted momentum
CAN["mom120_vs20_vol"] = mom120 / rv20 * (V / V.rolling(20).mean())
# J. rolling Sharpe 60d
CAN["sharpe60"] = (C.pct_change().rolling(60).mean()) / rv60

# ---- IC machinery ----
def rank_ic_series(F, fwd, min_valid=8):
    rF = F.rank(axis=1); rR = fwd.rank(axis=1)
    mask = F.notna() & fwd.notna()
    n = mask.sum(axis=1)
    def zscore(df, m):
        out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        out[m] = df[m]
        mu = out.mean(axis=1); sd = out.std(axis=1)
        return (out.sub(mu, axis=0)).div(sd, axis=0)
    zF = zscore(rF, mask); zR = zscore(rR, mask)
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
all_flat = {**LIB, **CAN}

print(f"{'candidate':<24} {'IC1':>7} {'ICIR1':>7} {'hit':>6} {'N':>5} | {'recIC':>7} {'recICIR':>7} | {'maxLib':>6} | {'cov':>5} {'turn10':>6}")
out = {}
for name, F in CAN.items():
    Ff = F.loc[F.index >= FULL0]
    s = rank_ic_series(Ff, fwd_ret(1).loc[Ff.index])
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
    cov = Ff.notna().mean().mean()
    t10 = turnover_10d(Ff)
    flag = "  <-- PASS" if (abs(ic) >= 0.007 and abs(icir) >= 0.084) else ""
    print(f"{name:<24} {ic:>7.4f} {icir:>7.3f} {hit:>6.3f} {n:>5d} | {ric:>7.4f} {ricir:>7.3f} | {maxc:>6.3f} | {cov:>5.2f} {t10:>6.3f}{flag}")
    out[name] = dict(ic=ic, icir=icir, hit=hit, n=n, ric=ric, ricir=ricir, rn=rn,
                     maxlib=maxc, coverage=cov, turnover10=t10,
                     last_date=str(s.index[-1].date()))

with open("scripts/miner2_20280623_screen_results.json", "w") as fh:
    json.dump(out, fh, indent=2, default=float)
print("\nsaved scripts/miner2_20280623_screen_results.json")
