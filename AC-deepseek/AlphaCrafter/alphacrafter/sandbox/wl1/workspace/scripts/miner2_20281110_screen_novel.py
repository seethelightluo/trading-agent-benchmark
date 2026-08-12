"""miner2 2028-11-10: screen novel factor candidates on the 15-name cross-asset panel.
Visible data through 2028-11-09. Daily rank IC vs 1-day forward return, ICIR, hit ratio,
coverage, turnover, year splits, and correlation vs existing library factors.
Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 (1d horizon).
"""
import pandas as pd
import numpy as np
import json
import pickle

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C, O, H, L, V = panel["close"], panel["open"], panel["high"], panel["low"], panel["vol"]
M = panel["macro"]
R = C.pct_change()

VISIBLE = "2028-11-09"
START = "2021-01-01"
C = C.loc[:VISIBLE]
O = O.loc[:VISIBLE]
H = H.loc[:VISIBLE]
L = L.loc[:VISIBLE]
V = V.loc[:VISIBLE]
M = M.loc[:VISIBLE]
R = R.loc[:VISIBLE]
print(f"panel dates: {C.index.min().date()} -> {C.index.max().date()} rows={C.shape[0]} assets={C.shape[1]}")

def er(px, w):
    d = px.diff().abs()
    path = d.rolling(w).sum()
    net = (px - px.shift(w)).abs()
    return net / path

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

FULL0 = pd.Timestamp(START)
fwd1 = fwd_ret(1)

# ---------- existing library signals ----------
LIB = {}
LIB["mom_120d_skip5"] = (C / C.shift(120).shift(5) - 1.0)
LIB["vol_of_vol20x60"] = R.rolling(20).std() / R.rolling(60).std()
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
cond = (vix.pct_change(20) > 0).reindex(R.index).fillna(False).to_numpy()[:, None]
LIB["vix_beta_cond_60x20"] = pd.DataFrame(np.where(cond, vix_beta.to_numpy(), 0.0), index=R.index, columns=C.columns)

# ---------- candidate factors ----------
cands = {}

# C1: relative (cross-sectional demeaned) 20d momentum
mom20 = C / C.shift(20) - 1.0
cands["rel_mom_20d"] = mom20.sub(mom20.mean(axis=1), axis=0)

# C2: relative 60d momentum
mom60 = C / C.shift(60) - 1.0
cands["rel_mom_60d"] = mom60.sub(mom60.mean(axis=1), axis=0)

# C3: vol-scaled 1d reversal
vol20 = R.rolling(20).std()
cands["rev_1d_voladj"] = -(np.log(C) - np.log(C.shift(1))) / vol20

# C4: trend-quality weighted 20d momentum (er * sign)
cands["mom20_tq"] = er(C, 20) * np.sign(C - C.shift(20))

# C5: momentum deceleration (20d minus 5d return)
cands["mom_decel_20x5"] = (C / C.shift(20) - 1.0) - (C / C.shift(5) - 1.0)

# C6: Bollinger z-score position (close vs 20d mean, 20d std)
cands["bbz_20d"] = (C - C.rolling(20).mean()) / (C.rolling(20).std())

# C7: short momentum 10d skip1
cands["mom_10d_skip1"] = C.shift(1) / C.shift(11) - 1.0

# C8: drawdown from 60d high (negative = below high)
cands["dd_60d"] = C / C.rolling(60).max() - 1.0

# C9: US10Y-rate beta conditional (beta to 5d rate change * rate up-trend sign)
us10y = C["US10Y"]
rate_ret = us10y.pct_change(5)
rate_beta = pd.DataFrame(index=R.index, columns=C.columns, dtype=float)
for col in C.columns:
    rate_beta[col] = R[col].rolling(60).cov(rate_ret) / rate_ret.rolling(60).var()
cond_rate = (rate_ret > 0).to_numpy()[:, None]
cands["rate_beta_cond_60x5"] = pd.DataFrame(np.where(cond_rate, rate_beta.to_numpy(), 0.0), index=R.index, columns=C.columns)

# C10: DXY-beta conditional (beta to DXY 5d change * DXY up-trend sign)
dxy = M["DXY"]
dxy_ret = dxy.pct_change(5)
dxy_beta = pd.DataFrame(index=R.index, columns=C.columns, dtype=float)
for col in C.columns:
    dxy_beta[col] = R[col].rolling(60).cov(dxy_ret) / dxy_ret.rolling(60).var()
cond_dxy = (dxy_ret > 0).to_numpy()[:, None]
cands["dxy_beta_cond_60x5"] = pd.DataFrame(np.where(cond_dxy, dxy_beta.to_numpy(), 0.0), index=R.index, columns=C.columns)

# C11: 20d realized skewness (negative skew = crash-prone)
cands["skew_20d"] = R.rolling(20).skew()

# C12: days since 60d high (log-scaled)
def days_since_high(px, w):
    roll_max = px.rolling(w).max()
    out = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for col in px.columns:
        s = px[col]
        rm = roll_max[col]
        cnt = 0
        vals = []
        for i in range(len(s)):
            if s.iloc[i] >= rm.iloc[i] or pd.isna(rm.iloc[i]):
                cnt = 0
            else:
                cnt += 1
            vals.append(cnt)
        out[col] = vals
    return -np.log1p(out)
cands["days_since_high_60d"] = days_since_high(C, 60)

# C13: overnight gap mean reversion: -(O / prevC - 1)
cands["gap_rev_1d"] = -(O / C.shift(1) - 1.0)

# C14: upside/downside volatility asymmetry (down-vol minus up-vol scaled)
up_vol = R.clip(lower=0).rolling(20).std()
dn_vol = (-R.clip(upper=0)).rolling(20).std()
cands["asym_vol_20d"] = dn_vol - up_vol

# C15: XAU-beta conditional (gold rally regime)
xau_ret = C["XAU"].pct_change(5)
xau_beta = pd.DataFrame(index=R.index, columns=C.columns, dtype=float)
for col in C.columns:
    xau_beta[col] = R[col].rolling(60).cov(xau_ret) / xau_ret.rolling(60).var()
cond_xau = (xau_ret > 0).to_numpy()[:, None]
cands["xau_beta_cond_60x5"] = pd.DataFrame(np.where(cond_xau, xau_beta.to_numpy(), 0.0), index=R.index, columns=C.columns)

# C16: 5d max drawdown (worst cumulative 5d return)
cands["maxdd_5d"] = C / C.rolling(5).max() - 1.0

lib_flat = {k: v.stack().dropna() for k, v in LIB.items()}

def turnover_10d(F):
    r = F.rank(axis=1)
    d = r.diff(10).abs().mean(axis=1)
    return d.median()

results = []
for name, F in cands.items():
    Ff = F.loc[F.index >= FULL0]
    s = rank_ic_series(Ff, fwd1.loc[Ff.index])
    s = s.dropna()
    ic = s.mean()
    icir = ic / s.std(ddof=1) if s.std(ddof=1) > 0 else 0.0
    hit = (np.sign(s) == np.sign(ic)).mean()
    rec = s[s.index >= s.index[-1] - pd.Timedelta(days=400)]
    ric = rec.mean()
    ricir = ric / rec.std(ddof=1) if rec.std(ddof=1) > 0 else 0.0
    cov = Ff.notna().mean().mean()
    t10 = turnover_10d(Ff)
    # max abs corr vs library
    fflat = F.stack().dropna()
    maxc, argmaxc = 0.0, ""
    corrs = {}
    for k, lf in lib_flat.items():
        j = fflat.index.intersection(lf.index)
        if len(j) > 200:
            c = np.corrcoef(fflat.loc[j].values, lf.loc[j].values)[0, 1]
            corrs[k] = round(float(c), 3)
            if abs(c) > maxc:
                maxc, argmaxc = abs(c), k
    yrs = {}
    for y, g in s.groupby(s.index.year):
        yrs[str(y)] = dict(ic=round(float(g.mean()), 4),
                           icir=round(float(g.mean() / g.std(ddof=1)), 3) if g.std(ddof=1) > 0 else 0.0,
                           n=int(g.shape[0]))
    results.append(dict(name=name, ic=ic, icir=icir, hit=hit, n=int(s.shape[0]),
                        ric=ric, ricir=ricir, rn=int(rec.shape[0]),
                        coverage=float(cov), turn10=float(t10),
                        max_lib_corr=maxc, argmax_corr=argmaxc, corrs=corrs, years=yrs))

print(f"\n{'name':<22} {'IC':>7} {'ICIR':>7} {'hit':>6} {'N':>5} | {'recIC':>7} {'recICIR':>7} | {'cov':>5} {'t10':>5} | {'maxLib':>6} {'argmax':>16}")
for r in sorted(results, key=lambda x: -abs(x["ic"])):
    print(f"{r['name']:<22} {r['ic']:>7.4f} {r['icir']:>7.3f} {r['hit']:>6.3f} {r['n']:>5d} | {r['ric']:>7.4f} {r['ricir']:>7.3f} | {r['coverage']:>5.2f} {r['turn10']:>5.2f} | {r['max_lib_corr']:>6.3f} {r['argmax_corr']:>16}")

print("\n--- gate check: |IC|>=0.007 & |ICIR|>=0.084 ---")
for r in sorted(results, key=lambda x: -abs(x["ic"])):
    ok = abs(r["ic"]) >= 0.007 and abs(r["icir"]) >= 0.084
    low_corr = r["max_lib_corr"] < 0.7
    print(f"{r['name']:<22} IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} {'PASS' if ok else 'fail':5s} maxLib={r['max_lib_corr']:.3f} {'lowCorr' if low_corr else 'HIGH-CORR'}")
    print("   years:", {k: f"{v['ic']:+.3f}/{v['icir']:+.2f}({v['n']})" for k, v in r["years"].items()})

with open("scripts/miner2_20281110_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=2, default=float)
print("\nsaved scripts/miner2_20281110_screen_results.json")
