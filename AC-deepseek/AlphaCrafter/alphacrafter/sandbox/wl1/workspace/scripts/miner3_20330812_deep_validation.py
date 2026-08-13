"""miner_3 deep validation of top candidates + library correlation check (v2: load OHLC properly).
Focus: rsi14, vol_adj_rev_20d, drawdown_60d, dxy_beta_cond_60x20.
Also recompute library factors to estimate max_abs_library_correlation.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

CUR_DATE = "2033-08-12"
WATCH = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225","NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]

def load(sym):
    p = f"../persistent/stock_data/{sym}.csv"
    if not os.path.exists(p):
        p = f"../persistent/index_data/{sym}.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[df.index <= CUR_DATE]

frames = {s: load(s) for s in WATCH}
px = pd.DataFrame({s: frames[s]["close"] for s in WATCH}).sort_index()
ret = px.pct_change()
hi = pd.DataFrame({s: frames[s]["high"] for s in WATCH}).sort_index()
lo = pd.DataFrame({s: frames[s]["low"] for s in WATCH}).sort_index()
op = pd.DataFrame({s: frames[s]["open"] for s in WATCH}).sort_index()
mac = pd.DataFrame({m: load(m)["close"] for m in ["VIX","DXY"]}).sort_index()
dxy_ret = mac["DXY"].pct_change()

# ---------- library factors (recompute from price data for correlation) ----------
def lib_factors():
    facs = {}
    facs["id_rev_1d"] = -(px.pct_change(1))
    facs["nclv_1d"] = -(px - lo.rolling(1).min()) / (hi.rolling(1).max() - lo.rolling(1).min())
    facs["nclv_5d"] = -(px - lo.rolling(5).min()) / (hi.rolling(5).max() - lo.rolling(5).min())
    facs["rev_1d"] = -np.log(px / px.shift(1))
    facs["rev_2d"] = -np.log(px / px.shift(2))
    facs["rev_5d"] = -np.log(px / px.shift(5))
    facs["rev_1d_vs"] = -np.log(px / px.shift(1)) / ret.rolling(20).std()
    facs["mom_120d_skip5"] = px.shift(5) / px.shift(125) - 1.0
    facs["vol_of_vol20x60"] = ret.rolling(20).std().rolling(60).std()
    vix_ret = mac["VIX"].pct_change()
    beta60 = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for i in range(60, len(ret)):
        a = ret.iloc[i-60:i]; b = vix_ret.iloc[i-60:i]
        m = a.notna() & b.notna()
        if m.sum().sum() < 20: continue
        cov = a[m].cov(b[m]); var = b[m].var()
        beta60.iloc[i] = cov / var if var > 0 else np.nan
    facs["vix_beta_cond_60x20"] = -beta60 * (mac["VIX"] / mac["VIX"].shift(20) - 1.0)
    return facs

# ---------- candidate factors ----------
def cand_factors():
    delta = px.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    vol20 = ret.rolling(20).std()
    beta60 = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for i in range(60, len(ret)):
        a = ret.iloc[i-60:i]; b = dxy_ret.iloc[i-60:i]
        m = a.notna() & b.notna()
        if m.sum().sum() < 20: continue
        cov = a[m].cov(b[m]); var = b[m].var()
        beta60.iloc[i] = cov / var if var > 0 else np.nan
    dxy_trend = dxy_ret.rolling(20).mean()
    return {
        "rsi14": rsi,
        "vol_adj_rev_20d": -(px.pct_change(20) / vol20),
        "drawdown_60d": px / px.rolling(60).max() - 1.0,
        "dxy_beta_cond_60x20": beta60 * np.sign(dxy_trend).values[:, None],
    }

def ic_series(fac, fwd, start=None, end=None):
    fwd_ret = px.pct_change(fwd).shift(-fwd)
    out = {}
    for dt in fac.index:
        if start and dt < pd.Timestamp(start): continue
        if end and dt > pd.Timestamp(end): continue
        f = fac.loc[dt]; r = fwd_ret.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() >= 8:
            rho, _ = spearmanr(f[m], r[m])
            out[dt] = rho
    return pd.Series(out)

def summarize(name, fac):
    print(f"\n=== {name} ===")
    for h in [1, 5, 10]:
        for label, (s, e) in {"full": (None, None), "2020-2026": ("2020-01-01","2026-07-15"),
                              "2026-2033": ("2026-07-16", None)}.items():
            s_ic = ic_series(fac, h, s, e)
            if len(s_ic) < 30: continue
            icm = s_ic.mean(); icir = icm / s_ic.std() if s_ic.std() > 0 else np.nan
            print(f"  h={h} {label:10s}: n={len(s_ic):5d} IC={icm:+.4f} ICIR={icir:+.3f}")
    rank = fac.rank(axis=1, pct=True)
    to = rank.diff().abs().mean().mean()
    cov = fac.notna().sum().sum() / (fac.shape[0] * fac.shape[1])
    print(f"  turnover(rank daily)={to:.4f} coverage={cov:.3f}")

cands = cand_factors()
libs = lib_factors()

for name, fac in cands.items():
    summarize(name, fac)

print("\n=== max_abs_library_correlation (pooled values) ===")
pool = {name: pd.Series(fac.values.ravel()).dropna() for name, fac in cands.items()}
lib_pool = {k: pd.Series(v.values.ravel()).dropna() for k, v in libs.items()}
for name, cp in pool.items():
    best = 0.0; bestk = None
    for k, lp in lib_pool.items():
        m = cp.index.intersection(lp.index)
        if len(m) < 1000: continue
        r = np.corrcoef(cp[m], lp[m])[0, 1]
        if abs(r) > abs(best):
            best, bestk = r, k
    print(f"  {name}: max_abs_corr={abs(best):.3f} with {bestk} (r={best:+.3f})")
