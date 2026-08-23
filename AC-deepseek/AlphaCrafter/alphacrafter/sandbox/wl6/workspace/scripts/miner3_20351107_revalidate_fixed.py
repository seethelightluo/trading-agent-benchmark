"""miner_3 2035-11-07: revalidate effective library through visible 2035-11-06.
Admission gate (15-instrument universe): |IC|>=0.0070 and |ICIR|>=0.0840.
Reports same-horizon (10d) IC/ICIR, coverage, turnover, recency splits.
"""
import sys, os, math
sys.path.insert(0, 'scripts')
from factor_validation_lib import TRADABLE, rank_ic_series, align_fwd_returns
import pandas as pd, numpy as np

VIS = "2035-11-06"
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).sort_index().ffill()
px = px[px.index >= "2021-01-01"]
ret = px.pct_change()
print("panel shape:", px.shape, "assets:", px.shape[1],
      "date:", px.index.min().date(), "->", px.index.max().date(), flush=True)

vix = pd.read_csv("../persistent/index_data/VIX.csv", parse_dates=["date"])
vixs = vix[vix["date"] <= pd.Timestamp(VIS)].set_index("date")["close"].reindex(px.index).ffill()
vixr = vixs.pct_change()

hsir = px["HSI"].pct_change()
us10r = px["US10Y"].pct_change()
cn10r = px["CN10Y"].pct_change()


def beta_col(sig):
    out = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for a in px.columns:
        out[a] = ret[a].rolling(60).cov(sig) / sig.rolling(60).var()
    return out


def sig_mom10(): return px.shift(5) / px.shift(15) - 1.0


def sig_mom120(): return px.shift(5) / px.shift(125) - 1.0


def sig_vov():
    return ret.rolling(20).std().rolling(60).std()


def sig_lowvol():
    return -ret.rolling(20).std()


def sig_betavix_neg():
    return -beta_col(vixr)


def sig_vixcond():
    return -beta_col(vixr) * (vixs / vixs.shift(20) - 1.0)


def sig_dvr():
    return ret.clip(upper=0).rolling(20).std() / ret.rolling(120).std()


def sig_betacn10():
    return beta_col(cn10r)


def sig_betachi():
    return beta_col(hsir)


def sig_corr10y():
    return pd.concat([ret[a].rolling(60).corr(us10r).rename(a) for a in px.columns], axis=1)


def sig_skew():
    return -ret.rolling(20).skew()


def sig_vovchg():
    v = ret.rolling(20).std()
    return v.diff(20) / v.rolling(20).mean()


def sig_xaucop():
    cond = ((px["XAU"].pct_change(20) > 0) & (px["COPPER"].pct_change(20) > 0)).astype(float)
    return -cond


def sig_volbeta():
    spxv = ret["SPX"].rolling(20).std()
    return beta_col(spxv)


def sig_signewma():
    return (px / px.ewm(span=60).mean() - 1.0).apply(np.sign)


factors = {
    'beta_chi_60d': sig_betachi,
    'beta_cn10y_60d': sig_betacn10,
    'beta_vix_60d_neg': sig_betavix_neg,
    'corr_us10y_60d': sig_corr10y,
    'down_vol_ratio_20x120': sig_dvr,
    'low_vol_20d': sig_lowvol,
    'mom_10d_skip5': sig_mom10,
    'mom_120d_skip5': sig_mom120,
    'sign_ewma_60d': sig_signewma,
    'skew_20d_neg': sig_skew,
    'vix_beta_cond_60x20': sig_vixcond,
    'vol_beta_spx_60d': sig_volbeta,
    'vol_of_vol20x60': sig_vov,
    'vol_of_vol_chg_20d': sig_vovchg,
    'xau_copper_cond_20d': sig_xaucop,
}
print("=" * 80, flush=True)
for name in sorted(factors):
    try:
        f = factors[name]().reindex(index=px.index, columns=px.columns)
        ic = rank_ic_series(f, align_fwd_returns(px, 10))
        icm = float(ic.mean())
        icstd = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
        icir = icm / icstd if icstd and math.isfinite(icstd) and icstd > 0 else np.nan
        hit = float((ic > 0).mean())
        r3 = ic[ic.index >= "2032-11-01"]
        ric3 = float(r3.mean()) if len(r3) else np.nan
        ricir3 = ric3 / r3.std(ddof=1) if len(r3) > 2 and r3.std(ddof=1) > 0 else np.nan
        cov = float(f.notna().mean().mean())
        gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
        flag = "PASS" if gate else ("WEAK-ICIR" if abs(icir) >= 0.0840 else "fail")
        print(f"{name:20s} IC10={icm:+.4f} ICIR10={icir:+.4f} hit={hit:.3f} n={len(ic)} "
              f"recent3yIC={ric3:+.4f} ricir3={ricir3:+.4f} cov={cov:.3f} {flag}", flush=True)
    except Exception as e:
        print(f"{name} ERROR {type(e).__name__}: {e}", flush=True)