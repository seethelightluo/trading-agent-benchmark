"""miner_3 screen: evaluate several novel factor candidates on the 15-instrument cross-asset universe.
Data window: 2020-01-01 .. current sim date 2033-08-12 (no future data).
Metrics: daily IC (Spearman) vs 1d/5d/10d forward returns, ICIR, hit ratio, coverage.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, glob, json

CUR_DATE = "2033-08-12"
WATCH = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225","NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]

def load(sym):
    p = f"../persistent/stock_data/{sym}.csv"
    if not os.path.exists(p):
        p = f"../persistent/index_data/{sym}.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[df.index <= CUR_DATE]
    return df

closes = {}
for s in WATCH:
    closes[s] = load(s)["close"]
px = pd.DataFrame(closes).sort_index()
px = px.dropna(how="all")
ret = px.pct_change()
print("price panel shape:", px.shape, px.index.min().date(), "->", px.index.max().date())

# macro
macro = {}
for m in ["VIX","DXY","USDCNY","USDJPY","EURUSD"]:
    macro[m] = load(m)["close"]
mac = pd.DataFrame(macro).sort_index()

# ---------------- candidate factors ----------------
# each returns a DataFrame (dates x assets) aligned to px.index

def factor_rev_20d():
    return -(px.pct_change(20))

def factor_rsi14():
    delta = px.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def factor_trend_consistency_20d():
    return (ret > 0).rolling(20).mean()

def factor_dxy_beta_cond_60x20():
    dxy_ret = mac["DXY"].pct_change()
    # beta of asset daily ret to DXY ret over 60d
    betas = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    asset_ret = ret
    for i in range(60, len(asset_ret)):
        a = asset_ret.iloc[i-60:i]
        b = dxy_ret.iloc[i-60:i]
        mask = a.notna() & b.notna()
        if mask.sum().sum() < 10:
            continue
        cov = a[mask].cov(b[mask]) if hasattr(a[mask], 'cov') else np.cov(a[mask].values, b[mask].values)[0,1]
        var = b[mask].var()
        betas.iloc[i] = cov / var if var > 0 else np.nan
    # condition on DXY 20d trend
    dxy_trend = dxy_ret.rolling(20).mean()
    return betas * np.sign(dxy_trend).values[:, None]

def factor_vol_adj_rev_20d():
    vol20 = ret.rolling(20).std()
    return -(px.pct_change(20) / vol20)

def factor_drawdown_60d():
    return px / px.rolling(60).max() - 1.0

cands = {
    "rev_20d": factor_rev_20d,
    "rsi14": factor_rsi14,
    "trend_consistency_20d": factor_trend_consistency_20d,
    "dxy_beta_cond_60x20": factor_dxy_beta_cond_60x20,
    "vol_adj_rev_20d": factor_vol_adj_rev_20d,
    "drawdown_60d": factor_drawdown_60d,
}

def ic_series(fac, fwd):
    out = {}
    fwd_ret = px.pct_change(fwd).shift(-fwd)
    for dt in fac.index:
        f = fac.loc[dt]
        r = fwd_ret.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() >= 8:
            rho, _ = spearmanr(f[m], r[m])
            out[dt] = rho
    s = pd.Series(out)
    return s

print("\n=== CANDIDATE SCREEN (full sample 2020..2033-08-12) ===")
for name, fn in cands.items():
    try:
        fac = fn()
        fac = fac.reindex(px.index)
        print(f"\n--- {name} ---")
        for h in [1, 5, 10]:
            s = ic_series(fac, h)
            if len(s) < 30:
                print(f"  h={h}: too few IC dates ({len(s)})")
                continue
            icm = s.mean()
            icir = icm / s.std() if s.std() > 0 else np.nan
            hit = (np.sign(s) == np.sign(icm)).mean()
            cov = fac.notna().sum().sum() / (fac.shape[0] * fac.shape[1])
            print(f"  h={h}: n={len(s)} IC={icm:+.4f} ICIR={icir:+.3f} hit={hit:.3f} cov={cov:.3f}")
    except Exception as e:
        print(f"  ERROR {name}: {e}")
