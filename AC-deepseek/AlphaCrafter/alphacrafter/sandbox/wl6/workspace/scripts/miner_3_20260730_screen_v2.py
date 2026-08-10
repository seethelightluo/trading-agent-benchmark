"""miner_3 screen v2: per-asset factor computation on each asset's own calendar,
then align to union panel for cross-sectional rank IC (>=8 valid instruments).
Data truncated at 2026-07-30. Horizon-10 IC preview with gate check."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_utils import load_close, load_panel, forward_returns, rank_ic_series, summarize_ic, DATA_DIR, INDEX_DIR, TRADABLES, OBSERVABLES

px, vol = load_panel()
fwd10 = forward_returns(px, 10)

def macro_close(name):
    df = load_close(name, INDEX_DIR)
    return df["close"].astype(float)

def per_asset(func, symbols=None):
    """Apply func(close_df) per asset on its own calendar; return aligned DataFrame."""
    symbols = symbols or TRADABLES
    out = {}
    for s in symbols:
        df = load_close(s)
        out[s] = func(df)
    return pd.DataFrame(out).reindex(px.index)

def beta_of(asset_ret, mkt_ret, win):
    a = asset_ret.rolling(win).cov(mkt_ret)
    b = mkt_ret.rolling(win).var()
    return a / b

# ---------- macro data per own calendar ----------
mac = {m: macro_close(m) for m in OBSERVABLES}
macr = {m: mac[m].pct_change() for m in OBSERVABLES}
mac20 = {m: mac[m].pct_change(20) for m in OBSERVABLES}
pxr = {s: px[s].pct_change() for s in TRADABLES}  # aligned panel returns (for beta vs macro)

def beta_vs_macro(win, m):
    mr = macr[m]
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        r = df["close"].pct_change()
        # align macro to asset calendar
        mrs = mr.reindex(df.index).ffill()
        out[s] = beta_of(r, mrs, win)
    return pd.DataFrame(out).reindex(px.index)

def cond_beta(win, m, mret):
    mr = macr[m]
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        r = df["close"].pct_change()
        mrs = mr.reindex(df.index).ffill()
        b = beta_of(r, mrs, win)
        out[s] = -b * mret.reindex(df.index).ffill()
    return pd.DataFrame(out).reindex(px.index)

def vol_trend(a, b):
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        r = df["close"].pct_change()
        v = r.rolling(20).std()
        out[s] = v.rolling(a).mean() / v.rolling(b).mean()
    return pd.DataFrame(out).reindex(px.index)

def mom_skip(lookback, skip):
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        c = df["close"]
        out[s] = c.shift(skip) / c.shift(skip + lookback) - 1.0
    return pd.DataFrame(out).reindex(px.index)

def risk_adj_mom(lookback):
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        c = df["close"]
        r = c.pct_change()
        out[s] = r.rolling(lookback).sum() / r.rolling(lookback).std()
    return pd.DataFrame(out).reindex(px.index)

def vol_of_vol(vwin, owin):
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        r = df["close"].pct_change()
        out[s] = r.rolling(vwin).std().rolling(owin).std()
    return pd.DataFrame(out).reindex(px.index)

def clv(win):
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        out[s] = ((df["close"] - df["low"]) / rng).rolling(win).mean()
    return pd.DataFrame(out).reindex(px.index)

def range_ratio(a, b):
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        rng = (df["high"] - df["low"])
        out[s] = rng.rolling(a).mean() / rng.rolling(b).mean()
    return pd.DataFrame(out).reindex(px.index)

def mdd(win):
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        c = df["close"]
        roll_max = c.rolling(win, min_periods=win // 2).max()
        out[s] = c / roll_max - 1.0
    return pd.DataFrame(out).reindex(px.index)

def dist_high(lookback):
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        c = df["close"]
        out[s] = c / c.rolling(lookback, min_periods=lookback // 2).max() - 1.0
    return pd.DataFrame(out).reindex(px.index)

def skew_60():
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        out[s] = df["close"].pct_change().rolling(60).skew()
    return pd.DataFrame(out).reindex(px.index)

def kurt_60():
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        out[s] = df["close"].pct_change().rolling(60).kurt()
    return pd.DataFrame(out).reindex(px.index)

def rel_strength(win, base="SPX"):
    out = {}
    b = load_close(base)["close"]
    for s in TRADABLES:
        df = load_close(s)
        c = df["close"]
        rel = c / b.reindex(df.index).ffill()
        out[s] = rel.pct_change(win)
    return pd.DataFrame(out).reindex(px.index)

def sma_dist(win):
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        c = df["close"]
        out[s] = c / c.rolling(win).mean() - 1.0
    return pd.DataFrame(out).reindex(px.index)

def ac1_10():
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        r = df["close"].pct_change()
        out[s] = r.rolling(10).apply(lambda x: pd.Series(x).autocorr(1) if len(x) >= 6 else np.nan, raw=False)
    return pd.DataFrame(out).reindex(px.index)

def amihud(win):
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        r = df["close"].pct_change().abs()
        v = df["volume"].replace(0, np.nan) if "volume" in df else np.nan
        out[s] = (r / v).rolling(win).mean()
    return pd.DataFrame(out).reindex(px.index)

candidates = {
    "dxy_beta_60d": beta_vs_macro(60, "DXY"),
    "dxy_beta_120d": beta_vs_macro(120, "DXY"),
    "usdjpy_beta_60d": beta_vs_macro(60, "USDJPY"),
    "usdcny_beta_60d": beta_vs_macro(60, "USDCNY"),
    "eurusd_beta_60d": beta_vs_macro(60, "EURUSD"),
    "vix_beta_60d": beta_vs_macro(60, "VIX"),
    "us10y_beta_60d": beta_vs_macro(60, "US10Y"),
    "dxy_cond_60x20": cond_beta(60, "DXY", mac20["DXY"]),
    "vix_cond_60x20": cond_beta(60, "VIX", mac20["VIX"]),
    "usdjpy_cond_60x20": cond_beta(60, "USDJPY", mac20["USDJPY"]),
    "clv_20d": clv(20),
    "vol_trend_20x60": vol_trend(20, 60),
    "range_ratio_20x60": range_ratio(20, 60),
    "mdd_60d": mdd(60),
    "dist_high_120d": dist_high(120),
    "skew_60d": skew_60(),
    "kurt_60d": kurt_60(),
    "risk_adj_mom_20d": risk_adj_mom(20),
    "mom_30d_skip5": mom_skip(30, 5),
    "mom_60d_skip20": mom_skip(60, 20),
    "mom_90d_skip5": mom_skip(90, 5),
    "rel_strength_60d": rel_strength(60),
    "rel_strength_120d": rel_strength(120),
    "sma_dist_50d": sma_dist(50),
    "sma_dist_200d": sma_dist(200),
    "ac1_10d": ac1_10(),
    "amihud_20d": amihud(20),
    "vol_of_vol20x60": vol_of_vol(20, 60),
    "mom_20d_skip5": mom_skip(20, 5),
    "dxy_cond_120x20": cond_beta(120, "DXY", mac20["DXY"]),
    "vix_cond_120x20": cond_beta(120, "VIX", mac20["VIX"]),
}

print(f"{'factor':<24}{'ic':>8}{'icir':>8}{'hit':>7}{'n':>6}  gate")
for name, f in candidates.items():
    f = f.reindex(px.index)
    s = rank_ic_series(f, fwd10)
    res = summarize_ic(s, name, 10)
    flag = "PASS" if res["pass_gate"] else ""
    print(f"{name:<24}{res['ic']:>8.4f}{res['icir']:>8.4f}{res['ic_hit_ratio']:>7.3f}{res['n_ic_dates']:>6d}  {flag}")
