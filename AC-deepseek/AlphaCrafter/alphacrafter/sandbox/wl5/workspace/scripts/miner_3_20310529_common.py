"""Shared data-loading + validation utilities for miner_3 (2031-05-29 cycle)."""
import json
import numpy as np
import pandas as pd

VISIBLE_THROUGH = "2031-05-28"
CURRENT_DATE = "2031-05-29"

IC_THRESHOLD = 0.0070
ICIR_THRESHOLD = 0.0840

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]


def load_asset(sym):
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date").reset_index(drop=True)
    return df


def load_macro(sym):
    df = pd.read_csv(f"../persistent/index_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date").reset_index(drop=True)
    return df


def price_panel(field="close", symbols=None):
    symbols = symbols or WATCH
    out = {}
    for s in symbols:
        df = load_asset(s)
        if len(df) == 0:
            continue
        out[s] = df.set_index("date")[field]
    return pd.DataFrame(out).sort_index()


def ohlcv_panels(symbols=None):
    symbols = symbols or WATCH
    out = {k: {} for k in ["open", "high", "low", "close", "volume"]}
    for s in symbols:
        df = load_asset(s)
        if len(df) == 0:
            continue
        d = df.set_index("date")
        for k in out:
            out[k][s] = d[k]
    return {k: pd.DataFrame(v).sort_index() for k, v in out.items()}


def macro_panel(sym):
    df = load_macro(sym)
    return df.set_index("date")["close"].sort_index()


def fwd_returns(close, horizons=(1, 2, 3, 5, 10, 20)):
    return {h: close.shift(-h) / close - 1.0 for h in horizons}


def _rankdata_np(a):
    """Spearman-friendly rank transform of 1-d array (ties averaged)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        ranks = avg[inv]
    return ranks


def rank_ic_series(factor_panel, fwd_ret, min_valid=8):
    """Daily cross-sectional Spearman IC between factor and forward return (vectorized)."""
    common = factor_panel.index.intersection(fwd_ret.index)
    f = factor_panel.loc[common]
    r = fwd_ret.loc[common]
    valid = f.notna() & r.notna()
    n_valid = valid.sum(axis=1)
    dates = n_valid[n_valid >= min_valid].index
    if len(dates) == 0:
        return pd.Series(dtype=float)
    fv = f.loc[dates].values
    rv = r.loc[dates].values
    ics = []
    for i in range(len(dates)):
        a = fv[i]
        b = rv[i]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < min_valid:
            ics.append(np.nan)
            continue
        ra = _rankdata_np(a[m])
        rb = _rankdata_np(b[m])
        if np.std(ra) == 0 or np.std(rb) == 0:
            ics.append(np.nan)
            continue
        ics.append(np.corrcoef(ra, rb)[0, 1])
    return pd.Series(ics, index=dates)


def summarize_ic(ic_s, label=""):
    ic_s = ic_s.dropna()
    n = len(ic_s)
    if n == 0:
        return {"ic": np.nan, "icir": np.nan, "ic_hit_ratio": np.nan, "n_ic_dates": 0}
    ic = float(ic_s.mean())
    sd = float(ic_s.std(ddof=1))
    icir = ic / sd * np.sqrt(n) if sd > 0 else np.nan
    return {"ic": ic, "icir": icir,
            "ic_hit_ratio": float((ic_s > 0).mean()), "n_ic_dates": n}


def decay_analysis(factor_panel, close, horizons=(1, 2, 3, 5, 10, 20)):
    fwd = fwd_returns(close, horizons)
    out = {}
    for h in horizons:
        s = rank_ic_series(factor_panel, fwd[h]).dropna()
        out[h] = float(s.mean()) if len(s) else np.nan
    return out


def turnover_10d(factor_panel, window=10):
    """Mean of (rank-abs-diff between t and t-window)/2 over dates (0..1 scale)."""
    ranks = factor_panel.rank(axis=1)
    diff = (ranks - ranks.shift(window)).abs().mean(axis=1)
    return float(diff.mean() / 2.0) if len(diff.dropna()) else np.nan


def coverage_stats(factor_panel, min_valid=8):
    valid = factor_panel.notna()
    cov_ad = float(valid.mean().mean()) if factor_panel.shape[1] else np.nan
    n_ge8 = float((valid.sum(axis=1) >= min_valid).mean()) if len(valid) else np.nan
    return {"coverage_asset_days": cov_ad, "coverage_dates_ge8": n_ge8}


def regime_split(ic_s, recency_buckets=None):
    ic_s = ic_s.dropna()
    if len(ic_s) == 0:
        return {}
    buckets = recency_buckets or [("2020-2022", "2020-01-01", "2022-12-31"),
                                  ("2023-2024", "2023-01-01", "2024-12-31"),
                                  ("2025-2026", "2025-01-01", "2026-12-31"),
                                  ("2027-2028", "2027-01-01", "2028-12-31"),
                                  ("2029-2030", "2029-01-01", "2030-12-31"),
                                  ("2031", "2031-01-01", "2099-12-31"),
                                  ("last180d", "2030-11-30", "2099-12-31"),
                                  ("last90d", "2031-02-28", "2099-12-31"),
                                  ("last60d", "2031-03-30", "2099-12-31")]
    out = {}
    for lab, lo, hi in buckets:
        sub = ic_s[(ic_s.index >= pd.Timestamp(lo)) & (ic_s.index <= pd.Timestamp(hi))]
        if len(sub):
            out[lab] = {"ic": float(sub.mean()),
                        "icir": float(sub.mean() / sub.std(ddof=1)) * np.sqrt(len(sub)) if sub.std(ddof=1) > 0 else np.nan,
                        "n": len(sub)}
    return out


# ---------------- library factor implementations (for correlation audit) ----------------
def _signed_r2(arr):
    x = np.arange(len(arr), dtype=float)
    y = np.log(arr)
    if len(y) < 3 or np.std(y) == 0:
        return np.nan
    slope = np.polyfit(x, y, 1)[0]
    corr = np.corrcoef(x, y)[0, 1]
    return np.sign(slope) * corr ** 2


def lib_trend_r2(close, window=30):
    return close.rolling(window, min_periods=18).apply(_signed_r2, raw=True)


def lib_semi_down(close, window=20):
    r = close.pct_change()
    neg = r.clip(upper=0)
    return neg.rolling(window, min_periods=10).mean() / r.rolling(window, min_periods=10).std()


def lib_mom(close, lookback, skip):
    return close.shift(skip) / close.shift(skip + lookback) - 1.0


def lib_dxy_beta(close, dxy, window=60):
    ra = close.pct_change()
    rb = dxy.pct_change()
    return ra.rolling(window).cov(rb) / rb.rolling(window).var()


def lib_cny_beta(close, cny, window=60):
    ra = close.pct_change()
    rb = cny.pct_change()
    return ra.rolling(window).cov(rb) / rb.rolling(window).var()


def lib_vol_of_vol(close, w1=20, w2=60):
    rv = close.pct_change().rolling(w1).std()
    return rv.rolling(w2).std()


def lib_tuw(close, window=120):
    roll_max = close.rolling(window, min_periods=60).max()
    dd = close / roll_max - 1.0
    tuw = (dd < 0).rolling(window, min_periods=60).mean()
    return -tuw


def lib_tail_ratio(close, window=20):
    r = close.pct_change()
    q95 = r.rolling(window, min_periods=10).quantile(0.95)
    q05 = r.rolling(window, min_periods=10).quantile(0.05)
    return q95 / q05.abs()


def lib_vix_beta_cond(close, vix, window=60, cond=20):
    ra = close.pct_change()
    rb = vix.pct_change()
    beta = ra.rolling(window).cov(rb) / rb.rolling(window).var()
    return -beta * (vix / vix.shift(cond) - 1.0)


def lib_kurt(close, window=20):
    r = close.pct_change()
    return r.rolling(window, min_periods=10).kurt()


def lib_wti_beta(close, wti, window=60):
    ra = close.pct_change()
    rb = wti.pct_change()
    return ra.rolling(window).cov(rb) / rb.rolling(window).var()


def library_correlation(factor_panel, close, macro, min_overlap=60):
    """Pooled Pearson correlation of candidate vs each effective library factor."""
    libs = {
        "trend_r2_30_signed": lib_trend_r2(close),
        "semi_down_ratio_20": lib_semi_down(close),
        "mom_120d_skip5": lib_mom(close, 120, 5),
        "dxy_beta_60": lib_dxy_beta(close, macro["DXY"]),
        "cny_beta_60": lib_cny_beta(close, macro["USDCNY"]),
        "vol_of_vol20x60": lib_vol_of_vol(close),
        "mom_10d_skip5": lib_mom(close, 10, 5),
        "time_under_water_120": lib_tuw(close),
        "tail_ratio_20": lib_tail_ratio(close),
        "vix_beta_cond_60x20": lib_vix_beta_cond(close, macro["VIX"]),
        "kurt_20": lib_kurt(close),
        "WTI_BETA_60": lib_wti_beta(close, close["WTI"]),
    }
    corrs = {}
    for name, lib in libs.items():
        a = factor_panel.stack()
        b = lib.stack()
        df = pd.concat([a.rename("f"), b.rename("l")], axis=1).dropna()
        if len(df) < min_overlap:
            corrs[name] = float("nan")
            continue
        corrs[name] = float(df["f"].corr(df["l"]))
    max_abs = max([abs(v) for v in corrs.values() if np.isfinite(v)], default=0.0)
    return corrs, max_abs
