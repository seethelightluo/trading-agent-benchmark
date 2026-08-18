"""Shared data-loading + validation utilities for miner_3 (2031-03-20 cycle)."""
import json
import numpy as np
import pandas as pd

VISIBLE_THROUGH = "2031-03-19"
CURRENT_DATE = "2031-03-20"

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
    F = factor_panel.loc[common].values
    R = fwd_ret.loc[common].values
    dates = common
    out_dates, ics = [], []
    for i in range(len(F)):
        frow = F[i]
        rrow = R[i]
        mask = ~(np.isnan(frow) | np.isnan(rrow))
        n = int(mask.sum())
        if n < min_valid:
            continue
        fa = frow[mask]
        ra = rrow[mask]
        if np.std(fa) == 0 or np.std(ra) == 0:
            continue
        ic = np.corrcoef(_rankdata_np(fa), _rankdata_np(ra))[0, 1]
        if np.isfinite(ic):
            out_dates.append(dates[i])
            ics.append(ic)
    return pd.Series(ics, index=pd.to_datetime(out_dates))


def summarize_ic(ic_s, label=""):
    n = len(ic_s)
    if n == 0:
        return {"ic": np.nan, "icir": np.nan, "ic_hit_ratio": np.nan, "n_ic_dates": 0, "label": label}
    ic = float(ic_s.mean())
    icir = float(ic_s.mean() / ic_s.std(ddof=1)) * np.sqrt(n) if ic_s.std(ddof=1) > 0 else np.nan
    hit = float((ic_s > 0).mean()) if n else np.nan
    return {"ic": ic, "icir": icir, "ic_hit_ratio": hit, "n_ic_dates": n, "label": label}


def decay_analysis(factor_panel, close, horizons=(1, 3, 5, 10, 20), min_valid=8):
    fwd = fwd_returns(close, horizons)
    out = {}
    for h in horizons:
        s = rank_ic_series(factor_panel, fwd[h], min_valid=min_valid)
        out[h] = float(s.mean()) if len(s) else np.nan
    return out


def turnover_10d(factor_panel):
    r = factor_panel.rank(axis=1, pct=True)
    to = (r - r.shift(10)).abs().mean(axis=1).mean()
    return float(to) if np.isfinite(to) else np.nan


def coverage_stats(factor_panel):
    total_cells = factor_panel.shape[0] * factor_panel.shape[1]
    valid_cells = int(factor_panel.notna().sum().sum())
    ge8 = float((factor_panel.notna().sum(axis=1) >= 8).mean())
    return {"coverage_asset_days": valid_cells / total_cells if total_cells else np.nan,
            "coverage_dates_ge8": ge8}


def regime_split(ic_s, recency_buckets=None):
    out = {}
    if len(ic_s) == 0:
        return out
    buckets = recency_buckets or [("2020-2022", "2020-01-01", "2022-12-31"),
                                  ("2023-2024", "2023-01-01", "2024-12-31"),
                                  ("2025-2026", "2025-01-01", "2026-12-31"),
                                  ("2027-2028", "2027-01-01", "2028-12-31"),
                                  ("2029-2030", "2029-01-01", "2030-12-31"),
                                  ("2031", "2031-01-01", "2099-12-31"),
                                  ("last180d", "2030-09-21", "2099-12-31"),
                                  ("last90d", "2030-12-20", "2099-12-31"),
                                  ("last60d", "2031-01-19", "2099-12-31")]
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
