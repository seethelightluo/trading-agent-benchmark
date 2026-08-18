"""Shared helpers for miner_2 factor validation (2029-05-03 cycle).

Data rules:
- Use tradable universe of 15 assets from ../persistent/stock_data.
- Macro observation-only signals from ../persistent/index_data.
- Never look past visible_through=2029-05-02 (current_date 2029-05-03).
"""
import json
import os
import numpy as np
import pandas as pd

VISIBLE_THROUGH = "2029-05-02"
CURRENT_DATE = "2029-05-03"

TRADABLE = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
            "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]

MACRO = ["DXY", "VIX", "USDJPY", "EURUSD", "USDCNY"]

STOCK_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"

IC_THRESHOLD = 0.0070
ICIR_THRESHOLD = 0.0840


def load_asset(sym):
    df = pd.read_csv(os.path.join(STOCK_DIR, f"{sym}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VISIBLE_THROUGH].sort_values("date").reset_index(drop=True)
    return df


def load_macro(sym):
    df = pd.read_csv(os.path.join(INDEX_DIR, f"{sym}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VISIBLE_THROUGH].sort_values("date").reset_index(drop=True)
    return df


def price_panel(field="close", symbols=None):
    """Return DataFrame (dates x assets) of the given OHLCV field."""
    symbols = symbols or TRADABLE
    out = {}
    for s in symbols:
        df = load_asset(s)
        out[s] = pd.Series(df[field].values, index=pd.to_datetime(df["date"].values))
    p = pd.DataFrame(out).sort_index()
    return p


def macro_panel(sym):
    df = load_macro(sym)
    return pd.Series(df["close"].values, index=pd.to_datetime(df["date"].values))


def fwd_returns(close, horizons=(1, 2, 3, 5, 10, 20)):
    """Forward returns at each horizon (t -> t+h), aligned on t."""
    out = {}
    for h in horizons:
        out[h] = close.shift(-h) / close - 1.0
    return out


def _rankdata(a):
    """Average-tie ranks (scipy.stats.rankdata equivalent), 1..n."""
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=np.intp)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return (count[dense] + count[dense - 1] + 1) / 2.0


def rank_ic_series(factor_panel, fwd_ret, min_valid=8):
    """Per-date Spearman rank IC between factor and forward return (numpy)."""
    F = factor_panel.values.astype(float)
    R = fwd_ret.values.astype(float)
    dates = factor_panel.index
    ics, out_dates = [], []
    for i in range(len(dates)):
        f, r = F[i], R[i]
        m = ~(np.isnan(f) | np.isnan(r))
        if m.sum() < min_valid:
            continue
        fr = _rankdata(f[m])
        rr = _rankdata(r[m])
        fr = fr - fr.mean()
        rr = rr - rr.mean()
        denom = np.sqrt((fr ** 2).sum() * (rr ** 2).sum())
        if denom == 0:
            continue
        ics.append(float((fr * rr).sum() / denom))
        out_dates.append(dates[i])
    return pd.Series(ics, index=pd.DatetimeIndex(out_dates))


def summarize_ic(ic_s, label=""):
    if len(ic_s) == 0:
        print(f"{label}: NO IC DATES")
        return None
    mean_ic = ic_s.mean()
    std_ic = ic_s.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit = (ic_s > 0).mean()
    res = {"ic": float(mean_ic), "icir": float(icir), "ic_hit_ratio": float(hit),
           "n_ic_dates": int(len(ic_s)), "ic_std": float(std_ic)}
    print(f"{label}: IC={mean_ic:.4f} ICIR={icir:.4f} hit={hit:.3f} n={len(ic_s)}")
    return res


def decay_analysis(factor_panel, close, horizons=(1, 2, 3, 5, 10, 20), min_valid=8):
    fwd = fwd_returns(close, horizons)
    out = {}
    for h in horizons:
        ic_s = rank_ic_series(factor_panel, fwd[h], min_valid=min_valid)
        out[str(h)] = float(ic_s.mean()) if len(ic_s) else float("nan")
    return out


def turnover_10d(factor_panel):
    """Mean absolute cross-sectional rank change over a 10-day span (per asset, averaged)."""
    ranks = factor_panel.rank(axis=1)
    diffs = []
    for i in range(10, len(ranks)):
        d = (ranks.iloc[i] - ranks.iloc[i - 10]).abs().mean()
        if np.isfinite(d):
            diffs.append(d)
    return float(np.mean(diffs)) if diffs else float("nan")


def coverage_stats(factor_panel):
    n_assets = factor_panel.shape[1]
    valid_days = factor_panel.notna().sum().sum()
    total_days = factor_panel.shape[0] * n_assets
    cov_asset_days = valid_days / total_days
    ge8 = (factor_panel.notna().sum(axis=1) >= 8).mean()
    return {"coverage_asset_days": float(cov_asset_days),
            "coverage_dates_ge8": float(ge8)}


def regime_split(ic_s):
    out = {}
    if len(ic_s) == 0:
        return out
    bins = [("2020-2022", "2020-01-01", "2022-12-31"),
            ("2023-2024", "2023-01-01", "2024-12-31"),
            ("2025-2026", "2025-01-01", "2026-12-31"),
            ("2027-2028", "2027-01-01", "2028-12-31"),
            ("2029", "2029-01-01", "2029-12-31")]
    for name, a, b in bins:
        sub = ic_s[(ic_s.index >= a) & (ic_s.index <= b)]
        if len(sub) == 0:
            continue
        mean_ic = sub.mean()
        std_ic = sub.std(ddof=1)
        out[name] = {"ic": float(mean_ic), "icir": float(mean_ic / std_ic) if std_ic > 0 else 0.0,
                     "n": int(len(sub))}
    return out


# ---------------- library factor implementations (identical to strategy.py) ----------------

def _signed_r2(arr):
    x = np.arange(len(arr), dtype=float)
    m = np.isfinite(arr)
    if m.sum() < 2:
        return np.nan
    x, y = x[m], arr[m]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    beta = np.cov(x, y)[0, 1] / np.var(x)
    r = np.corrcoef(x, y)[0, 1]
    return float(np.sign(beta) * r * r)


def lib_trend_r2(close, window=30):
    logc = np.log(close)
    return logc.rolling(window, min_periods=18).apply(_signed_r2, raw=False)


def lib_semi_down(close, window=20):
    r = close.pct_change()
    down = r.clip(upper=0.0) ** 2
    up = r.clip(lower=0.0) ** 2
    return np.sqrt(down.rolling(window).mean()) / np.sqrt(up.rolling(window).mean()) - 1.0


def lib_mom(close, lookback, skip):
    return close.shift(skip) / close.shift(skip + lookback) - 1.0


def lib_dxy_beta(close, dxy, window=60):
    ra = close.pct_change()
    rb = dxy.pct_change()
    cov = ra.rolling(window).cov(rb)
    var = rb.rolling(window).var()
    return cov / var


def lib_vol_of_vol(close, w1=20, w2=60):
    v = close.pct_change().rolling(w1).std()
    return v.rolling(w2).std()


def lib_tuw(close, window=120):
    roll_max = close.rolling(window, min_periods=60).max()
    dd = close / roll_max - 1.0
    tuw = (dd < 0).rolling(window, min_periods=60).mean()
    return -tuw  # direction -1 in ensemble (high time-under-water is bad)


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
