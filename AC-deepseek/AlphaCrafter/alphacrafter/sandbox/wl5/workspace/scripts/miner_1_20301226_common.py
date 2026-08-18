"""Shared data-loading + validation utilities for miner_1 (2030-12-26 cycle)."""
import json
import numpy as np
import pandas as pd

VISIBLE_THROUGH = "2030-12-25"
CURRENT_DATE = "2030-12-26"

IC_THRESHOLD = 0.0070
ICIR_THRESHOLD = 0.0840
CORR_THRESHOLD = 0.5

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
    nan_mask = np.isnan(a)
    out = np.full(len(a), np.nan)
    if nan_mask.all():
        return out
    valid = ~nan_mask
    ranks = pd.Series(a[valid]).rank(method="average").values
    out[valid] = ranks
    return out


def rank_ic_series(factor_panel, fwd_ret, min_valid=8):
    """Daily cross-sectional rank IC between factor and forward return panel."""
    dates = factor_panel.index
    out = []
    for i, dt in enumerate(dates):
        f = factor_panel.loc[dt]
        r = fwd_ret.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() < min_valid:
            continue
        fr = _rankdata_np(f.values)
        rr = _rankdata_np(r.values)
        # pairwise-complete correlation on the valid subset (NaN-safe)
        m2 = ~np.isnan(fr) & ~np.isnan(rr)
        frv = fr[m2]
        rrv = rr[m2]
        if len(frv) < min_valid or np.nanstd(frv) == 0 or np.nanstd(rrv) == 0:
            continue
        ic = np.corrcoef(frv, rrv)[0, 1]
        if np.isfinite(ic):
            out.append((dt, ic))
    if not out:
        return pd.Series(dtype=float)
    return pd.Series([x[1] for x in out], index=pd.DatetimeIndex([x[0] for x in out]))


def summarize_ic(ic_s, label=""):
    n = len(ic_s)
    if n == 0:
        return {"ic": np.nan, "icir": np.nan, "ic_hit_ratio": np.nan, "n_ic_dates": 0}
    ic = float(ic_s.mean())
    sd = float(ic_s.std(ddof=1))
    icir = ic / sd * np.sqrt(n) if sd > 0 else np.nan
    return {"ic": ic, "icir": float(icir), "ic_hit_ratio": float((ic_s > 0).mean()),
            "n_ic_dates": n}


def decay_analysis(factor_panel, close, horizons=(1, 3, 5, 10, 20), min_valid=8):
    out = {}
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        ic_s = rank_ic_series(factor_panel, fwd, min_valid)
        out[str(h)] = float(ic_s.mean()) if len(ic_s) else np.nan
    return out


def turnover_10d(factor_panel):
    """Mean absolute change in cross-sectional rank (normalized by n) between
    observations ~10 trading days apart (daily resample of the rank panel)."""
    ranks = factor_panel.rank(axis=1, pct=True)
    sampled = ranks.resample("10D").last()
    chg = sampled.diff().abs().mean(axis=1).dropna()
    return float(chg.mean()) if len(chg) else np.nan


def coverage_stats(factor_panel):
    valid = factor_panel.notna()
    n_asset_days = int(valid.sum().sum())
    total_asset_days = factor_panel.size
    ge8 = (valid.sum(axis=1) >= 8).mean()
    return {"coverage_asset_days": float(n_asset_days / total_asset_days),
            "coverage_dates_ge8": float(ge8)}


def regime_split(ic_s):
    out = {}
    if len(ic_s) == 0:
        return out
    for lab, lo, hi in [("2020-2022", "2020-01-01", "2022-12-31"),
                        ("2023-2024", "2023-01-01", "2024-12-31"),
                        ("2025-2026", "2025-01-01", "2026-12-31"),
                        ("2027+", "2027-01-01", "2099-12-31"),
                        ("2029+", "2029-01-01", "2099-12-31"),
                        ("2029-2030", "2029-01-01", "2030-12-31"),
                        ("2026-07+", "2026-07-16", "2099-12-31"),
                        ("last180d", "2030-06-29", "2099-12-31"),
                        ("last90d", "2030-09-27", "2099-12-31")]:
        sub = ic_s[(ic_s.index >= pd.Timestamp(lo)) & (ic_s.index <= pd.Timestamp(hi))]
        if len(sub):
            out[lab] = {"ic": float(sub.mean()),
                        "icir": float(sub.mean() / sub.std(ddof=1)) * np.sqrt(len(sub)) if sub.std(ddof=1) > 0 else np.nan,
                        "n": len(sub)}
    return out


# ---------------- library factor implementations (for correlation audit) ----------------
def lib_trend_r2(close, window=30):
    def _signed_r2(arr):
        x = np.arange(len(arr), dtype=float)
        y = np.log(arr)
        if len(y) < 3 or np.std(y) == 0:
            return np.nan
        slope = np.polyfit(x, y, 1)[0]
        corr = np.corrcoef(x, y)[0, 1]
        return np.sign(slope) * corr ** 2
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


def roll_beta(x, ref, win=60, minp=30):
    cov = x.rolling(win, min_periods=minp).cov(ref)
    var = ref.rolling(win, min_periods=minp).var()
    return cov.div(var, axis=0).replace([np.inf, -np.inf], np.nan)


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
        "cny_beta_60": roll_beta(close.pct_change(), macro["USDCNY"].pct_change(), 60, 30),
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


def full_validation(factor_panel, close, macro, label="", horizon=10):
    """Compute the full validation bundle for a candidate factor panel."""
    fwd = close.shift(-horizon) / close - 1.0
    ic_s = rank_ic_series(factor_panel, fwd)
    summ = summarize_ic(ic_s, label)
    cov = coverage_stats(factor_panel)
    turn = turnover_10d(factor_panel)
    decay = decay_analysis(factor_panel, close)
    regimes = regime_split(ic_s)
    corrs, max_abs = library_correlation(factor_panel, close, macro)
    return {
        "metrics": {
            "ic": summ["ic"], "icir": summ["icir"], "ic_hit_ratio": summ["ic_hit_ratio"],
            "n_ic_dates": summ["n_ic_dates"],
            "coverage_asset_days": cov["coverage_asset_days"],
            "coverage_dates_ge8": cov["coverage_dates_ge8"],
            "turnover_10d_rank": turn,
            "decay_ic_by_horizon": decay,
            "max_abs_library_correlation": max_abs,
        },
        "library_correlations": corrs,
        "regime_split": regimes,
    }
