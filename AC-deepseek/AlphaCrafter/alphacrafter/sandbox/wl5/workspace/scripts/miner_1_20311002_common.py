# -*- coding: utf-8 -*-
"""Shared data-loading + validation utilities for miner_1 (2031-10-02 cycle).
Visible data window: through 2031-10-01 (previous completed trading day).
"""
import json
import numpy as np
import pandas as pd

VISIBLE_THROUGH = "2031-10-01"
CURRENT_DATE = "2031-10-02"

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


def vol_panel(symbols=None):
    symbols = symbols or WATCH
    out = {}
    for s in symbols:
        df = load_asset(s)
        if len(df) == 0:
            continue
        out[s] = df.set_index("date")["volume"]
    return pd.DataFrame(out).sort_index()


def ohlcv_panels(symbols=None):
    symbols = symbols or WATCH
    out = {k: {} for k in ["open", "high", "low", "close", "volume"]}
    for s in symbols:
        df = load_asset(s)
        if len(df) == 0:
            continue
        sub = df.set_index("date")
        for k in out:
            out[k][s] = sub[k]
    return {k: pd.DataFrame(v).sort_index() for k, v in out.items()}


def macro_panel(symbols=None):
    symbols = symbols or MACRO
    out = {}
    for s in symbols:
        df = load_macro(s)
        if len(df) == 0:
            continue
        out[s] = df.set_index("date")["close"]
    return pd.DataFrame(out).sort_index()


def rank_ic(factor_panel, fwd_ret, min_valid=8):
    """Daily cross-sectional Spearman IC between factor and forward return.
    A date needs >= min_valid instruments to count as an IC observation."""
    fwd = fwd_ret.reindex(factor_panel.index)
    dates, ics = [], []
    for dt in factor_panel.index:
        f = factor_panel.loc[dt]
        r = fwd.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() >= min_valid:
            ic = f[mask].rank().corr(r[mask].rank())
            if pd.notna(ic):
                dates.append(dt)
                ics.append(ic)
    return pd.Series(ics, index=dates)


def summarize_ic(ics, horizon, label=""):
    if len(ics) == 0:
        print(f"[{label}] no IC observations")
        return None
    ic = float(ics.mean())
    icir = float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0
    hit = float((ics > 0).mean())
    print(f"[{label}] horizon={horizon}d n={len(ics)} IC={ic:.4f} ICIR={icir:.4f} hit={hit:.3f}")
    return {"ic": ic, "icir": icir, "hit": hit, "n": len(ics)}


def ic_by_regime(ics, regime_break="2024-01-01"):
    """Split IC series into pre/post break for regime robustness."""
    pre = ics[ics.index < pd.Timestamp(regime_break)]
    post = ics[ics.index >= pd.Timestamp(regime_break)]
    out = {}
    for name, s in [("pre", pre), ("post", post)]:
        if len(s) >= 30:
            out[name] = {"ic": float(s.mean()), "icir": float(s.mean() / s.std()) if s.std() > 0 else 0.0, "n": len(s)}
        else:
            out[name] = {"ic": None, "icir": None, "n": len(s)}
    return out


def turnover_rank(factor_panel, horizon=10):
    """Mean absolute rank change between dates horizon apart."""
    r = factor_panel.rank(axis=1)
    diff = r.diff(horizon).abs().mean(axis=1)
    return float(diff.dropna().mean())


def coverage_report(factor_panel):
    valid = factor_panel.notna()
    asset_days = float(valid.values.mean())
    dates_ge8 = float((valid.sum(axis=1) >= 8).mean())
    return {"coverage_asset_days": asset_days, "coverage_dates_ge8": dates_ge8}


# ---------------- library factor replicas (for correlation audit) ----------------

def lib_trend_r2(close, window=30):
    lc = np.log(close)
    t = np.arange(len(close), dtype=float)
    x = pd.Series(t, index=close.index)
    xm = x.rolling(window, min_periods=18).mean()
    ym = lc.rolling(window, min_periods=18).mean()
    cov = (x - xm) * (lc - ym)
    cov = cov.rolling(window, min_periods=18).mean()
    vx = ((x - xm) ** 2).rolling(window, min_periods=18).mean()
    vy = ((lc - ym) ** 2).rolling(window, min_periods=18).mean()
    r2 = cov ** 2 / (vx * vy)
    return np.sign(cov) * r2


def lib_semi_down(close, window=20):
    r = close.pct_change()
    return r.clip(upper=0).rolling(window, min_periods=10).mean() / r.rolling(window, min_periods=10).std()


def lib_mom(close, lookback, skip):
    return close.shift(skip) / close.shift(skip + lookback) - 1.0


def lib_beta(close, macro_ret, window=60):
    ra = close.pct_change()
    rb = macro_ret.pct_change()
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
        "dxy_beta_60": lib_beta(close, macro["DXY"]),
        "cny_beta_60": lib_beta(close, macro["USDCNY"]),
        "vol_of_vol20x60": lib_vol_of_vol(close),
        "mom_10d_skip5": lib_mom(close, 10, 5),
        "time_under_water_120": lib_tuw(close),
        "tail_ratio_20": lib_tail_ratio(close),
        "vix_beta_cond_60x20": lib_vix_beta_cond(close, macro["VIX"]),
        "kurt_20": lib_kurt(close),
        "WTI_BETA_60": lib_wti_beta(close, close["WTI"]),
    }
    res = {}
    for name, lf in libs.items():
        both = pd.concat([factor_panel.stack(), lf.stack()], axis=1, keys=["f", "l"]).dropna()
        if len(both) >= min_overlap:
            r = float(both["f"].corr(both["l"]))
            res[name] = {"pooled_rho": r, "n": len(both)}
        else:
            res[name] = {"pooled_rho": None, "n": 0}
    return res


def admission_check(ic, icir, label=""):
    passed = (abs(ic) >= IC_THRESHOLD) and (abs(icir) >= ICIR_THRESHOLD)
    print(f"[GATE] {label}: IC={ic:.4f} ICIR={icir:.4f} -> {'PASS' if passed else 'FAIL'}")
    return passed
