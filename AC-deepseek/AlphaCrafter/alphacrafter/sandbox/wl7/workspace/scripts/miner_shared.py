"""Shared factor-mining utilities for miner_1.

Loads the 15-asset tradable universe + macro observation series on the master
daily calendar, restricts to dates <= visible_through (no lookahead), and
provides cross-sectional IC machinery used for factor validation.

Admission gates (benchmark-wide, 15-instrument universe):
  abs(daily paper IC)  >= 0.0070
  abs(daily paper ICIR) >= 0.0840
"""
import json
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"

IC_GATE = 0.0070
ICIR_GATE = 0.0840
MIN_VALID_PER_DATE = 8


def master_calendar(end="2028-09-11"):
    d = json.load(open("../persistent/date.json"))
    td = [x for x in d["trading_days"] if x <= end]
    return pd.DatetimeIndex(pd.to_datetime(td))


def load_close(end="2028-09-11"):
    """Return DataFrame of closes for 15 assets on the master calendar."""
    cal = master_calendar(end)
    out = pd.DataFrame(index=cal)
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"].reindex(cal).ffill()
        out[a] = s
    return out


def load_macro(end="2028-09-11"):
    cal = master_calendar(end)
    out = pd.DataFrame(index=cal)
    for m in MACRO:
        df = pd.read_csv(f"{INDEX_DIR}/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"].reindex(cal).ffill()
        out[m] = s
    return out


def forward_ret(close, h):
    """Close-to-close forward return over h calendar days."""
    return close.shift(-h) / close - 1.0


def daily_ic(factor, fwd_ret, min_valid=MIN_VALID_PER_DATE):
    """Per-date Spearman IC between factor and forward return.

    factor, fwd_ret: aligned DataFrames (dates x assets).
    Returns Series of IC indexed by date (only dates with >= min_valid valid).
    """
    valid = factor.notna() & fwd_ret.notna()
    n_valid = valid.sum(axis=1)
    rows = []
    dates = factor.index
    # Vectorized ranks per row
    fr = factor.rank(axis=1)
    rr = fwd_ret.rank(axis=1)
    mask = valid
    fr = fr.where(mask)
    rr = rr.where(mask)
    fm = fr.subtract(fr.mean(axis=1), axis=0)
    rm = rr.subtract(rr.mean(axis=1), axis=0)
    num = (fm * rm).sum(axis=1)
    den = np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
    ic = num / den
    ic = ic.where(n_valid >= min_valid)
    return ic.rename("ic")


def ic_stats(ic, horizon=10):
    s = ic.dropna()
    if len(s) == 0:
        return dict(ic=np.nan, icir=np.nan, hit=0.0, n=0)
    icm = s.mean()
    ics = s.std(ddof=1)
    icir = icm / ics if ics > 0 else np.nan
    hit = float((s > 0).mean())
    return dict(ic=float(icm), icir=float(icir), hit=float(hit), n=int(len(s)))


def summarize(factor, close, horizons=(1, 2, 3, 5, 10, 20)):
    """Full validation summary: IC/ICIR/hit by horizon + decay."""
    out = {}
    for h in horizons:
        fwd = forward_ret(close, h)
        ic = daily_ic(factor, fwd)
        st = ic_stats(ic, h)
        out[h] = st
    return out


def rank_turnover(factor, window=10):
    """Mean abs rank change over `window` rows (normalized by n_assets)."""
    r = factor.rank(axis=1)
    chg = (r - r.shift(window)).abs().mean(axis=1)
    return float(chg.dropna().mean())


def coverage_stats(factor, fwd):
    valid = factor.notna() & fwd.notna()
    n_valid = valid.sum(axis=1)
    cov_asset_days = float(valid.values.mean())
    cov_dates_ge8 = float((n_valid >= MIN_VALID_PER_DATE).mean())
    return dict(coverage_asset_days=cov_asset_days, coverage_dates_ge8=cov_dates_ge8)


# ---------------------------------------------------------------------------
# Active library factor implementations (for pairwise-correlation provenance)
# ---------------------------------------------------------------------------
def lib_rel_mom(close, window=20, skip=5):
    mom = close / close.shift(window + skip) - 1.0
    return mom.subtract(mom.median(axis=1), axis=0)


def lib_beta_ew(close, window=60):
    ret = close.pct_change()
    mkt = ret.mean(axis=1)
    cov = ret.rolling(window).cov(mkt)
    var = mkt.rolling(window).var()
    return cov.divide(var, axis=0)


def lib_corr_ew(ret, window=60, min_periods=30):
    out = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for a in ret.columns:
        others = ret.drop(columns=[a])
        c = ret[a].rolling(window, min_periods=min_periods).corr(others)
        out[a] = c.mean(axis=1)
    return out


def lib_downside_vol_ratio(close, window=20):
    ret = close.pct_change()
    neg = ret.where(ret < 0, 0.0)
    ds = (neg ** 2).rolling(window).mean().apply(np.sqrt)
    tot = ret.rolling(window).std()
    return -(ds / tot)


def lib_kurt(close, window=20, skip=5, min_periods=12):
    ret = close.pct_change()
    r = ret.shift(skip)
    return r.rolling(window, min_periods=min_periods).kurt()


def lib_max_ret(close, window=20):
    ret = close.pct_change()
    return ret.rolling(window).max()


def lib_dxy_beta_cond(close, dxy, beta_win=60, cond_win=20, min_periods=30):
    ret = close.pct_change()
    dxy_r = dxy.pct_change()
    cov = ret.rolling(beta_win, min_periods=min_periods).cov(dxy_r)
    var = dxy_r.rolling(beta_win, min_periods=min_periods).var()
    beta = cov.divide(var, axis=0)
    dxy_mom = dxy / dxy.shift(cond_win) - 1.0
    return -beta.multiply(dxy_mom, axis=0)


def lib_eurusd_beta_cond(close, eurusd, beta_win=60, cond_win=20, min_periods=30):
    ret = close.pct_change()
    fx_r = eurusd.pct_change()
    cov = ret.rolling(beta_win, min_periods=min_periods).cov(fx_r)
    var = fx_r.rolling(beta_win, min_periods=min_periods).var()
    beta = cov.divide(var, axis=0)
    fx_mom = eurusd / eurusd.shift(cond_win) - 1.0
    return beta.multiply(fx_mom, axis=0)


ACTIVE_LIB = {
    "rel_mom_20d_skip5": (lib_rel_mom, dict(window=20, skip=5)),
    "beta_ew_60d": (lib_beta_ew, dict(window=60)),
    "corr_ew_60": (lib_corr_ew, dict(window=60, min_periods=30)),
    "downside_vol_ratio_20": (lib_downside_vol_ratio, dict(window=20)),
    "kurt_20d_skip5": (lib_kurt, dict(window=20, skip=5, min_periods=12)),
    "max_ret_20d": (lib_max_ret, dict(window=20)),
    "dxy_beta_cond_60x20": (lib_dxy_beta_cond, dict(beta_win=60, cond_win=20, min_periods=30)),
    "eurusd_beta_cond_60x20": (lib_eurusd_beta_cond, dict(beta_win=60, cond_win=20, min_periods=30)),
}


def library_panel(close, macro):
    """Recompute active library factor panels over the full window."""
    ret = close.pct_change()
    panels = {}
    for name, (fn, kw) in ACTIVE_LIB.items():
        if name in ("corr_ew_60",):
            panels[name] = fn(ret, **kw)
        elif name in ("dxy_beta_cond_60x20",):
            panels[name] = fn(close, macro["DXY"], **kw)
        elif name in ("eurusd_beta_cond_60x20",):
            panels[name] = fn(close, macro["EURUSD"], **kw)
        else:
            panels[name] = fn(close, **kw)
    return panels


def max_lib_corr(candidate, lib_panels):
    """Max abs pairwise correlation (dates x assets flattened) vs active library."""
    flat = candidate.stack()
    best = 0.0
    pairs = {}
    for name, p in lib_panels.items():
        pflat = p.reindex(candidate.index).stack()
        df = pd.concat([flat.rename("f"), pflat.rename("p")], axis=1).dropna()
        if len(df) < 30:
            continue
        rho = float(df["f"].corr(df["p"]))
        pairs[name] = round(rho, 4)
        if abs(rho) > best:
            best = abs(rho)
    return best, pairs
