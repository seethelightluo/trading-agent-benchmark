"""Shared data-loading and factor-validation library for miner_2 (2035-12-06 cycle).

Only uses data through the simulator's visible date (2035-12-05, previous
completed trading day before current date 2035-12-06). Never touches live
account state or advances persistent dates.
"""
import pandas as pd
import numpy as np
import os

VISIBLE_THROUGH = "2035-12-05"
STOCK_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"

WATCHLIST = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
             "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

_cache = {}


def load_close(symbol, dirpath=STOCK_DIR):
    key = (symbol, dirpath)
    if key in _cache:
        return _cache[key]
    df = pd.read_csv(os.path.join(dirpath, f"{symbol}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date")
    df = df.set_index("date")
    _cache[key] = df
    return df


def load_all_data():
    """Return dict symbol -> DataFrame (close, open, high, low, volume, ret)."""
    out = {}
    for sym in WATCHLIST:
        df = load_close(sym)
        d = pd.DataFrame(index=df.index)
        d["close"] = df["close"].astype(float)
        d["open"] = df["open"].astype(float)
        d["high"] = df["high"].astype(float)
        d["low"] = df["low"].astype(float)
        d["volume"] = df["volume"].astype(float)
        d["ret"] = d["close"].pct_change()
        out[sym] = d
    macro = {}
    for sym in MACRO:
        df = load_close(sym, INDEX_DIR)
        d = pd.DataFrame(index=df.index)
        d["close"] = df["close"].astype(float)
        d["ret"] = d["close"].pct_change()
        macro[sym] = d
    return out, macro


def build_panel(data, func, min_valid=8):
    cols = {}
    for sym, df in data.items():
        try:
            s = func(sym, df)
            if s is not None:
                cols[sym] = s.astype(float)
        except Exception as e:
            print(f"  [warn] {sym} factor failed: {type(e).__name__}: {e}")
    panel = pd.DataFrame(cols)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def forward_returns(data, horizon=10):
    cols = {}
    for sym, df in data.items():
        cols[sym] = df["close"].shift(-horizon) / df["close"] - 1.0
    return pd.DataFrame(cols)


def ic_series(panel, fwd, min_valid=8):
    dates, ics = [], []
    for dt in panel.index:
        fv = panel.loc[dt]
        fr = fwd.loc[dt]
        mask = fv.notna() & fr.notna()
        if mask.sum() >= min_valid:
            try:
                ic = fv[mask].corr(fr[mask], method="spearman")
            except Exception:
                ic = np.nan
            if pd.notna(ic):
                dates.append(dt)
                ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def summarize_ic(ics, label=""):
    if len(ics) == 0:
        return {"label": label, "n_dates": 0}
    ic = ics.mean()
    icir = ic / ics.std() if ics.std() > 0 else 0.0
    hit = float((np.sign(ics) == np.sign(ic)).mean())
    t = ics.std() / np.sqrt(len(ics))
    return {
        "label": label,
        "n_dates": len(ics),
        "IC": round(ic, 5),
        "ICIR": round(icir, 4),
        "hit_ratio": round(hit, 4),
        "IC_se": round(t, 5),
    }


def coverage_turnover(panel):
    cov = panel.notna().mean().mean()
    tn = panel.ffill().diff().abs().mean().mean()
    return round(cov, 4), round(tn, 5)


def decay_analysis(panel, data, horizons=(5, 10, 20), min_valid=8):
    res = {}
    for h in horizons:
        fwd = forward_returns(data, h)
        ics = ic_series(panel, fwd, min_valid)
        if len(ics):
            res[h] = round(ics.mean(), 5)
    return res


def beta_series(asset_ret, bench_ret, win, min_p=10):
    b = asset_ret.rolling(win, min_periods=max(win // 2, min_p)).cov(bench_ret) / \
        bench_ret.rolling(win, min_periods=max(win // 2, min_p)).var()
    return b