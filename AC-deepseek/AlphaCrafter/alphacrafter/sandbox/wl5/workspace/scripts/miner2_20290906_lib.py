"""Shared data-loading and factor-validation library for miner_2 (2029-09-06 cycle).

Only uses data through the simulator's visible date (2029-09-05). Never touches
live account state.
"""
import pandas as pd
import numpy as np
import os

VISIBLE_THROUGH = "2029-09-05"
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
    """Return dict symbol -> DataFrame (close, ret, plus OHLCV), sliced thru visible date."""
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
    """Apply func(symbol_df) -> Series(factor values indexed by date) for each symbol,
    return wide DataFrame (dates x symbols)."""
    cols = {}
    for sym, df in data.items():
        try:
            s = func(sym, df)
            if s is not None:
                cols[sym] = s.astype(float)
        except Exception as e:  # keep going; report failures
            print(f"  [warn] {sym} factor failed: {type(e).__name__}: {e}")
    panel = pd.DataFrame(cols)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def forward_returns(data, horizon=10):
    """Wide DataFrame of forward H-day returns (fwd_ret_t = close_{t+H}/close_t - 1)."""
    cols = {}
    for sym, df in data.items():
        cols[sym] = df["close"].shift(-horizon) / df["close"] - 1.0
    return pd.DataFrame(cols)


def ic_series(panel, fwd, min_valid=8):
    """Daily cross-sectional Spearman IC between factor panel and forward returns."""
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


def decay_analysis(panel, data, horizons=(1, 5, 10, 20), min_valid=8):
    res = {}
    for h in horizons:
        fwd = forward_returns(data, h)
        ics = ic_series(panel, fwd, min_valid)
        if len(ics):
            res[h] = round(ics.mean(), 5)
    return res


def run_validation(func, data, macro, label="", horizons=(5, 10, 20), min_valid=8, verbose=True):
    """Full validation pipeline for one factor function."""
    panel = build_panel(data, func, min_valid)
    cov, turn = coverage_turnover(panel)
    out = {"label": label, "coverage": cov, "turnover": turn, "n_instruments": panel.shape[1]}
    for h in horizons:
        fwd = forward_returns(data, h)
        ics = ic_series(panel, fwd, min_valid)
        s = summarize_ic(ics, f"{label} H={h}")
        out[f"H{h}"] = s
    if verbose:
        print(f"=== {label} | coverage={cov:.3f} turnover={turn:.4f} n_inst={panel.shape[1]}")
        for h in horizons:
            s = out[f"H{h}"]
            if s.get("n_dates"):
                print(f"  H={h}: IC={s['IC']:.5f} ICIR={s['ICIR']:.3f} hit={s['hit_ratio']:.3f} "
                      f"n_dates={s['n_dates']} IC_se={s['IC_se']:.5f}")
    return panel, out
