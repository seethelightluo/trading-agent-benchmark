"""Shared helper for miner3 factor validation."""
import pandas as pd, numpy as np, os, json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def load_data():
    uni = {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=4000)
        if df is None or len(df) < 300:
            df = get_index_daily_data(symbol=s, days=4000)
        if df is not None and len(df) >= 300:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            uni[s] = df
        else:
            print("WARN no data", s, None if df is None else len(df))
    return uni

def build_panel(uni):
    close = pd.DataFrame({s: uni[s]['close'] for s in uni}).sort_index()
    ret = close.pct_change()
    return close, ret

def rank_ic(factor_series, fwd, ret_panel, min_dates=8):
    """factor_series: DataFrame indexed by date with asset columns.
       fwd: forward return horizon in days, using ret_panel.
       Returns per-date IC series."""
    fwd_ret = ret_panel.shift(-fwd)
    dates = []
    ics = []
    for dt in factor_series.index:
        frow = factor_series.loc[dt]
        rrow = fwd_ret.loc[dt]
        mask = frow.notna() & rrow.notna()
        if mask.sum() < 8:
            continue
        ic = frow[mask].corr(rrow[mask], method='spearman')
        if np.isnan(ic):
            continue
        dates.append(dt); ics.append(ic)
    return pd.Series(ics, index=dates)

def summarize(name, ic_series, extra=None):
    ic_mean = ic_series.mean()
    ic_std = ic_series.std(ddof=1)
    icir = ic_mean/ic_std if ic_std and ic_std>0 else 0.0
    hit = (ic_series>0).mean()
    out = {
        "factor": name,
        "n_ic_dates": int(len(ic_series)),
        "ic": round(float(ic_mean),4),
        "icir": round(float(icir),4),
        "ic_hit_ratio": round(float(hit),4),
        "first": str(ic_series.index.min().date()) if len(ic_series) else None,
        "last": str(ic_series.index.max().date()) if len(ic_series) else None,
    }
    if extra: out.update(extra)
    # admission gates
    out["PASS"] = bool(abs(ic_mean)>=0.0070 and abs(icir)>=0.0840)
    print(json.dumps(out, indent=2))
    return out