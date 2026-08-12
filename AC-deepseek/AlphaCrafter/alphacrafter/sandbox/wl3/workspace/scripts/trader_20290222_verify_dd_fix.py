"""Trader verification: dd_duration_120_resid fix on pandas 3.0.5."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

OBS_ONLY = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}


def get_df(symbol, days=400):
    try:
        if symbol in OBS_ONLY:
            return get_index_daily_data(symbol, days=days)
        return get_stock_daily_data(symbol, days=days)
    except Exception:
        return None


def series(df, col="close"):
    if df is None or col not in df or len(df) < 40:
        return None
    s = df[col].astype(float)
    try:
        s.index = pd.to_datetime(df["date"])
    except Exception:
        s.index = pd.RangeIndex(len(s))
    return s


def beta_last(y, x, win=60, min_obs=20):
    q = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().tail(win)
    if len(q) < min_obs:
        return None
    vx = float(q.x.var())
    if vx <= 1e-14:
        return None
    return float(q.y.cov(q.x) / vx)


def dd_duration_resid(c, r, r_spx):
    """Current strategy.py implementation (claimed fixed)."""
    try:
        hi = c.rolling(120).max()
        if isinstance(c.index, pd.DatetimeIndex):
            last_high = c.index.to_series().where(c == hi).ffill()
            dur = np.log1p((c.index - last_high).dt.days.fillna(0).astype(float))
        else:
            pos = pd.Series(np.arange(len(c)), index=c.index)
            dur = np.log1p((pos - pos.where(c == hi).ffill()).fillna(0).astype(float))
        mom = c.shift(5) / c.shift(125) - 1.0
        zmom = (mom - mom.rolling(250).mean()) / mom.rolling(250).std()
        b = beta_last(r, r_spx)
        v = float(dur.iloc[-1]) - (b * float(zmom.iloc[-1]) if b is not None else 0.0)
        return v if np.isfinite(v) else None
    except Exception as e:
        print(f"  EXC: {type(e).__name__}: {e}")
        return None


acct = get_account_dict()
assets = list(acct["watch_list"])
print("watchlist:", assets)
print("date index type:", type(acct.get("date", "n/a")))

frames = {a: get_df(a) for a in assets}
close = {a: series(frames[a]) for a in assets}
ret = {a: close[a].pct_change() for a in assets}
r_spx = ret["SPX"]

vals = {}
for a in assets:
    v = dd_duration_resid(close[a], ret[a], r_spx)
    vals[a] = v
    print(f"  {a}: {v}")

finite = {a: v for a, v in vals.items() if v is not None and np.isfinite(v)}
print(f"\nfinite: {len(finite)}/{len(assets)}")
print("distinct values:", len(set(round(v, 6) for v in finite.values())))
if finite:
    lo = min(finite.values())
    hi = max(finite.values())
    print(f"range: [{lo:.4f}, {hi:.4f}]  spread={hi - lo:.4f}")
    print("cross-sectional signal ACTIVE" if hi - lo > 1e-6 else "STILL INERT (all equal)")
