"""Probe: find the pandas incompatibility breaking round-11 style screens."""
import traceback
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DAYS = 4000

closes = {}
for s in WATCH:
    df = get_stock_daily_data(s, days=DAYS)
    if df is None or not len(df):
        continue
    df = df.set_index("date")
    closes[s] = df["close"].astype(float)

close = pd.concat(closes, axis=1, sort=True)
close = close[~close.index.duplicated(keep="last")].sort_index()
print("panel:", close.shape, "pandas:", pd.__version__)

def per_asset(fn):
    def wrapper(panel):
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            cols[a] = fn(s)
        return pd.DataFrame(cols, index=panel.index)
    return wrapper

def fwd_returns(panel, h):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)

def rank_ic_series(factor, fwd):
    ics = []
    for d in factor.index.intersection(fwd.index):
        f = factor.loc[d].dropna()
        r = fwd.loc[d].reindex(f.index).dropna()
        if len(r) >= 8:
            ics.append((d, r.corr(f.reindex(r.index), method="spearman")))
    return pd.Series(dict(ics)).sort_index()

for name, fn in [
    ("kaufman", lambda: per_asset(lambda s: (s - s.shift(20)).abs() / s.diff().abs().rolling(20).sum().replace(0, np.nan))(close)),
    ("stoch", lambda: (close - close.rolling(14).min()) / (close.rolling(14).max() - close.rolling(14).min()).replace(0, np.nan)),
    ("rolling_apply", lambda: per_asset(lambda s: s.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True))(close)),
    ("fwd10_ic", lambda: rank_ic_series((close.pct_change().rolling(20).mean()).loc[:], fwd_returns(close, 10))),
]:
    try:
        out = fn()
        print(name, "OK", getattr(out, "shape", len(out)))
    except Exception:
        print("=== ", name, "FAILED")
        traceback.print_exc(limit=3)
