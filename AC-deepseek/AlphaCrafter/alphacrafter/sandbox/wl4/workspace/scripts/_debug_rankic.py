import numpy as np, pandas as pd
VISIBLE = pd.Timestamp('2028-03-10')
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

def load_close():
    frames = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
        frames[a] = df["close"]
    return pd.DataFrame(frames)

def load_ohlcv():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
        out[a] = df
    return out

def fwd_returns(panel, h=10):
    return panel.shift(-h) / panel - 1.0

def rank_ic_series(factor, fwd, min_valid=8):
    dates = factor.index.intersection(fwd.index)
    ics = {}
    for dt in dates:
        f = factor.loc[dt]
        r = fwd.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_valid:
            continue
        ic = f[mask].corr(r[mask], method="spearman")
        if np.isfinite(ic):
            ics[dt] = ic
    return pd.Series(ics, name="ic")

def rolling_beta(ret, bench, win=60, minp=40):
    cov = ret.rolling(win, min_periods=minp).cov(bench)
    var = bench.rolling(win, min_periods=minp).var()
    return cov / var

panel = load_close()
ohlcv = load_ohlcv()
rets = panel.pct_change()
mkt = rets.mean(axis=1)
vols = pd.DataFrame({a: ohlcv[a]["volume"] for a in ASSETS})
down_mkt = mkt.where(mkt < 0, 0.0)
f = rolling_beta(rets, down_mkt, 60, 40)
print("factor index type:", type(f.index), f.index[:2])
fwd = fwd_returns(panel, 10)
ic = rank_ic_series(f, fwd, min_valid=8)
print("ic index type:", type(ic.index), "len:", len(ic), "dtype:", ic.index.dtype)
try:
    filt = ic[ic.index >= pd.Timestamp(panel.index[-501])]
    print("filter ok len:", len(filt))
except Exception as e:
    print("filter ERR:", repr(e))
