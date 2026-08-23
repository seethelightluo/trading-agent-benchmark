"""miner_1 screening harness at 2031-11-27 (visible through 2031-11-26, no lookahead).

Screens several candidate factor families on the 15-asset cross-asset universe,
computing rank IC / ICIR (admission horizon 10d), decay, coverage, turnover,
and regime splits. Admission gates: |IC|>=0.0070, |ICIR|>=0.0840.
"""
import json, base64, zlib, io, glob, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI",
          "N225","NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
END = pd.Timestamp("2031-11-26")
START = pd.Timestamp("2020-01-02")
IC_GATE = 0.0070
ICIR_GATE = 0.0840
RHO_GATE = 0.5
MIN_ASSETS = 8
HORIZON = 10

def load_panel():
    closes, vols, opens, highs, lows = {}, {}, {}, {}, {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= END].set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        closes[a] = df["close"].astype(float)
        vols[a] = df["volume"].astype(float)
        opens[a] = df["open"].astype(float)
        highs[a] = df["high"].astype(float)
        lows[a] = df["low"].astype(float)
    return (pd.DataFrame(closes), pd.DataFrame(vols), pd.DataFrame(opens),
            pd.DataFrame(highs), pd.DataFrame(lows))

def load_macro():
    out = {}
    for k in ["DXY","USDCNY","USDJPY","EURUSD","VIX"]:
        df = pd.read_csv(f"{INDEX_DIR}/{k}.csv", parse_dates=["date"])
        df = df[df["date"] <= END].set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out[k] = df["close"].astype(float)
    return out

def dense_per_asset(close, vol, open_, high, low):
    d = {}
    for a in ASSETS:
        idx = close[a].dropna().index
        d[a] = {"close": close[a].reindex(idx), "vol": vol[a].reindex(idx),
                "open": open_[a].reindex(idx), "high": high[a].reindex(idx),
                "low": low[a].reindex(idx)}
    return d

def factor_panel(fn, close, vol, open_, high, low, macro, **params):
    dense = dense_per_asset(close, vol, open_, high, low)
    out = {}
    for a in ASSETS:
        dc = dense[a]
        try:
            s = fn(dc["close"], dc["vol"], dc["open"], dc["high"], dc["low"], macro, **params)
            out[a] = pd.Series(np.asarray(s).ravel(), index=dc["close"].index).reindex(close.index)
        except Exception as e:
            out[a] = pd.Series(np.nan, index=close.index)
    return pd.DataFrame(out)

def compute_ics(fdf, close, horizon, start=START, end=END, min_assets=MIN_ASSETS):
    fwd = close.pct_change(horizon).shift(-horizon)
    out = []
    for dt in fdf.index[(fdf.index>=start)&(fdf.index<=end)]:
        fv, rv = fdf.loc[dt], fwd.loc[dt]
        m = fv.notna() & rv.notna()
        if m.sum() < min_assets: continue
        ic,_ = spearmanr(fv[m], rv[m])
        if np.isfinite(ic): out.append((dt, ic))
    return out

def stats(ics):
    arr = np.array([x[1] for x in ics])
    mu=arr.mean(); sd=arr.std(ddof=1) if len(arr)>1 else 0
    icir = mu/sd if sd>0 else 0.0
    hit = float((arr>0).mean()) if mu>=0 else float((arr<0).mean())
    return round(mu,4), round(icir,4), round(hit,4), len(ics)

def full_validate(fdf, close, horizon=HORIZON):
    res = {}
    for h in [1,2,3,5,10,20]:
        ics = compute_ics(fdf, close, h)
        if ics:
            mu,icir,hit,n = stats(ics)
            res[str(h)] = mu
    ics = compute_ics(fdf, close, horizon)
    mu,icir,hit,n = stats(ics)
    total=int(fdf.notna().sum().sum()); cells=int(fdf.size)
    ge8 = int((fdf.notna().sum(axis=1)>=8).sum())
    sub=fdf.dropna(how="all"); rows=sub.iloc[::10]; ranks=rows.rank(axis=1)
    chg=[]; prev=None
    for _,r in ranks.iterrows():
        r=r.dropna()
        if prev is not None:
            both=prev.index.intersection(r.index)
            if len(both)>=MIN_ASSETS: chg.append(float((r[both]-prev[both]).abs().mean()))
        prev=r
    to=float(np.mean(chg)) if chg else float("nan")
    return {"ic":mu,"icir":icir,"hit":hit,"n":n,
            "cov_asset_days":round(total/cells,4),"cov_dates_ge8":round(ge8/len(fdf),4),
            "turnover_10d":round(to,4),"decay_ic":res}

def regime_split(fdf, close, macro):
    ics = compute_ics(fdf, close, HORIZON)
    ics_by = {}
    regs = {"2020-2021":(START,"2021-12-31"),"2022-2023":("2022-01-01","2023-12-31"),
            "2024-2025":("2024-01-01","2025-12-31"),"2026":("2026-01-01","2026-12-31"),
            "2027":("2027-01-01","2027-12-31"),"2028":("2028-01-01","2028-12-31"),
            "2029":("2029-01-01","2029-12-31"),"2030":("2030-01-01","2030-12-31"),
            "2031-YTD":("2031-01-01","2031-11-26"),"recent1y":("2030-11-26","2031-11-26")}
    for lbl,(lo,hi) in regs.items():
        sub=[x[1] for x in ics if pd.Timestamp(lo)<=x[0]<=pd.Timestamp(hi)]
        if len(sub)>=30:
            s=np.array(sub)
            ics_by_ic.append(ics_by_ic)
    # placeholder pattern
    return None

close, vol, open_, high, low = load_panel()
macro = load_macro()
print("data rows per asset (<= visible):", {a:int(close[a].notna().sum()) for a in ASSETS})
print("macro VIX last:", float(macro["VIX"].iloc[-1]), "DXY last:", float(macro["DXY"].iloc[-1]))
print("SPX close last:", float(close["SPX"].dropna().iloc[-1]))

# Candidate factor definitions
def f_mom30_skip10(c,v,o,h,l,m):
    return c.shift(10)/c.shift(40)-1.0
def f_mom60_skip20(c,v,o,h,l,m):
    return c.shift(20)/c.shift(80)-1.0
def f_drawdown_20(c,v,o,h,l,m):
    return h.rolling(20).max()/c-1.0
def f_range_20(c,v,o,h,l,m):
    return (h.