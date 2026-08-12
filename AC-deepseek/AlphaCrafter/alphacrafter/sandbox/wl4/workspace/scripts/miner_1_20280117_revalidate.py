"""miner_1 2028-01-17: revalidate 3 active factors on data through 2028-01-14.

Rank IC h=10 on the 15-asset cross-asset universe.
Gates (benchmark-wide): |IC|>=0.0070, |ICIR|>=0.0840; library corr < 0.5.
"""
import json
import numpy as np
import pandas as pd

VISIBLE = pd.Timestamp('2028-01-14')
IC_TH, ICIR_TH = 0.0070, 0.0840
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

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

def load_macro():
    frames = {}
    for m in MACRO:
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
        frames[m] = df["close"]
    return pd.DataFrame(frames)

def fwd_returns(panel, h=10):
    return panel.shift(-h) / panel - 1.0

def rank_ic_series(factor, fwd, min_valid=8):
    dates = factor.index.intersection(fwd.index)
    ics = {}
    for dt in dates:
        f = factor.loc[dt]; r = fwd.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_valid:
            continue
        ic = f[mask].corr(r[mask], method="spearman")
        if np.isfinite(ic):
            ics[dt] = ic
    return pd.Series(ics, name="ic")

def evaluate(factor, panel, h=10, min_valid=8, label="f", valid_from=None, valid_to=None):
    fwd = fwd_returns(panel, h=h)
    ic = rank_ic_series(factor, fwd, min_valid=min_valid)
    if valid_from is not None:
        ic = ic[ic.index >= pd.Timestamp(valid_from)]
    if valid_to is not None:
        ic = ic[ic.index <= pd.Timestamp(valid_to)]
    out = {"label": label, "n_ic_dates": len(ic)}
    if len(ic) == 0:
        out.update(ic=np.nan, icir=np.nan, ic_std=np.nan, ic_hit=np.nan)
        return out
    icm = ic.mean(); ics = ic.std(ddof=1)
    out.update(ic=float(icm), icir=float(icm/ics) if ics > 0 else np.nan,
               ic_std=float(ics), ic_hit=float((ic > 0).mean()))
    ge8 = (factor.notna().sum(axis=1) >= min_valid).mean()
    out["cov_ge8"] = float(ge8)
    ranks = factor.rank(axis=1)
    out["turnover_10d"] = float((ranks - ranks.shift(10)).abs().mean().mean())
    return out

def rolling_beta(ret, bench, win=60, minp=40):
    cov = ret.rolling(win, min_periods=minp).cov(bench)
    var = bench.rolling(win, min_periods=minp).var()
    return cov / var

panel = load_close()
ohlcv = load_ohlcv()
macro = load_macro()
rets = panel.pct_change()
mkt = rets.mean(axis=1)
print(f"Data: {panel.shape[0]} days, {panel.shape[1]} assets, through {VISIBLE.date()}")

# ---------- ACTIVE FACTORS ----------
active = {}
down_mkt = mkt.where(mkt < 0, 0.0)
active["dn_mkt_beta_60d"] = rolling_beta(rets, down_mkt, 60, 40)
vols = {a: ohlcv[a]["volume"] for a in ASSETS}
vol_df = pd.DataFrame(vols)
active["vol_price_corr_20"] = rets.rolling(20, min_periods=10).corr(vol_df)
cn10y_ret = panel["CN10Y"].pct_change()
active["rate_beta_cn10y_60d"] = rolling_beta(rets, cn10y_ret, 60, 40)

print("\n=== ACTIVE FACTOR REVALIDATION (through 2028-01-14) ===")
for fid, f in active.items():
    full = evaluate(f, panel, label=fid)
    recent = evaluate(f, panel, label=fid+"_recent250", valid_from=panel.index[-251])
    recent500 = evaluate(f, panel, label=fid+"_recent500", valid_from=panel.index[-501])
    print(f"{fid}: full ic={full['ic']:.4f} icir={full['icir']:.4f} hit={full['ic_hit']:.3f} "
          f"n={full['n_ic_dates']} cov8={full['cov_ge8']:.3f} turn={full['turnover_10d']:.2f}")
    print(f"   recent250: ic={recent['ic']:.4f} icir={recent['icir']:.4f} n={recent['n_ic_dates']}")
    print(f"   recent500: ic={recent500['ic']:.4f} icir={recent500['icir']:.4f} n={recent500['n_ic_dates']}")

json.dump({fid: {"full": {k: (None if not np.isfinite(v) if isinstance(v, float) else v) for k, v in evaluate(f, panel, label=fid).items()}}
           for fid, f in active.items()},
          open("scripts/_miner1_20280117_reval.json", "w"), indent=1, default=str)
print("\nsaved scripts/_miner1_20280117_reval.json")
