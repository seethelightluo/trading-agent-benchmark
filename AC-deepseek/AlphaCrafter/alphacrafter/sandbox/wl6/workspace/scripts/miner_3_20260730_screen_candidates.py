"""Exploration: screen candidate factor families on the 15-asset universe.
Data truncated at 2026-07-30. Horizon-10 rank IC preview.
Macro series reindexed to asset calendar with ffill (stale-quote convention).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_utils import (load_close, load_panel, forward_returns,
                          rank_ic_series, summarize_ic, DATA_DIR, INDEX_DIR)

px, vol = load_panel()
ret = px.pct_change()
fwd10 = forward_returns(px, 10)

def macro_close(name):
    df = load_close(name, INDEX_DIR)
    return df["close"].astype(float).reindex(px.index).ffill()

dxy = macro_close("DXY")
vix = macro_close("VIX")
usdjpy = macro_close("USDJPY")
usdcny = macro_close("USDCNY")
eurusd = macro_close("EURUSD")

def beta_of(asset_ret, mkt_ret, win):
    a = asset_ret.rolling(win).cov(mkt_ret)
    b = mkt_ret.rolling(win).var()
    return a / b

candidates = {}
candidates["dxy_beta_60d"] = beta_of(ret, dxy.pct_change(), 60)
candidates["dxy_beta_120d"] = beta_of(ret, dxy.pct_change(), 120)
candidates["usdjpy_beta_60d"] = beta_of(ret, usdjpy.pct_change(), 60)
candidates["usdcny_beta_60d"] = beta_of(ret, usdcny.pct_change(), 60)
candidates["eurusd_beta_60d"] = beta_of(ret, eurusd.pct_change(), 60)
candidates["us10y_beta_60d"] = beta_of(ret, px["US10Y"].pct_change(), 60)
candidates["cn10y_beta_60d"] = beta_of(ret, px["CN10Y"].pct_change(), 60)

clv = {}
for a in px.columns:
    df = load_close(a)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv[a] = ((df["close"] - df["low"]) / rng).rolling(20).mean()
candidates["clv_20d"] = pd.DataFrame(clv).reindex(px.index)

ami = {}
for a in px.columns:
    v = vol[a].replace(0, np.nan)
    ami[a] = np.log((ret[a].abs() / v)).rolling(20).mean()
candidates["amihud_illiq_20d"] = pd.DataFrame(ami).reindex(px.index)

candidates["skew_60d"] = ret.rolling(60).skew()

def ac1(x):
    if len(x) < 6:
        return np.nan
    s = pd.Series(x)
    return s.autocorr(1)
candidates["autocorr_5d"] = ret.rolling(6).apply(ac1, raw=False)

candidates["vol_trend_20x60"] = vol.rolling(20).mean() / vol.rolling(60).mean()

rng_panel = {}
for a in px.columns:
    df = load_close(a)
    rng_panel[a] = (df["high"] - df["low"])
rng = pd.DataFrame(rng_panel).reindex(px.index)
candidates["range_ratio_20x60"] = rng.rolling(20).mean() / rng.rolling(60).mean()

wealth = (1 + ret).cumprod()
dd = wealth / wealth.cummax() - 1.0
candidates["mdd_60d"] = dd.rolling(60).min()

candidates["risk_adj_mom_20d"] = ret.rolling(20).sum() / ret.rolling(20).std()

dxy20 = dxy.pct_change(20)
candidates["dxy_cond_60x20"] = -beta_of(ret, dxy.pct_change(), 60) * dxy20

print(f"{'factor':<24}{'ic':>8}{'icir':>8}{'hit':>7}{'n':>6}  gate")
for name, f in candidates.items():
    f = f.reindex(px.index)
    s = rank_ic_series(f, fwd10)
    res = summarize_ic(s, name, 10)
    flag = "PASS" if res["pass_gate"] else ""
    print(f"{name:<24}{res['ic']:>8.4f}{res['icir']:>8.4f}{res['ic_hit_ratio']:>7.3f}{res['n_ic_dates']:>6d}  {flag}")
