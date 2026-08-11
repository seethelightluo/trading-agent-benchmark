import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from factor_validation_lib import ASSETS, load_closes, load_index, factor_panel

close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]}
macro["__wti_ret__"] = close["WTI"].pct_change()
macro["__xau_ret__"] = close["XAU"].pct_change()
macro["__mkt_ret__"] = close.pct_change().mean(axis=1)  # proper equal-weight market return

def wti_beta_60(c, v, o, h, l, m, win=60):
    mkt = m["__wti_ret__"].reindex(c.index)
    r = c.pct_change()
    cov = r.rolling(win).cov(mkt)
    var = mkt.rolling(win).var()
    return cov / var

def wti_beta_60b(c, v, o, h, l, m, win=60):
    """manual beta via corr * std ratio (more robust)"""
    mkt = m["__wti_ret__"].reindex(c.index)
    r = c.pct_change()
    corr = r.rolling(win).corr(mkt)
    b = corr * r.rolling(win).std() / mkt.rolling(win).std()
    return b

def market_corr_60(c, v, o, h, l, m, win=60):
    mkt = m["__mkt_ret__"].reindex(c.index)
    r = c.pct_change()
    return r.rolling(win).corr(mkt)

# quick standalone debug on one asset
c = close["SPX"].dropna()
mkt = macro["__wti_ret__"].reindex(c.index)
print("SPX n=", len(c), "wti_ret non-nan:", mkt.notna().sum())
r = c.pct_change()
print("cov tail:\n", r.rolling(60).cov(mkt).tail(3))
print("var tail:\n", mkt.rolling(60).var().tail(3))

p1 = factor_panel(wti_beta_60, close, vol, open_, high, low, macro)
print("wti_beta_60 panel non-nan:", int(p1.notna().sum().sum()), "/", p1.size)
p2 = factor_panel(wti_beta_60b, close, vol, open_, high, low, macro)
print("wti_beta_60b panel non-nan:", int(p2.notna().sum().sum()), "/", p2.size)
p3 = factor_panel(market_corr_60, close, vol, open_, high, low, macro)
print("market_corr_60(proper) non-nan:", int(p3.notna().sum().sum()), "/", p3.size)
print(p3.tail(2))
