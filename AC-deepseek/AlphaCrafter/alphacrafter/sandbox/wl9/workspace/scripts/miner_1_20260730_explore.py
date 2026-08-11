"""Exploration: batch of candidate cross-asset factors, IC/ICIR at h=10."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_20260730_helpers import (WATCH, MACRO, load_panel, forward_returns,
                                      factor_ic_report, factor_turnover, coverage,
                                      decay_report)

closes = load_panel(WATCH)
rets = closes.pct_change()
macro = load_panel(MACRO)
mrets = macro.pct_change()

def beta_to(x, y, win):
    """Rolling beta of x to y."""
    xr = x.rename("x"); yr = y.rename("y")
    df = pd.concat([xr, yr], axis=1).dropna()
    cov = df["x"].rolling(win).cov(df["y"])
    var = df["y"].rolling(win).var()
    b = cov / var
    return b.reindex(x.index)

def skew(x, win):
    return x.rolling(win).skew()

def downside_dev(x, win):
    neg = x.clip(upper=0)
    return neg.rolling(win).apply(lambda v: np.sqrt(np.mean(v * v)), raw=True)

def max_dd(x, win):
    return (1 + x).rolling(win).apply(lambda v: np.prod(v) / np.maximum.accumulate(np.cumprod(np.r_[1, v]))[-1] - 1, raw=True)

candidates = {}
candidates["dxy_beta_60d"] = beta_to(rets, mrets["DXY"], 60)
candidates["rate_beta_60d"] = beta_to(rets, mrets["US10Y"], 60)
candidates["skew_20d"] = skew(rets, 20)
candidates["vol_ratio_5_60"] = rets.rolling(5).std() / rets.rolling(60).std()
candidates["downside_vol_60d"] = downside_dev(rets, 60)
candidates["trend_quality_60d"] = (rets > 0).rolling(60).mean()
candidates["range_pos_20d"] = ((closes - closes.rolling(20).min()) /
                               (closes.rolling(20).max() - closes.rolling(20).min()))
candidates["autocorr_10d"] = rets.rolling(10).apply(lambda v: pd.Series(v).autocorr() if len(v) > 3 else np.nan, raw=True)
candidates["vix_beta_plain_60d"] = beta_to(rets, mrets["VIX"], 60)
candidates["crypto_beta_60d"] = beta_to(rets, mrets["BTC"], 60)
candidates["max_drawdown_60d"] = max_dd(rets, 60)
candidates["mom60_vol_adj"] = rets.rolling(60).sum() / rets.rolling(60).std()
candidates["oil_beta_60d"] = beta_to(rets, mrets["WTI"], 60)
candidates["dxy_corr_change_20_60"] = (rets.rolling(20).corr(mrets["DXY"]) -
                                       rets.rolling(60).corr(mrets["DXY"]))

print(f"{'factor':<28}{'IC':>8}{'ICIR':>8}{'hit':>7}{'n_dates':>8}{'turn':>7}{'cov':>7}  decay1/3/5/20")
for name, f in candidates.items():
    h = 10
    fwd = forward_returns(rets, h)
    rep = factor_ic_report(f, fwd, horizon=h)
    if rep is None:
        print(f"{name:<28} insufficient data")
        continue
    turn = factor_turnover(f)
    cov = coverage(f)
    dec = decay_report(f, rets)
    print(f"{name:<28}{rep['ic']:>8.4f}{rep['icir']:>8.4f}{rep['ic_hit_ratio']:>7.3f}"
          f"{rep['n_ic_dates']:>8d}{turn:>7.2f}{cov['coverage_asset_days']:>7.2f}"
          f"  {dec['1']}/{dec['3']}/{dec['5']}/{dec['20']}")
