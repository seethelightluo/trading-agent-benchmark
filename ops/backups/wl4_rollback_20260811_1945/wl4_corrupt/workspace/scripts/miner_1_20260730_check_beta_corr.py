"""Check pairwise correlation among batch-3a passing beta candidates (2026-07-30)."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, library_signals)

panels = load_panels()
closes = close_panel(panels)
rets = closes.pct_change()
lib = library_signals(panels, closes, rets)


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        b = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
        beta[a] = b
    return pd.DataFrame(beta, index=asset_ret.index)


mkt = rets.mean(axis=1)
up = mkt.where(mkt > 0).fillna(0.0)
dn = mkt.where(mkt < 0).fillna(0.0)
cands = {
    "eurusd_beta_60d": rolling_beta(rets, panels["EURUSD"]["close"].astype(float).pct_change(), 60),
    "rate_beta_cn10y_60d": rolling_beta(rets, panels["CN10Y"]["close"].astype(float).pct_change(), 60),
    "dxy_beta_60d": rolling_beta(rets, panels["DXY"]["close"].astype(float).pct_change(), 60),
    "dn_mkt_beta_60d": rolling_beta(rets, dn, 60),
    "btc_beta_60d": rolling_beta(rets, panels["BTC"]["close"].astype(float).pct_change(), 60),
    "up_mkt_beta_60d": rolling_beta(rets, up, 60),
}

stacked = {k: v.stack().rename(k) for k, v in cands.items()}
panel = pd.concat(stacked, axis=1)
corr = panel.corr().abs()
print("=== pairwise |corr| among passing beta candidates ===")
print(corr.round(3).to_string())

# also vs library
for k, v in cands.items():
    for lk, lv in lib.items():
        both = pd.concat([v.stack().rename("c"), lv.stack().rename("l")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = both["c"].corr(both["l"])
        if abs(r) > 0.35:
            print(f"{k} vs {lk}: {r:.3f}")
