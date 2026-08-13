"""miner_2 revalidation (2031-04-07) - revalidate 3 currently effective factors.

Checks |IC|>=0.0070 and |ICIR|>=0.0840 at h=10 on:
  (a) full window 2020..visible-through
  (b) recent 2y
  (c) recent 1y
Also computes decay, coverage, turnover, pairwise library correlation.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

panels = load_panels(days=3300)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)

# data availability probe
print("=== data probe (last close dates per asset) ===", flush=True)
for a in TRADABLE:
    if a in panels:
        print(f"{a:10s} rows={len(panels[a]):5d} last={panels[a].index[-1].date()} last_close={panels[a]['close'].iloc[-1]:.4f}", flush=True)
    else:
        print(f"{a:10s} MISSING", flush=True)
for m in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    if m in panels:
        print(f"{m:10s} rows={len(panels[m]):5d} last={panels[m].index[-1].date()}", flush=True)
    else:
        print(f"{m:10s} MISSING", flush=True)

def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)

lib = {}
lib["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
lib["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, panels["CN10Y"]["close"].pct_change(), 60)

H = 10
fwd = forward_returns(closes, H)

def eval_window(name, panel, fwd, tag, expected_sign):
    ics = rank_ic_series(panel, fwd)
    if len(ics) < 60:
        print(f"{name} [{tag}] SKIP n={len(ics)}", flush=True)
        return None
    m = summarize_ic(ics, expected_sign)
    ic, icir = m["ic"], m["icir"]
    gate = (abs(ic) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"{name:26s} [{tag:12s}] IC={ic:+.4f} ICIR={icir:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} GATE={'PASS' if gate else 'FAIL'}", flush=True)
    return m

print("\n=== revalidation h=10 (visible through latest) ===", flush=True)
for name, panel in lib.items():
    exp_sign = 1 if name != "rate_beta_cn10y_60d" else -1
    m_full = eval_window(name, panel, fwd, "full", exp_sign)
    sub = fwd.iloc[-520:]
    m2y = eval_window(name, panel, sub, "recent2y", exp_sign)
    sub1 = fwd.iloc[-260:]
    m1y = eval_window(name, panel, sub1, "recent1y", exp_sign)

print("\n=== decay profiles (full window) ===", flush=True)
for name, panel in lib.items():
    dec = decay_profile(panel, closes, horizons=(1,2,3,5,10,20))
    print(name, dec, flush=True)

print("\n=== coverage / turnover ===", flush=True)
for name, panel in lib.items():
    cov = coverage_metrics(panel)
    to = turnover_rank(panel)
    print(f"{name}: coverage_asset_days={cov['coverage_asset_days']} dates_ge8={cov['coverage_dates_ge8']} turnover10d={to}", flush=True)
