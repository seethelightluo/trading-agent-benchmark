"""miner_3 2027-01-18 focused validation of batch H shortlist.

Shortlist (good coverage + low library corr from batch H screen):
  mom_20d_skip5   (+1) momentum 20d skipping last 5d
  mom_60d_skip20  (-1) medium-term momentum 60d skipping last 20d (observed reversal)
  btc_beta_60d    (+1) 60d rolling beta vs BTC returns
  vix_beta_60d    (-1) 60d rolling beta vs VIX returns (defensive)

Computes full validation metrics at h=10 + sub-period splits + decay + library corr.
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
vol_panel = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns}).reindex(closes.index)
mkt = rets.mean(axis=1)
H_ADM, MIN_VALID = 10, 8

def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        beta[a] = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)

# library signals
dn = mkt.where(mkt < 0)
eur_ret = panels["EURUSD"]["close"].pct_change()
cn_ret = rets["CN10Y"]
LIBRARY = {
    "vol_price_corr_20": pd.DataFrame({a: rets[a].rolling(20).corr(vol_panel[a]) for a in closes.columns}, index=rets.index),
    "dn_mkt_beta_60d": rolling_beta(rets, dn, 60, 40),
    "eurusd_beta_60d": rolling_beta(rets, eur_ret, 60, 40),
    "rate_beta_cn10y_60d": rolling_beta(rets, cn_ret, 60, 40),
}

cands = {
    "mom_20d_skip5": (closes.shift(5) / closes.shift(25) - 1.0, 1),
    "mom_60d_skip20": (closes.shift(20) / closes.shift(80) - 1.0, -1),
    "btc_beta_60d": (rolling_beta(rets, rets["BTC"], 60, 40), 1),
    "vix_beta_60d": (rolling_beta(rets, panels["VIX"]["close"].pct_change(), 60, 40), -1),
}

fwd = forward_returns(closes, H_ADM)
out = {}
for name, (panel, exp) in cands.items():
    ics = rank_ic_series(panel, fwd, MIN_VALID)
    m = summarize_ic(ics, exp)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, (1, 2, 3, 5, 10, 20), MIN_VALID, exp)
    # library corr
    best, best_key = 0.0, None
    c = panel.stack()
    for ln, ls in LIBRARY.items():
        both = pd.concat([c.rename("cand"), ls.stack().rename("lib")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both["cand"].corr(both["lib"]))
        if abs(r) > best:
            best, best_key = abs(r), ln
    m["max_abs_library_correlation"] = round(best, 4)
    m["max_corr_factor"] = best_key
    # sub-period splits (3 equal parts of IC series)
    n = len(ics)
    third = n // 3
    splits = {}
    labels = ["early", "mid", "late"]
    for i, lab in enumerate(labels):
        seg = ics.iloc[i * third:(i + 1) * third if i < 2 else n]
        s = summarize_ic(seg, exp)
        splits[lab] = {"ic": s["ic"], "icir": s["icir"], "n": s["n_ic_dates"]}
    m["sub_period_ic"] = splits
    m["n_ic_dates"] = m["n_ic_dates"]
    out[name] = m
    print(f"=== {name} (exp {exp:+d}) ===", flush=True)
    print(f"  IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} ic_std={m['ic_std']}", flush=True)
    print(f"  coverage_asset_days={m['coverage_asset_days']} coverage_dates_ge8={m['coverage_dates_ge8']} turnover={m['turnover_10d_rank']}", flush=True)
    print(f"  max_lib_corr={m['max_abs_library_correlation']} ({m['max_corr_factor']})", flush=True)
    print(f"  decay={ {k: round(v,4) for k,v in m['decay_ic_by_horizon'].items()} }", flush=True)
    print(f"  sub_period: early {splits['early']} | mid {splits['mid']} | late {splits['late']}", flush=True)

with open("scripts/_miner3_batchH_shortlist.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "decay_ic_by_horizon" or True} for k, v in out.items()}, f, indent=1, default=str)
print(f"\nrun time {time.time()-t0:.1f}s")
