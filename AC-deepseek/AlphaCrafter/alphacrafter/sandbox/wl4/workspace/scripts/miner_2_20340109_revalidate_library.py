"""miner_2 2034-01-09: Re-validate the 3 currently EFFECTIVE library factors
on full history 2020-01-01..2034-01-06 (visible through previous completed day)."""
import sys, warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, ret_panel, forward_returns,
    rank_ic_series, summarize_ic, coverage_metrics, turnover_rank, decay_profile,
)

panels = load_panels(days=6000)
closes = close_panel(panels)
rets = closes.pct_change()
fwd10 = forward_returns(closes, 10)

# ---------- 1. vol_adj_mom_accel_20x60 ----------
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
vol20 = rets.rolling(20).std()
sig1 = (mom20 - mom60) / vol20

# ---------- 2. dn_mkt_beta_60d ----------
mkt_ret = rets.mean(axis=1)
down = mkt_ret.where(mkt_ret < 0)
beta_down = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), down.rename("m")], axis=1).dropna()
    cov = z["a"].rolling(60).cov(z["m"])
    var = z["m"].rolling(60).var()
    beta_down[a] = cov / var
sig2 = pd.DataFrame(beta_down, index=rets.index)

# ---------- 3. rate_beta_cn10y_60d ----------
cn10y = panels["CN10Y"]["close"].astype(float)
cn10y_ret = cn10y.pct_change()
beta_cn = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), cn10y_ret.rename("r")], axis=1).dropna()
    cov = z["a"].rolling(60).cov(z["r"])
    var = z["r"].rolling(60).var()
    beta_cn[a] = cov / var
sig3 = pd.DataFrame(beta_cn, index=rets.index)

def eval_factor(name, sig, expected_sign, window=None):
    s = sig if window is None else sig.loc[window[0]:window[1]]
    c = closes if window is None else closes.loc[window[0]:window[1]]
    fwd = forward_returns(c, 10)
    ics = rank_ic_series(s, fwd, 8)
    m = summarize_ic(ics, expected_sign)
    m.update(coverage_metrics(s, min_valid=8))
    m["turnover_10d_rank"] = turnover_rank(s, 10)
    m["decay_ic_by_horizon"] = decay_profile(s, c, (1,2,3,5,10,20), 8, expected_sign)
    m["admission_gate"] = {
        "ic_gate_abs": 0.0070, "icir_gate_abs": 0.0840,
        "ic_pass": abs(m["ic"]) >= 0.0070,
        "icir_pass": abs(m["icir"]) >= 0.0840,
    }
    print(f"=== {name} (expected dir {expected_sign:+d}) ===")
    print(json.dumps(m, indent=1))
    return m

print("=" * 70)
print("FULL-HISTORY REVALIDATION 2020-01-01..2034-01-06")
print("=" * 70)
r1 = eval_factor("vol_adj_mom_accel_20x60", sig1, 1)
r2 = eval_factor("dn_mkt_beta_60d", sig2, 1)
r3 = eval_factor("rate_beta_cn10y_60d", sig3, -1)

print("=" * 70)
print("RECENT-3Y REVALIDATION 2031-01-01..2034-01-06 (drift check)")
print("=" * 70)
r1r = eval_factor("vol_adj_mom_accel_20x60", sig1, 1, ("2031-01-01", "2034-01-06"))
r2r = eval_factor("dn_mkt_beta_60d", sig2, 1, ("2031-01-01", "2034-01-06"))
r3r = eval_factor("rate_beta_cn10y_60d", sig3, -1, ("2031-01-01", "2034-01-06"))

print("=" * 70)
print("RECENT-1Y REVALIDATION 2033-01-09..2034-01-06 (timeliness)")
print("=" * 70)
r1y = eval_factor("vol_adj_mom_accel_20x60", sig1, 1, ("2033-01-09", "2034-01-06"))
r2y = eval_factor("dn_mkt_beta_60d", sig2, 1, ("2033-01-09", "2034-01-06"))
r3y = eval_factor("rate_beta_cn10y_60d", sig3, -1, ("2033-01-09", "2034-01-06"))
