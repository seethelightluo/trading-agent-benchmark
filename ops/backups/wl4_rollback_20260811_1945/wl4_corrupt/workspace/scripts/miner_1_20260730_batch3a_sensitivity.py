"""miner_1 batch-3a factor exploration: macro-sensitivity / beta family (2026-07-30).

One idea family per script: rolling beta of each tradable asset's returns on
macro drivers (US10Y, CN10Y, USDJPY, EURUSD, XAU, BTC) and on market up/down
states. Evaluates rank-IC at h=10 against forward returns on the 15-asset
cross-asset universe. Admission gates: |IC|>=0.007, |ICIR|>=0.084.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, library_signals,
                                 max_library_corr, TRADABLE, MACRO)

panels = load_panels()
closes = close_panel(panels)
rets = closes.pct_change()
fwd10 = forward_returns(closes, 10)
lib = library_signals(panels, closes, rets)

print(f"Panel: {closes.shape[0]} dates x {closes.shape[1]} assets; "
      f"{closes.index[0].date()} .. {closes.index[-1].date()}")


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        b = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
        beta[a] = b
    return pd.DataFrame(beta, index=asset_ret.index)


cands = {}
# --- rate sensitivity: beta on 10Y yield changes (both US and CN) ---
for m, tag in (("US10Y", "us10y"), ("CN10Y", "cn10y")):
    mret = panels[m]["close"].astype(float).pct_change()
    cands[f"rate_beta_{tag}_60d"] = rolling_beta(rets, mret, 60)
    # interaction with rate direction: beta * 20d rate change (conditional)
    mmom = (panels[m]["close"].astype(float) / panels[m]["close"].astype(float).shift(20) - 1.0)
    cands[f"rate_beta_{tag}_cond_60x20"] = -rolling_beta(rets, mret, 60) * mmom

# --- FX sensitivity ---
for m, tag in (("USDJPY", "usdjpy"), ("EURUSD", "eurusd"), ("DXY", "dxy")):
    mret = panels[m]["close"].astype(float).pct_change()
    cands[f"{tag}_beta_60d"] = rolling_beta(rets, mret, 60)

# --- cross-asset beta: gold and crypto as drivers ---
cands["gold_beta_60d"] = rolling_beta(rets, panels["XAU"]["close"].astype(float).pct_change(), 60)
cands["btc_beta_60d"] = rolling_beta(rets, panels["BTC"]["close"].astype(float).pct_change(), 60)

# --- market up/down beta ---
mkt = rets.mean(axis=1)
up = mkt.where(mkt > 0)
dn = mkt.where(mkt < 0)
cands["up_mkt_beta_60d"] = rolling_beta(rets, up.fillna(0.0), 60)
cands["dn_mkt_beta_60d"] = rolling_beta(rets, dn.fillna(0.0), 60)
# beta asymmetry: downside beta - upside beta
ub = rolling_beta(rets, up.fillna(0.0), 60)
db = rolling_beta(rets, dn.fillna(0.0), 60)
cands["beta_asym_60d"] = db - ub

# --- evaluate all ---
rows = []
for name, panel in cands.items():
    ics = rank_ic_series(panel, fwd10, 8)
    if len(ics) < 200:
        print(f"{name}: skipped (n_ic_dates={len(ics)})")
        continue
    m = summarize_ic(ics, 1)
    m.update(coverage_metrics(panel))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    corr, key = max_library_corr(panel, lib)
    m["max_lib_corr"], m["max_corr_key"] = corr, key
    m["name"] = name
    rows.append(m)
    flag = "PASS" if (abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084 and corr < 0.5) else ""
    print(f"[{flag}] {name}: IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} covA={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.2f} libcorr={corr:.3f}({key})")

df = pd.DataFrame(rows).set_index("name")
print("\n=== SUMMARY (sorted by |ICIR|) ===")
print(df[["ic", "icir", "ic_hit_ratio", "n_ic_dates", "coverage_asset_days",
          "coverage_dates_ge8", "turnover_10d_rank", "max_lib_corr", "max_corr_key"]]
      .sort_values("icir", key=lambda s: s.abs(), ascending=False).to_string())
