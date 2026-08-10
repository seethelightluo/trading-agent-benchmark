"""miner_3 2026-07-30: exploration v6 - re-validate previous passers + new cross-asset ideas.

All validation at h=10, min_valid=8, on the 15-asset tradable universe,
visible window through 2026-07-29. Prints IC/ICIR/hit/coverage/turnover and
the admission gate flags (|IC|>=0.007, |ICIR|>=0.084).
Also computes pairwise mean-abs-Spearman rho of signal panels (the contract's
conflict metric) so we persist only non-redundant factors.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
ret = close.pct_change()
fr = forward_returns(close, H)
WATCH = list(close.columns)

GATE_IC, GATE_ICIR = 0.007, 0.084


def rolling_beta(a_ret, m_ret, win, minp=40):
    out = {}
    for a in a_ret.columns:
        pair = pd.concat([a_ret[a].rename("a"), m_ret.rename("m")], axis=1).dropna()
        b = pair["a"].rolling(win, min_periods=minp).cov(pair["m"]) / pair["m"].rolling(win, min_periods=minp).var()
        out[a] = b
    return pd.DataFrame(out).reindex(a_ret.index)


def pair_abs_spearman(left: pd.DataFrame, right: pd.DataFrame) -> float:
    """Mean cross-sectional abs Spearman rho over common dates (contract metric)."""
    common = left.index.intersection(right.index)
    vals = []
    for d in common:
        a, b = left.loc[d], right.loc[d]
        mask = a.notna() & b.notna()
        if mask.sum() < 3:
            continue
        va, vb = a[mask], b[mask]
        if va.nunique() < 2 or vb.nunique() < 2:
            continue
        vals.append(abs(va.corr(vb, method="spearman")))
    return float(np.mean(vals)) if vals else float("nan")


candidates = {}

# ---- re-validate previous passers (beta family + relative momentum) ----
candidates["ETH_BETA_60"] = rolling_beta(ret, ret["ETH"], 60)
candidates["WTI_BETA_60"] = rolling_beta(ret, ret["WTI"], 60)
candidates["BTC_BETA_60"] = rolling_beta(ret, ret["BTC"], 60)
mom20 = close / close.shift(20) - 1.0
candidates["MOM_REL_EQ_20"] = mom20 - mom20.mean(axis=1)  # cross-sectionally demeaned

# ---- new ideas ----
# 1) Gold-relative 20d momentum: asset momentum minus XAU momentum (risk rotation)
candidates["XAU_REL_MOM_20"] = mom20.sub(mom20["XAU"], axis=0)
# 2) 20d range position: (close - min20) / (max20 - min20)
lo20 = close.rolling(20, min_periods=12).min()
hi20 = close.rolling(20, min_periods=12).max()
candidates["RANGE_POS_20D"] = (close - lo20) / (hi20 - lo20)
# 3) 60d drawdown depth: close / rolling_max(close, 60) - 1
candidates["DD_60D"] = close / close.rolling(60, min_periods=40).max() - 1.0
# 4) 60d beta to gold returns (risk-off sensitivity)
candidates["XAU_BETA_60"] = rolling_beta(ret, ret["XAU"], 60)
# 5) volume expansion 5x60 (re-validate from earlier cycle)
frames = __import__("factor_validate", fromlist=["load_panel"]).load_panel(WATCH, "stock", VIS)
vol = pd.DataFrame({s: df.set_index("date")["volume"].astype(float) for s, df in frames.items()}).sort_index()
vol = vol.reindex(close.index).replace(0.0, np.nan)
candidates["VOL_RATIO_5X60"] = vol.rolling(5, min_periods=3).mean() / vol.rolling(60, min_periods=30).mean() - 1.0

# ---- validate all ----
results = {}
for fid, sig in candidates.items():
    sig = sig.reindex(close.index)
    ics = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ics, sig, fr, close, h=H)
    if m is None:
        print(f"{fid:20s} INSUFFICIENT IC dates n={len(ics)}")
        continue
    reg = regime_split(ics)
    gate = abs(m["ic"]) >= GATE_IC and abs(m["icir"]) >= GATE_ICIR
    flag = "*** PASS ***" if gate else ""
    print(f"{fid:20s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} cov={m['coverage_asset_days']:.2f} turn={m['turnover_10d_rank']:.3f} {flag}")
    print(f"    regime: " + ", ".join(f"{k}: IC={v['ic']:+.3f} ICIR={v['icir']:+.3f}" for k, v in reg.items()))
    results[fid] = {"sig": sig, "m": m, "reg": reg, "pass": gate}

print("\n=== pairwise mean-abs-Spearman rho of signal panels (passers only) ===")
passers = {k: v for k, v in results.items() if v["pass"]}
names = list(passers)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        rho = pair_abs_spearman(passers[names[i]]["sig"], passers[names[j]]["sig"])
        print(f"  {names[i]:18s} vs {names[j]:18s} rho={rho:.3f}")

with open("scripts/miner3_20260730_explore_v6_results.json", "w") as f:
    json.dump({k: {"ic": v["m"]["ic"], "icir": v["m"]["icir"],
                   "n": v["m"]["n_ic_dates"], "regime": v["reg"], "pass": v["pass"]}
               for k, v in results.items()}, f, indent=1, default=str)
print("\nsaved scripts/miner3_20260730_explore_v6_results.json")
