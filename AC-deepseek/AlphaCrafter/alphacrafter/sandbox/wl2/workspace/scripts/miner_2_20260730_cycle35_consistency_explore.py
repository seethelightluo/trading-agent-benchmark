"""miner_2 cycle35: return-path consistency & streak family.

Rationale: library covers level momentum (mom20_volproxy60), vol, macro-beta,
intraday path (intraday_drift_20), recovery freshness (days_since_high_60) and
gain/loss asymmetry (gain_loss_20). Untested here: the *consistency* of daily
returns -- win-rate (fraction of up days), longest consecutive losing/winning
streaks, and the short-term acceleration gap (recent 5d momentum minus 20d
momentum). These capture behavioral persistence/distribution of daily outcomes
rather than cumulative magnitude.

Candidates (one family, validated separately):
  - pos_freq_20       : fraction of positive daily returns over 20d
  - max_consec_loss_20: longest run of consecutive down days over 20d
  - max_consec_gain_20: longest run of consecutive up days over 20d
  - accel_5x20        : pct_change(5) - pct_change(20) (recent acceleration)

Admission gates: abs(IC)>=0.0070, abs(ICIR)>=0.0840 @10d horizon.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import (load_close_panel, per_asset, compute_ic,
                        forward_returns, validate_factor, regime_breakdown,
                        report)

close = load_close_panel()
print(f"panel {close.shape}, last date {close.index[-1].date()}")

idx = close.index
EFF = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20",
       "gain_loss_20", "intraday_drift_20", "usdjpy_beta_cond_120x60",
       "downside_dev_60", "days_since_high_60"]
lib = {}
for e in EFF:
    p = Path("factors") / f"{e}.signal.npy"
    if p.exists():
        a = np.load(p)
        if a.shape[0] == len(idx):
            lib[e] = pd.DataFrame(a, index=idx, columns=close.columns)
print(f"[lib] loaded {len(lib)} artifacts")

fwd = {str(h): forward_returns(close, h) for h in (1, 2, 3, 5, 10, 20)}


def pos_freq(s, w=20, mp=10):
    def _f(x):
        x = np.asarray(x, dtype=float)
        r = np.diff(x) / x[:-1]
        if len(r) < mp:
            return np.nan
        return float(np.mean(r > 0))
    return s.rolling(w + 1, min_periods=mp + 1).apply(_f, raw=True)


def max_consec_loss(s, w=20, mp=10):
    def _m(x):
        x = np.asarray(x, dtype=float)
        r = np.diff(x) / x[:-1]
        if len(r) < mp:
            return np.nan
        best = cur = 0
        for v in r:
            cur = cur + 1 if v < 0 else 0
            best = max(best, cur)
        return float(best)
    return s.rolling(w + 1, min_periods=mp + 1).apply(_m, raw=True)


def max_consec_gain(s, w=20, mp=10):
    def _m(x):
        x = np.asarray(x, dtype=float)
        r = np.diff(x) / x[:-1]
        if len(r) < mp:
            return np.nan
        best = cur = 0
        for v in r:
            cur = cur + 1 if v > 0 else 0
            best = max(best, cur)
        return float(best)
    return s.rolling(w + 1, min_periods=mp + 1).apply(_m, raw=True)


def accel(s, w1=5, w2=20, mp=10):
    r = s.pct_change()
    m1 = r.rolling(w1, min_periods=3).mean()
    m2 = r.rolling(w2, min_periods=mp).mean()
    return m1 - m2


cands = {
    "pos_freq_20": per_asset(close, pos_freq),
    "max_consec_loss_20": per_asset(close, max_consec_loss),
    "max_consec_gain_20": per_asset(close, max_consec_gain),
    "accel_5x20": per_asset(close, accel),
}

print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, close, library=lib, fwd_cache=fwd)
    p = report(name, m)
    print("    decay:", m["decay_ic_by_horizon"])
    print("    pairwise:", m.get("library_pairwise_corr"))
    print()
    results[name] = {"metrics": m, "pass": p}

print("=== REGIME BREAKDOWN (10d IC) ===")
for name, f in cands.items():
    ic_ser = compute_ic(f, fwd["10"]).dropna()
    reg = regime_breakdown(ic_ser)
    print(f"  {name:20s} | " + " | ".join(
        f"{k}: ic={v['ic']:+.4f} icir={v['icir']:+.3f} n={v['n_dates']}"
        for k, v in reg.items()))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/_miner2_cycle35_consistency_results.json", "w"), indent=1, default=float)
print("DONE")
