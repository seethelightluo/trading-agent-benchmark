"""miner_2 cycle33: tail-risk & path-recovery family.

Rationale: the library already covers level momentum (mom20_volproxy60),
low-vol (calmness_20), macro-beta (dxy/usdjpy/vix cond), intraday path
structure (intraday_drift_20, gain_loss_20) and downside deviation
(downside_dev_60, currently DEPRECATED).  What has NOT been systematically
mined yet in this cross-asset universe is (a) tail fatness (excess kurtosis,
incl. downside-only), and (b) the recovery / underwater geometry of the price
path (fraction of time below highs, depth of underwater, days since last
high), plus (c) trend significance from an OLS fit (slope t-stat), which
measures trend quality rather than raw level momentum.

These are crash-risk / trend-quality signals with different construction from
existing library factors, so library correlation is expected to be moderate.

Candidates (one family, validated separately):
  - kurt_60          : excess kurtosis of daily returns, 60d (tail fatness)
  - kurt_down_60     : excess kurtosis of NEGATIVE daily returns, 60d (downside tail)
  - underwater_60    : fraction of days below running max in trailing 60d
  - dd_depth_60      : mean drawdown depth (close/runmax - 1) in trailing 60d
  - days_since_high_60: days since last 60d high
  - trend_t_60       : OLS t-stat of log-price slope over 60d (trend significance)

Admission gates (shared): abs(IC) >= 0.0070, abs(ICIR) >= 0.0840 on 10d horizon,
15-instrument cross-section, dates with >=8 valid instruments.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import (load_close_panel, compute_ic, forward_returns,
                        validate_factor, regime_breakdown, report, per_asset)

panel = load_close_panel()
print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets "
      f"(last date {panel.index[-1].date()})")

# ---- effective library signal artifacts (real, persisted) ----
EFF_ARTIFACTS = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20",
                 "gain_loss_20", "intraday_drift_20", "usdjpy_beta_cond_120x60",
                 "downside_dev_60"]
lib = {}
idx = panel.index
for fid in EFF_ARTIFACTS:
    p = Path("factors") / f"{fid}.signal.npy"
    if not p.exists():
        print(f"[lib] MISSING artifact {fid}")
        continue
    a = np.load(p)
    if a.shape[0] == len(idx):
        lib[fid] = pd.DataFrame(a, index=idx, columns=panel.columns)
    else:
        print(f"[lib] shape mismatch {fid}: {a.shape}")
print(f"[lib] loaded {len(lib)} artifacts: {list(lib)}")

fwd = {str(h): forward_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}


# ---------- factor definitions ----------
def kurt_60(s, w=60, mp=40):
    return s.pct_change().rolling(w, min_periods=mp).kurt()


def kurt_down_60(s, w=60, mp=40):
    def _k(x):
        x = np.asarray(x, dtype=float)
        r = np.diff(x) / x[:-1]
        rn = r[r < 0]
        if len(rn) < 10:
            return np.nan
        mu, sd = rn.mean(), rn.std()
        if sd == 0:
            return np.nan
        return float(np.mean((rn - mu) ** 4) / sd ** 4 - 3.0)
    return s.rolling(w, min_periods=mp).apply(_k, raw=True)


def underwater_60(s, w=60, mp=40):
    def _frac(x):
        x = np.asarray(x, dtype=float)
        if len(x) < mp:
            return np.nan
        cmax = np.maximum.accumulate(x)
        return float(np.mean(x < cmax))
    return s.rolling(w, min_periods=mp).apply(_frac, raw=True)


def dd_depth_60(s, w=60, mp=40):
    def _dd(x):
        x = np.asarray(x, dtype=float)
        if len(x) < mp:
            return np.nan
        cmax = np.maximum.accumulate(x)
        return float(np.mean(x / cmax - 1.0))
    return s.rolling(w, min_periods=mp).apply(_dd, raw=True)


def days_since_high_60(s, w=60, mp=40):
    def _d(x):
        x = np.asarray(x, dtype=float)
        if len(x) < mp:
            return np.nan
        cmax = np.max(x)
        idx = np.where(np.isclose(x, cmax, rtol=1e-6))[0]
        return float(len(x) - 1 - idx[-1]) if len(idx) else np.nan
    return s.rolling(w, min_periods=mp).apply(_d, raw=True)


def trend_t_60(s, w=60, mp=40):
    def _t(x):
        x = np.asarray(x, dtype=float)
        if len(x) < mp or (x <= 0).any():
            return np.nan
        y = np.log(x)
        t = np.arange(len(x))
        n = len(x)
        tbar = t.mean()
        den = np.sum((t - tbar) ** 2)
        if den == 0:
            return np.nan
        b1 = np.sum((t - tbar) * (y - y.mean())) / den
        b0 = y.mean() - b1 * tbar
        resid = y - (b0 + b1 * t)
        s2 = np.sum(resid ** 2) / (n - 2)
        se = np.sqrt(s2 / den)
        return float(b1 / se) if se > 0 else np.nan
    return s.rolling(w, min_periods=mp).apply(_t, raw=True)


cands = {
    "kurt_60": per_asset(panel, kurt_60),
    "kurt_down_60": per_asset(panel, kurt_down_60),
    "underwater_60": per_asset(panel, underwater_60),
    "dd_depth_60": per_asset(panel, dd_depth_60),
    "days_since_high_60": per_asset(panel, days_since_high_60),
    "trend_t_60": per_asset(panel, trend_t_60),
}

print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, panel, library=lib, fwd_cache=fwd)
    p = report(name, m)
    print("    decay:", m["decay_ic_by_horizon"])
    print("    pairwise:", m.get("library_pairwise_corr"))
    print()
    results[name] = {"metrics": m, "pass": p}

print("=== REGIME BREAKDOWN (10d IC) ===")
for name, f in cands.items():
    ic_ser = compute_ic(f, fwd["10"]).dropna()
    reg = regime_breakdown(ic_ser)
    print(f"  {name:18s} | " + " | ".join(
        f"{k}: ic={v['ic']:+.4f} icir={v['icir']:+.3f} n={v['n_dates']}"
        for k, v in reg.items()))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/_miner2_cycle33_tailrisk_results.json", "w"), indent=1, default=float)
print("DONE")
