"""miner_2 exploration batch 9 (2030-03-26).
Motivation: existing library is momentum/volatility/beta dominated; test 7
orthogonal low-correlation ideas (retracement quality, upside breadth, risk
premium, volatility asymmetry, regime persistence, crossover trend, range
position). Gates: |IC|>=0.007 and |ICIR|>=0.084 @ h=10 on the 15-instrument
cross-asset universe, warm-up through 2026-07-15.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
from miner_2_lib import (load_panel, load_macro, per_asset, abs_zscore,
                         validate_factor)

# Per-asset candidate builders (s: close series, returns Series aligned to s index)
def f_retracement_quality(s, window=20):
    ret = s.pct_change()
    up = ret.clip(lower=0).rolling(window).sum()
    dn = (-ret.clip(upper=0)).rolling(window).sum()
    return (up - dn) / (up + dn).replace(0, float("nan"))

def f_upside_share(s, window=20):
    return (s.pct_change() > 0).rolling(window).mean()

def f_risk_premium(s, window=20):
    ret = s.pct_change()
    vol = ret.rolling(window).std()
    return ret.rolling(window).sum() - vol

def f_vol_skew(s, window=20):
    r = s.pct_change()
    mu = r.rolling(window).mean()
    med = r.rolling(window).median()
    sd = r.rolling(window).std()
    return (mu - med) / sd.replace(0, float("nan"))

def f_state_persistence(s, win1=5, win2=20):
    r = s.pct_change()
    return r.rolling(win1).mean() - r.rolling(win2).mean()

def f_crossover(s, fast=20, slow=60):
    r = s.pct_change()
    return r.rolling(fast).mean() - r.rolling(slow).mean()

def f_range_pos(s, window=20):
    hi = s.rolling(window).max()
    lo = s.rolling(window).min()
    return (s - lo) / (hi - lo).replace(0, float("nan"))


def make_factor_fn(fn, kw):
    """Return factor_fn(panel, macro) -> DataFrame using per_asset + robust zscore."""
    def factor_fn(panel, macro):
        inner = per_asset(lambda s, fn=fn, kw=kw: abs_zscore(fn(s, **kw)))
        f = inner(panel, macro)
        return f
    return factor_fn


if __name__ == "__main__":
    panel = load_panel()
    macro = load_macro()
    cands = {
        "retracement_quality_20": (f_retracement_quality, dict(window=20)),
        "upside_share_20": (f_upside_share, dict(window=20)),
        "risk_premium_20": (f_risk_premium, dict(window=20)),
        "vol_skew_20": (f_vol_skew, dict(window=20)),
        "state_persistence_5x20": (f_state_persistence, dict(win1=5, win2=20)),
        "crossover_20x60": (f_crossover, dict(fast=20, slow=60)),
        "range_pos_20": (f_range_pos, dict(window=20)),
    }
    out = {}
    for name, (fn, kw) in cands.items():
        res = validate_factor(name, make_factor_fn(fn, kw))
        out[name] = {k: res[k] for k in
                     ["ic_h1", "ic_h2", "ic_h3", "ic_h5", "ic_h10", "ic_h20",
                      "icir_h10", "hit_h10", "n_dates_h10", "coverage_asset_days",
                      "coverage_dates_ge8", "turnover_10d_rank",
                      "max_abs_library_correlation", "direction",
                      "admission_gate"]}
    import json
    with open("scripts/miner_2_20300326_batch9_results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nSAVED batch9 results")