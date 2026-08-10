"""miner_2 persistence: risk family factors passing IC/ICIR gate (h=10).
Persists: beta_ew_60d, rel_mom_20d_skip5, max_ret_20d, downside_vol_ratio_20.
Each file is then read back and verified (JSON validity, id, status, gates).
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_2_lib import (validate_factor, persist_factor, load_panel, load_macro,
                         per_asset, ADMISSION)
from pathlib import Path

panel = load_panel()
macro = load_macro()
rets = panel.pct_change()
mkt = rets.mean(axis=1)

# 1) beta vs EW market (60d)
def make_low_beta(win):
    def f(s):
        r = s.pct_change()
        z = pd.concat([r.rename("r"), mkt.reindex(s.index).rename("m")], axis=1)
        return z["r"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var().replace(0, np.nan)
    return per_asset(f)

# 2) relative momentum 20d skip 5
def make_rel_mom(n, skip):
    def f(s):
        return s.shift(skip) / s.shift(n + skip) - 1.0
    def inner(pnl, mcr):
        mom = per_asset(f)(pnl, mcr)
        return mom.sub(mom.median(axis=1), axis=0)
    return inner

# 3) max daily return 20d
def make_max_ret(n):
    return per_asset(lambda s: s.pct_change().rolling(n).max())

# 4) downside vol ratio 20d (flip sign to make IC positive)
def make_downside_vol_ratio(win, flip=False):
    def f(s):
        r = s.pct_change()
        tot = r.rolling(win).std()
        dd = r.clip(upper=0).rolling(win).std()
        v = dd / tot
        return -v if flip else v
    return per_asset(f)

if __name__ == "__main__":
    specs = [
        ("beta_ew_60d", "Beta vs EW market (60d)", "rolling 60d beta of asset returns vs equal-weight cross-asset market return",
         "High-beta assets' relative performance vs the EW basket over 60 days; IC was positive over the warm-up window.",
         ["close"], {"window": 60, "market": "equal_weight_15"}, make_low_beta(60),
         ["risk", "beta", "cross_asset"], "2020-01..2026-07: broad cross-asset risk-on/off rotation; IC positive at h=10."),
        ("rel_mom_20d_skip5", "Relative momentum 20d skip 5", "per-asset 20d momentum (skip 5) minus cross-sectional median",
         "Relative (cross-sectionally demeaned) 20-day trend, skipping the last 5 days to reduce short-term reversal.",
         ["close"], {"window": 20, "skip": 5}, make_rel_mom(20, 5),
         ["momentum", "relative", "cross_asset"], "2020-01..2026-07: persistent cross-asset trend regime; strongest hit ratio of family."),
        ("max_ret_20d", "Max daily return 20d", "rolling 20-day maximum of daily returns",
         "Extreme up-move intensity over 20 days; positive relation with forward 10d returns in the sample.",
         ["close"], {"window": 20}, make_max_ret(20),
         ["risk", "tail", "cross_asset"], "2020-01..2026-07: crypto/commodity up-runs positively anticipated."),
        ("downside_vol_ratio_20", "Downside vol ratio 20d (flipped)", "- (downside semi-vol / total vol) over 20d",
         "Ratio of downside semi-volatility to total volatility over 20 days, sign-flipped so higher = more symmetric/lower downside concentration.",
         ["close"], {"window": 20}, make_downside_vol_ratio(20, flip=True),
         ["risk", "volatility", "asymmetry", "cross_asset"], "2020-01..2026-07: downside-heavy assets underperformed; flipped sign gives positive IC."),
    ]

    persisted = []
    for fid, fname, expr, desc, deps, params, fn, tags, regime in specs:
        res = validate_factor(fid, fn)
        if not res["admission_gate"]["pass"]:
            print(f"SKIP {fid}: gate FAIL -> not persisted")
            continue
        p = persist_factor(fid, fname, expr, desc, deps, params, res, tags, regime)
        # read back + verify
        back = json.loads(p.read_text())
        v = back["validation"]
        ok = (back["factor_id"] == fid and v["status"] == "EFFECTIVE"
              and abs(v["metrics"]["ic"]) >= ADMISSION["ic"]
              and abs(v["metrics"]["icir"]) >= ADMISSION["icir"])
        print(f"VERIFY {fid}: id={back['factor_id']!r} status={v['status']} "
              f"ic={v['metrics']['ic']:+.4f} icir={v['metrics']['icir']:+.4f} "
              f"max_abs_lib_corr={v['metrics'].get('max_abs_library_correlation')} -> {'OK' if ok else 'MISMATCH'}")
        persisted.append((fid, ok))

    print("\nPERSISTED:", [f for f, ok in persisted if ok])
