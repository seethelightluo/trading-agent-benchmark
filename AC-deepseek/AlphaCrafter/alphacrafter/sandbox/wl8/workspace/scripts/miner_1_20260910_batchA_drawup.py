"""miner_1 2026-09-10 -- cycle-24 batch A: risk-adjusted drawup magnitude factors.

Motivation: prior histograms (raw 1d momentum, max_ret_20) showed very high
cross-sectional dispersion (rank IC reachable +/-10-20% on single dates) in this
15-asset universe. Drawup/range-position factors capture how far an asset has
climbed within its recent high-low range (or from its rolling min) - a
position/overbought-extendedness signal that is NOT a raw return (thus naturally
less collinear with the raw-momentum family that keeps getting evicted vs
usdcny_beta_60).

Candidates:
  1. hl_pos_20     : (close - min_low_20) / (max_high_20 - min_low_20)  [range position]
  2. hl_pos_60     : same with 60d window
  3. drawup_20     : (close - min_low_20) / vol_20  [normalized drawup from min]
  4. drawup_60     : (close - min_low_60) / vol_60
  5. drawdown_20   : -1 * (max_high_20 - close) / vol_20  [proximity to high -> negative]
  6. rr_20         : (close - min_low_20) / (max_high_20 - min_low_20 + eps) cross-check

Validation through 2026-09-09 (visible data), primary horizon 10d, gates
|IC|>=0.007 & |ICIR|>=0.084, pooled-Spearman library rho < 0.5 vs live library
(usdcny_beta_60; also report mom_10d_skip5-like proxies where artifacts exist).

No lookahead: factors use data <= t; forward returns t..t+10.
"""
import sys
import json
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner1_fvl_20260910 as fvl

END = fvl.END
IC_GATE, ICIR_GATE, RHO_GATE = fvl.IC_GATE, fvl.ICIR_GATE, fvl.RHO_GATE
t0 = time.time()

close, vol, open_, high, low = fvl.load_panel()
macro = fvl.load_macro()
print(f"panel {close.index[0].date()}..{close.index[-1].date()} rows={len(close)} "
      f"assets={close.shape[1]} end={END.date()}", flush=True)

lib = fvl.load_live_library()
print(f"live library for correlation gate: {list(lib.keys())}", flush=True)


def hl_pos(c, vol, o, h, l, macro, window=20):
    mx = h.rolling(window, min_periods=window // 2).max()
    mn = l.rolling(window, min_periods=window // 2).min()
    rng = (mx - mn).replace(0, np.nan)
    return ((c - mn) / rng).clip(0, 1)


def drawup(c, vol, o, h, l, macro, window=20):
    mn = l.rolling(window, min_periods=window // 2).min()
    rv = c.pct_change().rolling(window, min_periods=window // 2).std()
    return (c - mn) / rv.replace(0, np.nan)


def drawdown(c, vol, o, h, l, macro, window=20):
    mx = h.rolling(window, min_periods=window // 2).max()
    rv = c.pct_change().rolling(window, min_periods=window // 2).std()
    return -1.0 * (mx - c) / rv.replace(0, np.nan)


def rr_20(c, vol, o, h, l, macro, window=20):
    """close relative to (low->low+vol) bucket - alternative normalized position"""
    mn = l.rolling(window, min_periods=window // 2).min()
    rv = c.pct_change().rolling(window, min_periods=window // 2).std()
    return ((c - mn) / rv.replace(0, np.nan)).clip(0, 3)


CANDIDATES = {
    "hl_pos_20": lambda *a, **k: hl_pos(*a, **k, window=20),
    "hl_pos_60": lambda *a, **k: hl_pos(*a, **k, window=60),
    "drawup_20": lambda *a, **k: drawup(*a, **k, window=20),
    "drawup_60": lambda *a, **k: drawup(*a, **k, window=60),
    "drawdown_20": lambda *a, **k: drawdown(*a, **k, window=20),
}

results, panels = {}, {}
for name, fn in CANDIDATES.items():
    panel = fvl.factor_panel(fn, close, vol, open_, high, low, macro)
    res = fvl.validate_factor(panel, close, horizons=(1, 2, 3, 5, 10, 20),
                              admission_horizon=10)
    panels[name] = panel
    mrho, cdetail = fvl.max_abs_library_corr(panel, lib)
    res["max_abs_library_correlation"] = mrho
    res["library_correlation_detail"] = {k: v["pooled_spearman"] for k, v in cdetail.items()}
    res["regime_ic_icir"] = fvl.regime_ic(panel, close)
    ok = (abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
          and (mrho is None or mrho < RHO_GATE))
    res["gate_pass"] = bool(ok)
    results[name] = res
    print("=" * 72, flush=True)
    print(f"{name}: IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"n={res['n_ic_dates']} cov={res['coverage_asset_days']:.3f} "
          f"cov8={res['coverage_dates_ge8']:.3f} to={res['turnover_10d_rank']:.2f} "
          f"maxrho={mrho} PASS={ok}", flush=True)
    print(f"  decay={res['decay_ic_by_horizon']}", flush=True)
    print(f"  regime={res['regime_ic_icir']}", flush=True)
    print(f"  lib_rho={res['library_correlation_detail']}", flush=True)

with open("_miner1_cycle24_batchA_drawup_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "panel"}
               for k, v in results.items()}, f, indent=1)

print(f"\n[all done] elapsed={time.time()-t0:.1f}s", flush=True)