# -*- coding: utf-8 -*-
"""miner_3 2026-09-24: batch exploration of novel cross-asset factors.
Visible data through 2026-09-23. Admission gates: |IC|>=0.007, |ICIR|>=0.084 @10d.
Each candidate validated with same-horizon (10d) rank IC, regime split, coverage,
turnover, decay, and max abs library correlation (from signal artifacts)."""
import sys, json, math
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lo, O = L.load_close_panel(2500)
R = C.pct_change()

def realized_vol(panel, win):
    r = panel.pct_change()
    return r.rolling(win).std() * math.sqrt(252)

# ---------------- candidate factor builders ----------------
def vol_slope_10x60():
    rv10 = realized_vol(C, 10)
    rv60 = realized_vol(C, 60)
    return ((rv10 - rv60) / rv60).replace([np.inf, -np.inf], np.nan)

def vol_ratio_20x60():
    rv20 = realized_vol(C, 20)
    rv60 = realized_vol(C, 60)
    return (rv20 / rv60).replace([np.inf, -np.inf], np.nan)

def range_pos_20():
    hi = H.rolling(20).max()
    lo = Lo.rolling(20).min()
    return ((C - lo) / (hi - lo)).replace([np.inf, -np.inf], np.nan)

def pullback_depth_20():
    hi = H.rolling(20).max()
    return (C / hi - 1.0)

def up_vol_ratio_20():
    up = (R > 0).astype(float) * V
    up_sum = up.rolling(20).sum()
    tot = V.rolling(20).sum()
    return (up_sum / tot).replace([np.inf, -np.inf], np.nan)

def rev_5d():
    return -R.rolling(5).sum()

def max_gain_60():
    return R.rolling(60).max()

def clv_20():
    rng = (H - Lo).replace(0, np.nan)
    clv = ((C - Lo) / rng)
    return clv.rolling(20).mean()

CANDIDATES = {
    'vol_slope_10x60': (vol_slope_10x60, 'Vol term-structure slope (rv10-rv60)/rv60'),
    'vol_ratio_20x60': (vol_ratio_20x60, 'Vol ratio rv20/rv60'),
    'range_pos_20': (range_pos_20, '20d range position (stochastic %K)'),
    'pullback_depth_20': (pullback_depth_20, 'Depth below 20d high (negative = pullback)'),
    'up_vol_ratio_20': (up_vol_ratio_20, 'Buy-side volume pressure over 20d'),
    'rev_5d': (rev_5d, '5d short-term reversal'),
    'max_gain_60': (max_gain_60, 'Max 1d gain over 60d (lottery)'),
    'clv_20': (clv_20, 'Mean close-location-value (C-L)/(H-L) over 20d'),
}

out = {'visible_through': str(C.index.max().date()), 'results': {}}
for fid, (fn, desc) in CANDIDATES.items():
    try:
        panel = fn()
        summ = L.full_validate(panel, R, horizon=10, label=fid)
        ic, icir = summ['ic'], summ['icir']
        gate = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
        summ['pass_gate'] = bool(gate)
        summ['description'] = desc
        out['results'][fid] = {k: summ[k] for k in
                               ['label', 'horizon', 'ic', 'icir', 'ic_hit_ratio', 'n_ic_dates',
                                'regime', 'coverage_asset_days', 'coverage_dates_ge8',
                                'turnover_10d_rank', 'decay_ic_by_horizon',
                                'max_abs_library_correlation', 'library_rho_by_factor', 'pass_gate']}
        print(f"[{fid}] {desc}")
        print(f"   IC={ic:.4f} ICIR={icir:.4f} hit={summ['ic_hit_ratio']:.3f} n={summ['n_ic_dates']} "
              f"cov_asset={summ['coverage_asset_days']:.3f} cov_dates_ge8={summ['coverage_dates_ge8']:.3f} "
              f"turnover={summ['turnover_10d_rank']:.3f} maxrho={summ['max_abs_library_correlation']:.3f} gate={gate}")
        reg = summ['regime']
        for k, v in reg.items():
            print(f"     regime {k}: IC={v['ic']:.4f} ICIR={v['icir']:.4f} n={v['n']}")
        dec = summ['decay_ic_by_horizon']
        print(f"     decay: " + ", ".join(f"{h}:{icv:.4f}" for h, icv in dec.items()))
    except Exception as e:
        import traceback; traceback.print_exc()
        out['results'][fid] = {'error': str(e)}
        print(f"[{fid}] ERROR {e}")

with open('scripts/miner_3_20260924_explore_batch_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("\nSaved results to scripts/miner_3_20260924_explore_batch_results.json")
