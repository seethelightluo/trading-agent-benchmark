# -*- coding: utf-8 -*-
"""miner_1 2026-10-22: batch exploration of novel cross-asset factor candidates.
Visible data through 2026-10-21 (last completed trading day; current date 2026-10-22).
Admission gates: |IC| >= 0.007 and |ICIR| >= 0.084 @10d horizon on the 15-asset universe.
Library correlation threshold: 0.5 (deterministic post-gate recomputes rho from artifacts).
"""
import sys, json, math, traceback
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lo, O = L.load_close_panel(3000)
# keep only rows strictly before current date 2026-10-22 (last completed trading day is 10-21)
C = C[C.index < '2026-10-22']
V = V[V.index < '2026-10-22']
H = H[H.index < '2026-10-22']
Lo = Lo[Lo.index < '2026-10-22']
O = O[O.index < '2026-10-22']
R = C.pct_change()

def rv(win):
    return R.rolling(win).std() * math.sqrt(252)

def atr(win):
    tr = pd.concat([(H - Lo), (H - C.shift()).abs(), (Lo - C.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(win).mean()

# ---------------- candidate factor builders ----------------
def eff_ratio_20():
    """Kaufman efficiency ratio: |C - C_{t-20}| / sum(|r|, 20). Trend consistency."""
    num = (C - C.shift(20)).abs()
    den = R.abs().rolling(20).sum()
    return (num / den).replace([np.inf, -np.inf], np.nan)

def obv_slope_20():
    """OBV 20d slope normalized by mean volume: volume-confirmed trend."""
    obv = (np.sign(R) * V).cumsum()
    num = obv - obv.shift(20)
    den = V.rolling(20).mean() * 20
    return (num / den).replace([np.inf, -np.inf], np.nan)

def amihud_20():
    """Amihud illiquidity: mean(|r| / volume) over 20d."""
    ill = (R.abs() / V.replace(0, np.nan))
    return ill.rolling(20).mean()

def atr_comp_5x60():
    """Volatility compression: ATR5 / ATR60 (low = narrow range / breakout setup)."""
    return (atr(5) / atr(60)).replace([np.inf, -np.inf], np.nan)

def autocorr_1x20():
    """Rolling lag-1 autocorrelation of daily returns over 20d (trend persistence vs reversion)."""
    out = {}
    for s in C.columns:
        rs = R[s]
        ac = rs.rolling(20).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) >= 5 else np.nan, raw=True)
        out[s] = ac
    return pd.DataFrame(out).sort_index()

def rel_strength_20():
    """Relative strength: 20d return minus cross-sectional median 20d return (rotation signal)."""
    m20 = C / C.shift(20) - 1.0
    return m20 - m20.median(axis=1)

def volume_z_20():
    """Abnormal volume z-score over 20d."""
    mu = V.rolling(20).mean()
    sd = V.rolling(20).std()
    return ((V - mu) / sd).replace([np.inf, -np.inf], np.nan)

def gap_ratio_10():
    """Overnight gap magnitude: mean(|open_t / close_{t-1} - 1|) over 10d."""
    g = (O / C.shift(1) - 1.0).abs()
    return g.rolling(10).mean()

def corr_spx_30():
    """Rolling 30d correlation of each asset's daily returns with SPX returns."""
    spx = R['SPX']
    out = {}
    for s in C.columns:
        out[s] = R[s].rolling(30).corr(spx)
    return pd.DataFrame(out).sort_index()

def range_pos_60():
    """60d stochastic position: (C - min60)/(max60 - min60)."""
    hi = H.rolling(60).max()
    lo = Lo.rolling(60).min()
    return ((C - lo) / (hi - lo)).replace([np.inf, -np.inf], np.nan)

CANDIDATES = {
    'eff_ratio_20': (eff_ratio_20, 'Kaufman efficiency ratio 20d (trend consistency)'),
    'obv_slope_20': (obv_slope_20, 'OBV 20d slope / mean volume (volume-confirmed trend)'),
    'amihud_20': (amihud_20, 'Amihud illiquidity 20d'),
    'atr_comp_5x60': (atr_comp_5x60, 'ATR5/ATR60 volatility compression'),
    'autocorr_1x20': (autocorr_1x20, 'Lag-1 return autocorr over 20d'),
    'rel_strength_20': (rel_strength_20, 'Relative strength vs cross-sectional median (20d)'),
    'volume_z_20': (volume_z_20, 'Abnormal volume z-score 20d'),
    'gap_ratio_10': (gap_ratio_10, 'Overnight gap magnitude 10d'),
    'corr_spx_30': (corr_spx_30, '30d return correlation with SPX'),
    'range_pos_60': (range_pos_60, '60d stochastic range position'),
}

out = {'visible_through': str(C.index.max().date()), 'current_date': '2026-10-22', 'results': {}}
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
        traceback.print_exc()
        out['results'][fid] = {'error': str(e)}
        print(f"[{fid}] ERROR {e}")

with open('scripts/miner_1_20261022_explore_batch_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("\nSaved results to scripts/miner_1_20261022_explore_batch_results.json")
