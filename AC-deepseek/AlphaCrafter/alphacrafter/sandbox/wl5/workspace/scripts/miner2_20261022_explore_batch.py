# -*- coding: utf-8 -*-
"""miner_2 2026-10-22 cycle: novel cross-asset factor candidates (batch B).

Visible data through 2026-10-21 (current date 2026-10-22). Admission gates on the
15-asset cross-asset universe: |IC| >= 0.007 and |ICIR| >= 0.084 at h=10.

Avoids library (trend_r2_30_signed, semi_down_ratio_20, mom_120d_skip5, mom_10d_skip5,
vol_of_vol20x60, dxy_beta_60, time_under_water_120, kurt_20, WTI_BETA_60, vix_beta_cond_60x20)
and recent screens by miner_1 (eff_ratio, obv_slope, amihud, atr_comp, autocorr, rel_strength,
volume_z, gap_ratio, corr_spx, range_pos) and miner_2 batch A (eff_ratio_10, ma_cross, ret_autocorr,
vol_cluster, gk_vol_term, dist_ma60, downside_mean, dd_depth, corr_change, range_amp, cn10y_beta,
corr_us10y, vol_ratio, sharpe_60) and miner_3 (fx_beta, downside_beta_asym, tail_ratio, skew_term,
xau_beta, updown_vol_asym, zscore_60, breadth_cond_mom etc).

New candidate families (batch B):
  accel_20x60        - momentum acceleration: 20d ret minus 60d ret (2nd-order momentum)
  hurst_20           - Hurst exponent via rescaled range over 20d (trend persistence)
  range_pos_gap_5x60 - short-term (5d) vs long-term (60d) stochastic range position gap
  safe_haven_60      - defensive composite: avg corr with XAU,US10Y minus corr with SPX (60d)
  gain_part_60       - magnitude-weighted up-day participation over 60d
  rel_vol_20         - cross-sectional relative volatility (asset vol / median vol)
  entropy_sign_20    - sign-sequence entropy over 20d (low = directional persistence)
  max_gain_loss_20   - tail asymmetry: max daily gain vs max daily loss magnitude over 20d
  yield_spread_beta_60 - beta of asset returns to (US10Y-CN10Y) spread changes
  streak_avg_20      - average length of consecutive same-sign daily return runs over 20d
"""
import sys, json, math, os, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lo, O = L.load_close_panel(4000)
C = C[C.index < '2026-10-22']
V = V[V.index < '2026-10-22']
H = H[H.index < '2026-10-22']
Lo = Lo[Lo.index < '2026-10-22']
O = O[O.index < '2026-10-22']
R = C.pct_change()
LR = np.log(C).diff()

def load_macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv', parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

# ---------------- candidate builders ----------------
def accel_20x60():
    """20d momentum minus 60d momentum: recent acceleration vs medium trend."""
    m20 = C / C.shift(20) - 1.0
    m60 = C / C.shift(60) - 1.0
    return (m20 - m60)

def hurst_20():
    """Hurst exponent via rescaled range (R/S) on 20d log-return window."""
    out = {}
    for s in C.columns:
        x = LR[s]
        def rs(seg):
            seg = np.asarray(seg, dtype=float)
            if len(seg) < 8 or np.isnan(seg).any():
                return np.nan
            mean_adj = seg - seg.mean()
            z = np.cumsum(mean_adj)
            r = z.max() - z.min()
            sd = seg.std()
            if sd == 0 or np.isnan(sd) or r == 0:
                return np.nan
            return r / sd
        h = x.rolling(20).apply(rs, raw=True)
        out[s] = h
    return pd.DataFrame(out).sort_index()

def range_pos_gap_5x60():
    """(5d stochastic position) - (60d stochastic position): near-term pressure vs trend."""
    def sp(win):
        hi = H.rolling(win).max()
        lo = Lo.rolling(win).min()
        return ((C - lo) / (hi - lo).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return sp(5) - sp(60)

def safe_haven_60():
    """Defensive composite: (corr with XAU + corr with US10Y) - corr with SPX over 60d."""
    xau = R['XAU']
    us10y = R['US10Y']
    spx = R['SPX']
    out = {}
    for s in C.columns:
        c_xau = R[s].rolling(60).corr(xau)
        c_us = R[s].rolling(60).corr(us10y)
        c_spx = R[s].rolling(60).corr(spx)
        out[s] = (c_xau + c_us) - c_spx
    return pd.DataFrame(out).sort_index()

def gain_part_60():
    """Magnitude-weighted up-day participation: sum(pos rets) / sum(|rets|) over 60d."""
    pos = R.where(R > 0, 0.0)
    neg = R.where(R < 0, 0.0)
    return (pos.rolling(60).sum() / (pos.rolling(60).sum() - neg.rolling(60).sum()).replace(0, np.nan))

def rel_vol_20():
    """Cross-sectional relative volatility: asset 20d vol / cross-sectional median 20d vol."""
    vol = R.rolling(20).std()
    med = vol.median(axis=1)
    return (vol / med.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def entropy_sign_20():
    """Sign-sequence entropy over 20d: -sum(p*log2(p)) for p = P(up), P(down); low = persistence."""
    out = {}
    for s in C.columns:
        up = (R[s] > 0).astype(float)
        def ent(seg):
            if len(seg) < 10:
                return np.nan
            p = seg.mean()
            if p <= 0 or p >= 1:
                return 0.0
            return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        e = up.rolling(20).apply(ent, raw=True)
        out[s] = e
    return pd.DataFrame(out).sort_index()

def max_gain_loss_20():
    """Tail asymmetry: max daily gain / |max daily loss| over 20d (positive = fat upside tail)."""
    maxg = R.rolling(20).max()
    maxl = R.rolling(20).min()
    return (maxg / maxl.abs().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def yield_spread_beta_60():
    """Beta of asset returns to (US10Y - CN10Y) spread changes over 60d (rate-differential sensitivity)."""
    spread = C['US10Y'] - C['CN10Y']
    dspread = spread.diff()
    out = {}
    for s in C.columns:
        y = R[s]
        cov = y.rolling(60, min_periods=30).cov(dspread)
        var = dspread.rolling(60, min_periods=30).var()
        out[s] = (cov / var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(out).sort_index()

def streak_avg_20():
    """Average length of consecutive same-sign daily-return runs over 20d."""
    out = {}
    for s in C.columns:
        sgn = np.sign(R[s])
        def avg_streak(seg):
            seg = np.asarray(seg, dtype=float)
            if len(seg) < 10:
                return np.nan
            seg = seg[seg != 0]
            if len(seg) < 5:
                return np.nan
            runs = []
            cur = 1
            for i in range(1, len(seg)):
                if np.sign(seg[i]) == np.sign(seg[i - 1]):
                    cur += 1
                else:
                    runs.append(cur)
                    cur = 1
            runs.append(cur)
            return float(np.mean(runs))
        out[s] = sgn.rolling(20).apply(avg_streak, raw=True)
    return pd.DataFrame(out).sort_index()

CANDIDATES = {
    'accel_20x60': (accel_20x60, 'Momentum acceleration: 20d minus 60d momentum'),
    'hurst_20': (hurst_20, 'Hurst exponent via rescaled range 20d'),
    'range_pos_gap_5x60': (range_pos_gap_5x60, '5d vs 60d stochastic range position gap'),
    'safe_haven_60': (safe_haven_60, 'Defensive composite corr (XAU+US10Y-SPX) 60d'),
    'gain_part_60': (gain_part_60, 'Magnitude-weighted up-day participation 60d'),
    'rel_vol_20': (rel_vol_20, 'Cross-sectional relative volatility 20d'),
    'entropy_sign_20': (entropy_sign_20, 'Sign-sequence entropy 20d'),
    'max_gain_loss_20': (max_gain_loss_20, 'Max daily gain vs loss tail asymmetry 20d'),
    'yield_spread_beta_60': (yield_spread_beta_60, 'Beta to US10Y-CN10Y spread changes 60d'),
    'streak_avg_20': (streak_avg_20, 'Average same-sign return run length 20d'),
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
        if summ.get('library_rho_by_factor'):
            top_rho = sorted([(k2, v2) for k2, v2 in summ['library_rho_by_factor'].items() if v2 is not None],
                             key=lambda kv: abs(kv[1]), reverse=True)[:3]
            print(f"     top library rho: " + ", ".join(f"{k2}={v2:.3f}" for k2, v2 in top_rho))
    except Exception as e:
        traceback.print_exc()
        out['results'][fid] = {'error': str(e)}
        print(f"[{fid}] ERROR {e}")

with open('scripts/miner2_20261022_explore_batch_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("\nSaved results to scripts/miner2_20261022_explore_batch_results.json")
