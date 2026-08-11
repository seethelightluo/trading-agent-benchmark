# -*- coding: utf-8 -*-
"""miner_2 2026-11-05 cycle: cross-asset factor candidates (batch C).

Visible data through 2026-11-04 (current date 2026-11-05). Admission gates on the
15-asset cross-asset universe: |IC| >= 0.007 and |ICIR| >= 0.084 at h=10.

Includes:
- Queued batch-B candidates (never validated last cycle): accel_20x60, hurst_20,
  range_pos_gap_5x60, safe_haven_60, gain_part_60, rel_vol_20, entropy_sign_20,
  max_gain_loss_20, yield_spread_beta_60, streak_avg_20
- New batch-C defensive/rotation families (motivated by 4 cycles of tech-heavy
  tilt hurting via NDX/SOX): market beta (defensive low-beta), safe-haven relative
  momentum vs XAU, risk-on relative momentum vs crypto, rate sensitivity,
  commodity-growth beta, damage-per-risk mean reversion, volume attention trend,
  asymmetric up/down beta, commodity relative momentum.

All candidates are interpretable and chosen to avoid near-duplicates of the library
(trend_r2_30_signed, semi_down_ratio_20, mom_120d_skip5, mom_10d_skip5,
vol_of_vol20x60, dxy_beta_60, time_under_water_120, kurt_20, WTI_BETA_60,
vix_beta_cond_60x20) and recent miner_1/miner_3 screens.
"""
import sys, json, math, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
import miner3_lib as L

VIS = '2026-11-05'  # current date; data visible only through previous completed day

C, V, H, Lo, O = L.load_close_panel(4000)
mask = C.index < VIS
C = C[mask]; V = V[mask]; H = H[mask]; Lo = Lo[mask]; O = O[mask]
R = C.pct_change()
LR = np.log(C).diff()
EW = LR.mean(axis=1)  # equal-weight cross-asset index (log ret)

def load_macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv', parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    df = df[df.index < VIS]
    return df

def rolling_beta(asset_ret, ref_ret, win):
    cov = asset_ret.rolling(win).cov(ref_ret)
    var = ref_ret.rolling(win).var()
    return cov / var

# ---------- Batch B (queued) ----------
def accel_20x60():
    return LR.rolling(20).sum() - LR.rolling(60).sum()

def hurst_20():
    """Hurst exponent via rescaled range on trailing 20d log returns (per asset)."""
    out = pd.DataFrame(index=LR.index, columns=LR.columns, dtype=float)
    arr = LR.values
    n = arr.shape[0]
    for j in range(arr.shape[1]):
        for i in range(20, n):
            seg = arr[i-20:i, j]
            seg = seg - seg.mean()
            if np.std(seg) == 0 or not np.isfinite(seg).all():
                continue
            rs_vals, ns = [], []
            for sub in (4, 5, 10):
                m = 20 // sub
                for k in range(m):
                    s = seg[k*sub:(k+1)*sub]
                    sd = np.std(s)
                    if sd == 0 or len(s) < 2:
                        continue
                    rs = (s.max() - s.min()) / sd
                    rs_vals.append(rs); ns.append(len(s))
            if len(rs_vals) >= 3:
                out.iloc[i, j] = np.polyfit(np.log(ns), np.log(rs_vals), 1)[0]
    return out

def range_pos_gap_5x60():
    def sp(win):
        hi = C.rolling(win).max(); lo = C.rolling(win).min()
        return (C - lo) / (hi - lo).replace(0, np.nan)
    return sp(5) - sp(60)

def safe_haven_60():
    """(corr with XAU + corr with US10Y)/2 - corr with SPX, 60d."""
    c_xau = C.rolling(60).corr(C['XAU'])
    c_bnd = C.rolling(60).corr(C['US10Y'])
    c_spx = C.rolling(60).corr(C['SPX'])
    return (c_xau + c_bnd) / 2.0 - c_spx

def gain_part_60():
    up = R.clip(lower=0).rolling(60).sum()
    tot = R.abs().rolling(60).sum()
    return up / tot.replace(0, np.nan)

def rel_vol_20():
    v = LR.rolling(20).std()
    med = v.median(axis=1)
    return v.div(med, axis=0)

def entropy_sign_20():
    """Binary entropy of up-day fraction over 20d (low = directional persistence)."""
    p = (R > 0).rolling(20).mean()
    p = p.clip(1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)

def max_gain_loss_20():
    g = R.rolling(20).max()
    l = R.rolling(20).min().abs()
    return g / l.replace(0, np.nan)

def yield_spread_beta_60():
    sp = LR['CN10Y'] - LR['US10Y']
    return rolling_beta(LR, sp, 60)

def streak_avg_20():
    """Average length of consecutive same-sign daily return runs over 20d."""
    out = pd.DataFrame(index=LR.index, columns=LR.columns, dtype=float)
    sgn = np.sign(R.values)
    n = sgn.shape[0]
    for j in range(n):
        pass
    for j in range(sgn.shape[1]):
        for i in range(20, n):
            seg = sgn[i-20:i, j]
            seg = seg[seg != 0]
            if len(seg) == 0:
                continue
            runs, cur = [], 1
            for k in range(1, len(seg)):
                if seg[k] == seg[k-1]:
                    cur += 1
                else:
                    runs.append(cur); cur = 1
            runs.append(cur)
            out.iloc[i, j] = np.mean(runs)
    return out

# ---------- Batch C (new) ----------
def mkt_beta_60():
    return rolling_beta(LR, EW, 60)

def mkt_beta_20():
    return rolling_beta(LR, EW, 20)

def xau_rel_mom_20():
    return C.pct_change(20).sub(C['XAU'].pct_change(20), axis=0)

def crypto_rel_mom_20():
    cr = C[['BTC', 'ETH']].pct_change(20).mean(axis=1)
    return C.pct_change(20).sub(cr, axis=0)

def commodity_rel_mom_20():
    cm = C[['COPPER', 'WTI']].pct_change(20).mean(axis=1)
    return C.pct_change(20).sub(cm, axis=0)

def rate_beta_60():
    """Beta of asset log returns to US10Y log changes (rate sensitivity)."""
    return rolling_beta(LR, LR['US10Y'], 60)

def copper_beta_60():
    """Beta to COPPER log returns (global growth sensitivity)."""
    return rolling_beta(LR, LR['COPPER'], 60)

def dd_depth_vol_60():
    """Current distance from 60d high per unit of 60d vol (damage/risk; mean reversion)."""
    dd = 1.0 - C / C.rolling(60).max()
    vol = LR.rolling(60).std()
    return dd.div(vol.replace(0, np.nan), axis=0)

def vol_trend_20x60():
    return V.rolling(20).mean() / V.rolling(60).mean().replace(0, np.nan)

def asym_beta_60():
    """Up-day beta minus down-day beta vs equal-weight index (60d, >=12 obs each)."""
    out = pd.DataFrame(index=LR.index, columns=LR.columns, dtype=float)
    lr = LR.values; ew = EW.values
    n = lr.shape[0]
    for j in range(lr.shape[1]):
        a = lr[:, j]
        for i in range(60, n):
            seg = slice(i-60, i)
            m_up = ew[seg] > 0; m_dn = ew[seg] < 0
            if m_up.sum() < 12 or m_dn.sum() < 12:
                continue
            vu = np.var(ew[seg][m_up]); vd = np.var(ew[seg][m_dn])
            if vu == 0 or vd == 0:
                continue
            bu = np.cov(a[seg][m_up], ew[seg][m_up])[0, 1] / vu
            bd = np.cov(a[seg][m_dn], ew[seg][m_dn])[0, 1] / vd
            out.iloc[i, j] = bu - bd
    return out

CANDIDATES = {
    # batch B (queued from 2026-10-22, never run)
    'accel_20x60': (accel_20x60, 'Momentum acceleration: 20d minus 60d momentum'),
    'hurst_20': (hurst_20, 'Hurst exponent via rescaled range 20d'),
    'range_pos_gap_5x60': (range_pos_gap_5x60, '5d vs 60d stochastic range position gap'),
    'safe_haven_60': (safe_haven_60, 'Defensive composite corr (XAU+US10Y vs SPX) 60d'),
    'gain_part_60': (gain_part_60, 'Magnitude-weighted up-day participation 60d'),
    'rel_vol_20': (rel_vol_20, 'Cross-sectional relative volatility 20d'),
    'entropy_sign_20': (entropy_sign_20, 'Sign-sequence binary entropy 20d'),
    'max_gain_loss_20': (max_gain_loss_20, 'Max daily gain vs loss tail asymmetry 20d'),
    'yield_spread_beta_60': (yield_spread_beta_60, 'Beta to CN10Y-US10Y spread changes 60d'),
    'streak_avg_20': (streak_avg_20, 'Average same-sign return run length 20d'),
    # batch C (new this cycle)
    'mkt_beta_60': (mkt_beta_60, 'Beta vs equal-weight cross-asset index 60d (defensive)'),
    'mkt_beta_20': (mkt_beta_20, 'Beta vs equal-weight cross-asset index 20d'),
    'xau_rel_mom_20': (xau_rel_mom_20, '20d return minus XAU 20d return (safe-haven rotation)'),
    'crypto_rel_mom_20': (crypto_rel_mom_20, '20d return minus mean(BTC,ETH) 20d return (risk-on rotation)'),
    'commodity_rel_mom_20': (commodity_rel_mom_20, '20d return minus mean(COPPER,WTI) 20d return'),
    'rate_beta_60': (rate_beta_60, 'Beta to US10Y log changes 60d (rate sensitivity)'),
    'copper_beta_60': (copper_beta_60, 'Beta to COPPER log changes 60d (growth sensitivity)'),
    'dd_depth_vol_60': (dd_depth_vol_60, 'Distance from 60d high per unit vol (damage/risk)'),
    'vol_trend_20x60': (vol_trend_20x60, 'Volume attention trend: 20d/60d mean volume'),
    'asym_beta_60': (asym_beta_60, 'Up-day beta minus down-day beta vs index 60d'),
}

out = {'visible_through': str(C.index.max().date()), 'current_date': VIS, 'results': {}}
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
        for k, v in summ['regime'].items():
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

with open('scripts/miner2_20261105_explore_batchC_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("\nSaved results to scripts/miner2_20261105_explore_batchC_results.json")
