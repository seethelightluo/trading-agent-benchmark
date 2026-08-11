# -*- coding: utf-8 -*-
"""miner_1 2027-02-25 cycle: intraday OHLC price-structure + return-dynamics batch.
Motivation: existing library is built almost entirely on close prices (momentum, vol,
drawdown, beta). O/H/L/C intraday structure (close position in range, overnight gaps,
intraday session returns, shadow proportions, gap-fill behavior) and return dynamics
(autocorrelation, tail asymmetry, drawdown depth, up/down vol asymmetry) are untapped
dimensions that work for ALL 15 tradable assets (full coverage).
Data visible through 2027-02-24 (previous completed trading day). No future data used.
Gates (h=10): |IC| >= 0.0070, |ICIR| >= 0.0840.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
import miner3_lib as L

# active library for correlation audit (all factors present in factors/ with artifacts)
L.LIB_FACTORS = ['trend_r2_30_signed', 'semi_down_ratio_20', 'mom_120d_skip5',
                 'mom_10d_skip5', 'time_under_water_120', 'vol_of_vol20x60',
                 'dxy_beta_60', 'WTI_BETA_60', 'vix_beta_cond_60x20', 'kurt_20']

VIS = '2027-02-25'
C, V, H, Lo, O = L.load_close_panel(4000)
mask = C.index < VIS
C, V, H, Lo, O = C[mask], V[mask], H[mask], Lo[mask], O[mask]
R = C.pct_change()
LR = np.log(C).diff()
EW = LR.mean(axis=1)

print(f"Panel dates: {C.index.min().date()} -> {C.index.max().date()}, {len(C)} rows, {C.shape[1]} assets")

MP = lambda w: int(w * 0.5)

def _clean(a):
    a = np.asarray(a, dtype=float)
    return a[~np.isnan(a)]

def _apply_win(panel, w, fn):
    return panel.rolling(w, min_periods=MP(w)).apply(
        lambda a: fn(_clean(a)), raw=True)

# ---------- intraday price structure (all 15 assets) ----------
def close_pos_range_20(w=20):
    """Mean of (C-L)/(H-L): where the close sits inside the daily range (buying pressure)."""
    rng = (H - Lo)
    pos = (C - Lo) / rng.replace(0, np.nan)
    return pos.rolling(w, min_periods=MP(w)).mean()

def overnight_gap_20(w=20):
    """Mean overnight gap (O - C_prev)/C_prev: persistence of gap direction."""
    gap = (O - C.shift(1)) / C.shift(1).replace(0, np.nan)
    return gap.rolling(w, min_periods=MP(w)).mean()

def intraday_ret_20(w=20):
    """Mean intraday session return (C-O)/O: strength of the intraday session."""
    ir = (C - O) / O.replace(0, np.nan)
    return ir.rolling(w, min_periods=MP(w)).mean()

def range_ratio_20(w=20):
    """Mean (H-L)/C: average intraday range size (intraday volatility level)."""
    rr = (H - Lo) / C.replace(0, np.nan)
    return rr.rolling(w, min_periods=MP(w)).mean()

def upper_shadow_20(w=20):
    """Mean (H - max(O,C))/(H-L): proportion of upper wick (supply at highs)."""
    rng = (H - Lo).replace(0, np.nan)
    us = (H - np.maximum(O, C)) / rng
    return us.rolling(w, min_periods=MP(w)).mean()

def lower_shadow_20(w=20):
    """Mean (min(O,C) - L)/(H-L): proportion of lower wick (support at lows)."""
    rng = (H - Lo).replace(0, np.nan)
    ls = (np.minimum(O, C) - Lo) / rng
    return ls.rolling(w, min_periods=MP(w)).mean()

def gap_fill_20(w=20):
    """Mean(overnight_gap * intraday_ret): positive => gaps persist (trend),
    negative => gaps fade (mean reversion)."""
    gap = (O - C.shift(1)) / C.shift(1).replace(0, np.nan)
    ir = (C - O) / O.replace(0, np.nan)
    prod = gap * ir
    return prod.rolling(w, min_periods=MP(w)).mean()

# ---------- return dynamics ----------
def autocorr_1_20(w=20):
    """Rolling 1-lag autocorrelation of daily returns: momentum persistence vs reversion."""
    return R.rolling(w, min_periods=MP(w)).corr(R.shift(1))

def tail_ratio_20(w=20):
    """q95/q05 of daily returns over 20d: positive => fat right tail (upward skew)."""
    q95 = _apply_win(R, w, lambda a: np.percentile(a, 95))
    q05 = _apply_win(R, w, lambda a: np.percentile(a, 5))
    return q95 / q05.abs().replace(0, np.nan)

def dd_depth_20(w=20):
    """Current drawdown depth vs 20d high (negative = deep drawdown)."""
    hi = C.rolling(w, min_periods=MP(w)).max()
    return (C - hi) / hi.replace(0, np.nan)

def up_down_vol_ratio_20(w=20):
    """std of up-day returns / std of down-day returns over 20d."""
    up = _apply_win(R, w, lambda a: a[a > 0].std(ddof=0) if (a > 0).sum() >= 5 else np.nan)
    dn = _apply_win(R, w, lambda a: a[a < 0].std(ddof=0) if (a < 0).sum() >= 5 else np.nan)
    return up / dn.replace(0, np.nan)

CANDIDATES = {
    'close_pos_range_20': close_pos_range_20,
    'overnight_gap_20': overnight_gap_20,
    'intraday_ret_20': intraday_ret_20,
    'range_ratio_20': range_ratio_20,
    'upper_shadow_20': upper_shadow_20,
    'lower_shadow_20': lower_shadow_20,
    'gap_fill_20': gap_fill_20,
    'autocorr_1_20': autocorr_1_20,
    'tail_ratio_20': tail_ratio_20,
    'dd_depth_20': dd_depth_20,
    'up_down_vol_ratio_20': up_down_vol_ratio_20,
}

results = {}
for name, fn in CANDIDATES.items():
    try:
        fp = fn()
        fp = fp.replace([np.inf, -np.inf], np.nan)
        s = L.rank_ic(fp, R.shift(-10))
        if s is None or len(s) < 30:
            results[name] = {"error": f"insufficient IC dates ({0 if s is None else len(s)})"}
            print(f"{name}: INSUFFICIENT ({0 if s is None else len(s)} dates)")
            continue
        summ = L.summarize(s, 10, name)
        summ['decay_ic_by_horizon'] = L.decay_analysis(fp, R)
        cov = L.coverage_turnover(fp, R, 10)
        summ.update(cov)
        rhos, maxrho = L.library_max_rho(fp)
        summ['library_rho_by_factor'] = rhos
        summ['max_abs_library_correlation'] = round(maxrho, 3)
        results[name] = summ
        gate = abs(summ['ic']) >= 0.0070 and abs(summ['icir']) >= 0.0840
        print(f"{name}: ic={summ['ic']:.4f} icir={summ['icir']:.4f} n={summ['n_ic_dates']} "
              f"hit={summ['ic_hit_ratio']:.2f} cov_a={summ['coverage_asset_days']:.3f} "
              f"cov_d={summ['coverage_dates_ge8']:.3f} rho={summ['max_abs_library_correlation']:.3f} "
              f"decay10={summ['decay_ic_by_horizon'].get('10')} {'*** PASS ***' if gate else 'fail'}")
        print(f"    regime: {summ['regime']}")
    except Exception as e:
        results[name] = {"error": str(e)}
        print(f"{name}: ERROR {e}")

with open('scripts/miner_1_20270225_intraday_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nSaved scripts/miner_1_20270225_intraday_results.json")
