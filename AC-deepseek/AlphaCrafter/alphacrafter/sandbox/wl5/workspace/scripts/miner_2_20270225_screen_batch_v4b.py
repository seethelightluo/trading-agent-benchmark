# -*- coding: utf-8 -*-
"""miner_2 2027-02-25 cycle: refinement pass on near-miss dimensions (v4b).
Longer smoothing / slope-difference transforms of close_pos, overnight share,
gap-follow, autocorr; rate beta conditioned on yield-down days (relaxed MP).
Visible data through 2027-02-24. Gates (h=10): |IC| >= 0.0070, |ICIR| >= 0.0840.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
import miner3_lib as L

L.LIB_FACTORS = [f for f in L.LIB_FACTORS if f != 'kurt_20'] + ['kurt_20']

VIS = '2027-02-25'
C, V, H, Lo, O = L.load_close_panel(4000)
mask = C.index < VIS
C, V, H, Lo, O = C[mask], V[mask], H[mask], Lo[mask], O[mask]
R = C.pct_change()
LR = np.log(C).diff()

MP = lambda w: int(w * 0.5)

def _clean(a):
    a = np.asarray(a, dtype=float)
    return a[~np.isnan(a)]

def _apply_win(panel, w, fn):
    return panel.rolling(w, min_periods=MP(w)).apply(
        lambda a: fn(_clean(a)), raw=True)

def rolling_ref_beta(asset_r, ref_r, w, cond_series=None, mp=None):
    ref_df = pd.DataFrame({c: ref_r for c in asset_r.columns}, index=asset_r.index)
    out = pd.DataFrame(index=asset_r.index, columns=asset_r.columns, dtype=float)
    a = asset_r if cond_series is None else asset_r.where(cond_series)
    r = ref_df if cond_series is None else ref_df.where(cond_series)
    mp = mp if mp is not None else MP(w)
    for c in asset_r.columns:
        if c == ref_r.name:
            out[c] = np.nan
            continue
        cov = a[c].rolling(w, min_periods=mp).cov(r[c])
        var = r[c].rolling(w, min_periods=mp).var()
        out[c] = cov / var.replace(0, np.nan)
    return out

# candlestick / buying pressure
def close_pos(w):
    rng = (H - Lo).replace(0, np.nan)
    return ((C - Lo) / rng).rolling(w, min_periods=MP(w)).mean()

def close_pos_slope(w=20, look=20):
    cp = close_pos(w)
    return cp - cp.shift(look)

def body_trend(w=20):
    rng = (H - Lo).replace(0, np.nan)
    return ((C - O) / rng).rolling(w, min_periods=MP(w)).mean()

def close_pos_diff(w_short=20, w_long=60):
    return close_pos(w_short) - close_pos(w_long)

# overnight / gap
def overnight_signed_share(w=40):
    ov = np.log(O) - np.log(C.shift(1))
    so = ov.rolling(w, min_periods=MP(w)).sum()
    st = LR.abs().rolling(w, min_periods=MP(w)).sum()
    return (so / st.replace(0, np.nan)).where(O.notna())

def gap_follow(w=40):
    ov = np.log(O) - np.log(C.shift(1))
    intr = np.log(C) - np.log(O)
    return ov.rolling(w, min_periods=MP(w)).corr(intr).where(O.notna())

# serial dependence, shorter window
def autocorr_1(w=40):
    def f(a):
        if len(a) < 5:
            return np.nan
        x, y = a[:-1], a[1:]
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])
    return _apply_win(R, w, f)

# rate beta conditioned on yield-down days, relaxed min periods
def rate_beta_cond_down_60(w=60, mp=15):
    cond = R['US10Y'] < 0
    return rolling_ref_beta(R, R['US10Y'], w, cond, mp)

CANDIDATES = {
    'close_pos_40': lambda: close_pos(40),
    'close_pos_60': lambda: close_pos(60),
    'close_pos_slope_20': lambda: close_pos_slope(20, 20),
    'close_pos_diff_20x60': lambda: close_pos_diff(20, 60),
    'body_trend_20': lambda: body_trend(20),
    'overnight_signed_share_40': lambda: overnight_signed_share(40),
    'gap_follow_40': lambda: gap_follow(40),
    'autocorr_1_40': lambda: autocorr_1(40),
    'rate_beta_cond_down_60': lambda: rate_beta_cond_down_60(60, 15),
}

RESULTS = {}
for name, fn in CANDIDATES.items():
    try:
        panel = fn()
        summ = L.full_validate(panel, R, horizon=10, label=name)
        gate = (abs(summ['ic']) >= 0.0070 and abs(summ['icir']) >= 0.0840)
        summ['PASS'] = gate
        RESULTS[name] = summ
        print(f"\n=== {name} PASS={gate} ===")
        print(f"  ic={summ['ic']:+.4f} icir={summ['icir']:+.4f} hit={summ['ic_hit_ratio']:.3f} n={summ['n_ic_dates']} "
              f"cov_ge8={summ['coverage_dates_ge8']:.3f} cov_asset={summ['coverage_asset_days']:.3f} "
              f"turn={summ['turnover_10d_rank']:.2f} rho_max={summ['max_abs_library_correlation']}")
        print(f"  regime: {summ['regime']}")
        print(f"  decay: {summ['decay_ic_by_horizon']}")
    except Exception as e:
        RESULTS[name] = {'error': str(e)}
        print(f"\n=== {name} ERROR: {e}")

with open('scripts/miner_2_20270225_screen_results_v4b.json', 'w') as f:
    json.dump(RESULTS, f, indent=1, default=str)
npass = sum(1 for v in RESULTS.values() if isinstance(v, dict) and v.get('PASS'))
print(f"\nSaved. PASSING candidates: {npass}/{len(CANDIDATES)}")
