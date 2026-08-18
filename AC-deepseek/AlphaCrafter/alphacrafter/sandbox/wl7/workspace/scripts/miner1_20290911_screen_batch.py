"""miner_1 factor exploration cycle 2029-09-11.
Candidate factors for deep-bear / extreme-VIX regime (VIX 80+):
tail-risk, drawdown, oscillator, and downside-capture families.
Validation: daily cross-sectional Spearman IC at h=10 over full history
2020-01-01..2029-09-10, plus recent-window checks (2028-01-01.., 2029-01-01..).
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840 (15-instrument universe).
"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import factor_validator as fv

# Extend the validation window to current data
fv.LIVE_END = pd.Timestamp('2029-09-10')

WATCH = fv.WATCH
panel = fv.load_close_panel(days=2600)
print(f'panel shape {panel.shape}, dates {panel.index[0].date()}..{panel.index[-1].date()}')
print(f'panel last 3 dates: {panel.index[-3:].tolist()}')


def max_drawdown_60(s, w=60):
    roll_max = s.rolling(w, min_periods=20).max()
    dd = s / roll_max - 1.0
    return dd.rolling(w, min_periods=20).min()  # most negative drawdown depth


def cvar_60(s, w=60, q=0.10):
    ret = s.pct_change()
    out = ret.rolling(w, min_periods=30).apply(
        lambda x: np.mean(np.sort(x)[: max(1, int(len(x) * q))]), raw=True)
    return out


def downside_ratio_60(s, w=60):
    ret = s.pct_change()
    dn = ret.where(ret < 0)
    up = ret.where(ret > 0)
    dvol = dn.rolling(w, min_periods=30).std()
    uvol = up.rolling(w, min_periods=30).std()
    return dvol / uvol


def range_pos_20(s, w=20):
    hi = s.rolling(w, min_periods=10).max()
    lo = s.rolling(w, min_periods=10).min()
    return (s - lo) / (hi - lo)


def zscore_60(s, w=60):
    mu = s.rolling(w, min_periods=30).mean()
    sd = s.rolling(w, min_periods=30).std()
    return (s - mu) / sd


def skew_20(s, w=20):
    ret = s.pct_change()
    return ret.rolling(w, min_periods=10).skew()


def recovery_10(s, w=60):
    roll_max = s.rolling(w, min_periods=20).max()
    dd = s / roll_max - 1.0
    dd_min = dd.rolling(w, min_periods=20).min()
    ret10 = s.pct_change(10)
    # speed of bounce relative to drawdown depth (positive = recovering)
    return ret10 / (-dd_min)


def vol_scaled_rev_20(s, w=20, v=60):
    """20d reversal scaled by 60d vol (contrarian tilt for whipsaw tape)."""
    mom = s.pct_change(w)
    vol = s.pct_change().rolling(v, min_periods=30).std()
    return -mom / vol


def ulcer_60(s, w=60):
    """Ulcer index: sqrt(mean(squared drawdown)) over window."""
    roll_max = s.rolling(w, min_periods=20).max()
    dd = (s / roll_max - 1.0) * 100.0
    return np.sqrt((dd ** 2).rolling(w, min_periods=20).mean())


def hi_lo_vol_20(s, w=20):
    """Parkinson-style range vol: ln(H/L) avg, low = calm."""
    # approximate with close-based range
    hi = s.rolling(w, min_periods=10).max()
    lo = s.rolling(w, min_periods=10).min()
    return (hi / lo - 1.0).rolling(w, min_periods=10).mean()


CANDIDATES = {
    'dd_60d': (max_drawdown_60, {'w': 60}),
    'cvar_60d': (cvar_60, {'w': 60, 'q': 0.10}),
    'downside_ratio_60d': (downside_ratio_60, {'w': 60}),
    'range_pos_20d': (range_pos_20, {'w': 20}),
    'zscore_60d': (zscore_60, {'w': 60}),
    'skew_20d': (skew_20, {'w': 20}),
    'recovery_10d': (recovery_10, {'w': 60}),
    'vol_scaled_rev_20x60': (vol_scaled_rev_20, {'w': 20, 'v': 60}),
    'ulcer_60d': (ulcer_60, {'w': 60}),
    'hi_lo_vol_20d': (hi_lo_vol_20, {'w': 20}),
}

results = {}
for fid, (fn, kw) in CANDIDATES.items():
    fdf = fv.apply_factor_per_asset(panel, lambda s, fn=fn, kw=kw: fn(s, **kw))
    fwd10 = fv.fwd_returns(panel, 10)
    ic_all = fv.cross_sectional_ic(fdf, fwd10)
    s_all = fv.summarize_ic(ic_all, 'full')
    # recent windows
    for label, cut in [('since2028', '2028-01-01'), ('since2029', '2029-01-01'), ('last500', None)]:
        if label == 'last500' and len(ic_all) > 500:
            ic_w = ic_all.iloc[-500:]
        else:
            ic_w = ic_all[ic_all.index >= cut]
        sw = fv.summarize_ic(ic_w, label)
        s_all[label] = sw
    cov = fv.coverage_stats(fdf)
    s_all['coverage'] = cov
    lib = fv.max_library_corr(fdf)
    s_all['max_lib_corr'] = lib
    results[fid] = s_all
    print(f"--- {fid} ---")
    print(f"  full: IC={s_all['ic']:.4f} ICIR={s_all['icir']:.3f} hit={s_all['ic_hit_ratio']:.3f} n={s_all['n_dates']}")
    print(f"  since2028: IC={s_all['since2028']['ic']:.4f} ICIR={s_all['since2028']['icir']:.3f} n={s_all['since2028']['n_dates']}")
    print(f"  since2029: IC={s_all['since2029']['ic']:.4f} ICIR={s_all['since2029']['icir']:.3f} n={s_all['since2029']['n_dates']}")
    if s_all.get('last500'):
        print(f"  last500: IC={s_all['last500']['ic']:.4f} ICIR={s_all['last500']['icir']:.3f} n={s_all['last500']['n_dates']}")
    print(f"  coverage: asset_days={cov['coverage_asset_days']:.3f} dates_ge8={cov['coverage_dates_ge8']:.3f}")
    print(f"  max_lib_corr: {lib}")

print('\n=== GATE CHECK (full-period) ===')
for fid, r in results.items():
    gate_ic = abs(r['ic']) >= 0.0070
    gate_icir = abs(r['icir']) >= 0.0840
    print(f"{fid}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} | IC_gate={gate_ic} ICIR_gate={gate_icir} -> {'PASS' if (gate_ic and gate_icir) else 'FAIL'}")
