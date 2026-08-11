# -*- coding: utf-8 -*-
"""miner_2 2027-02-25 cycle: screen novel factor candidates (v4).
Dimensions: rate beta (US10Y/CN10Y), serial dependence, gap structure,
gold linkage, candlestick/range structure. NaN-aware rolling windows.
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

print(f"Panel dates: {C.index.min().date()} -> {C.index.max().date()}, {len(C)} rows, {C.shape[1]} assets")
print("Recent 10d log returns by asset (through last visible day):")
print((LR.tail(10).sum().sort_values(ascending=False).round(4).to_string()))

MP = lambda w: int(w * 0.5)

def _clean(a):
    a = np.asarray(a, dtype=float)
    return a[~np.isnan(a)]

def _apply_win(panel, w, fn):
    return panel.rolling(w, min_periods=MP(w)).apply(
        lambda a: fn(_clean(a)), raw=True)

# ---------- 1) rate beta (US10Y / CN10Y sensitivity) ----------
def rolling_ref_beta(asset_r, ref_r, w, cond_series=None):
    ref_df = pd.DataFrame({c: ref_r for c in asset_r.columns}, index=asset_r.index)
    out = pd.DataFrame(index=asset_r.index, columns=asset_r.columns, dtype=float)
    a = asset_r if cond_series is None else asset_r.where(cond_series)
    r = ref_df if cond_series is None else ref_df.where(cond_series)
    for c in asset_r.columns:
        if c == ref_r.name:
            out[c] = np.nan  # self-beta excluded
            continue
        cov = a[c].rolling(w, min_periods=MP(w)).cov(r[c])
        var = r[c].rolling(w, min_periods=MP(w)).var()
        out[c] = cov / var.replace(0, np.nan)
    return out

def rate_beta_60(w=60):
    return rolling_ref_beta(R, R['US10Y'], w)

def rate_beta_cond_up_60(w=60):
    cond = R['US10Y'] > 0
    return rolling_ref_beta(R, R['US10Y'], w, cond)

def cn_rate_beta_60(w=60):
    return rolling_ref_beta(R, R['CN10Y'], w)

# ---------- 2) serial dependence ----------
def autocorr_1_60(w=60):
    def f(a):
        if len(a) < 5:
            return np.nan
        x, y = a[:-1], a[1:]
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])
    return _apply_win(R, w, f)

def sign_persist_20(w=20):
    def f(a):
        if len(a) < 5:
            return np.nan
        s = np.sign(a)
        return float((s[:-1] == s[1:]).mean())
    return _apply_win(R, w, f)

# ---------- 3) gap structure (OHLC) ----------
def overnight_abs_share_20(w=20):
    ov = (np.log(O) - np.log(C.shift(1))).abs()
    tot = LR.abs()
    so = ov.rolling(w, min_periods=MP(w)).sum()
    st = tot.rolling(w, min_periods=MP(w)).sum()
    return (so / st.replace(0, np.nan)).where(O.notna())

def overnight_signed_share_20(w=20):
    ov = np.log(O) - np.log(C.shift(1))
    tot = LR.abs()
    so = ov.rolling(w, min_periods=MP(w)).sum()
    st = tot.rolling(w, min_periods=MP(w)).sum()
    return (so / st.replace(0, np.nan)).where(O.notna())

def gap_follow_20(w=20):
    ov = np.log(O) - np.log(C.shift(1))
    intr = np.log(C) - np.log(O)
    return ov.rolling(w, min_periods=MP(w)).corr(intr).where(O.notna())

# ---------- 4) gold linkage ----------
def xau_beta_60(w=60):
    return rolling_ref_beta(R, R['XAU'], w)

# ---------- 5) candlestick / range structure ----------
def close_pos_20(w=20):
    rng = (H - Lo).replace(0, np.nan)
    cp = (C - Lo) / rng
    return cp.rolling(w, min_periods=MP(w)).mean()

def upper_shadow_20(w=20):
    rng = (H - Lo).replace(0, np.nan)
    us = (H - np.maximum(O, C)) / rng
    return us.rolling(w, min_periods=MP(w)).mean()

def range_trend_60(w_recent=20, w_old=60):
    rng = (H - Lo) / C
    recent = rng.rolling(w_recent, min_periods=MP(w_recent)).mean()
    older = rng.rolling(w_old, min_periods=MP(w_old)).mean()
    return recent / older.replace(0, np.nan) - 1.0

CANDIDATES = {
    'rate_beta_60': rate_beta_60,
    'rate_beta_cond_up_60': rate_beta_cond_up_60,
    'cn_rate_beta_60': cn_rate_beta_60,
    'autocorr_1_60': autocorr_1_60,
    'sign_persist_20': sign_persist_20,
    'overnight_abs_share_20': overnight_abs_share_20,
    'overnight_signed_share_20': overnight_signed_share_20,
    'gap_follow_20': gap_follow_20,
    'xau_beta_60': xau_beta_60,
    'close_pos_20': close_pos_20,
    'upper_shadow_20': upper_shadow_20,
    'range_trend_60': range_trend_60,
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
        if gate:
            print("  >>> PASSES GATE - must persist")
    except Exception as e:
        RESULTS[name] = {'error': str(e)}
        print(f"\n=== {name} ERROR: {e}")

with open('scripts/miner_2_20270225_screen_results.json', 'w') as f:
    json.dump(RESULTS, f, indent=1, default=str)
print("\nSaved scripts/miner_2_20270225_screen_results.json")
npass = sum(1 for v in RESULTS.values() if isinstance(v, dict) and v.get('PASS'))
print(f"PASSING candidates: {npass}/{len(CANDIDATES)}")
