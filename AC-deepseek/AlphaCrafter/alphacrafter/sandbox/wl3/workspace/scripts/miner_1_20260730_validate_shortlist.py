"""Dedicated full validation battery for shortlisted candidates from screening.

For each candidate: IC/ICIR on 10d horizon, decay profile, coverage, turnover,
and max absolute correlation vs the 4 library factors (provenance audit).
"""
import sys, json
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate, build_library_panels

prices = load_prices(days=2200)
library = build_library_panels(prices)
print('library panels:', {k: v.shape for k, v in library.items()})

# --- candidate 1: vol-price correlation 60d (novel volume-price factor) ---
def vol_price_corr_60(df, s):
    r = df['close'].pct_change()
    vc = df['volume'].pct_change()
    return r.rolling(60).corr(vc)

# --- candidate 2: risk-adjusted momentum 20d/vol20 ---
def mom20_risk_adj(df, s):
    m = df['close'].shift(5) / df['close'].shift(25) - 1.0
    v = df['close'].pct_change().rolling(20).std()
    return m / v.replace(0, __import__('numpy').nan)

# --- candidate 3: RSI-14 ---
def rsi_14(df, s):
    import numpy as np
    d = df['close'].diff()
    up = d.clip(lower=0.0).rolling(14).mean()
    dn = (-d.clip(upper=0.0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

# --- candidate 4: conditional momentum in low-vol regime ---
def mom10_cond_lowvol(df, s):
    m = df['close'].shift(5) / df['close'].shift(15) - 1.0
    v = df['close'].pct_change().rolling(20).std()
    lowvol = (v < v.rolling(60).median()).astype(float)
    return m * lowvol

# --- candidate 5: downside deviation ratio 20d ---
def downside_ratio_20(df, s):
    r = df['close'].pct_change()
    dd = r.clip(upper=0.0).rolling(20).std()
    td = r.rolling(20).std()
    return dd / td.replace(0, __import__('numpy').nan)

# --- candidate 6: z-score 20d (Bollinger position) ---
def zscore_20(df, s):
    m = df['close'].rolling(20).mean(); sd = df['close'].rolling(20).std()
    return (df['close'] - m) / sd

for fid, fn in [
    ('vol_price_corr_60', vol_price_corr_60),
    ('mom20_risk_adj', mom20_risk_adj),
    ('rsi_14', rsi_14),
    ('mom10_cond_lowvol', mom10_cond_lowvol),
    ('downside_ratio_20', downside_ratio_20),
    ('zscore_20', zscore_20),
]:
    print('#' * 70)
    m = evaluate_candidate(fid, fn, prices, library_panels=library)
    if m is not None:
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f'FINAL: {fid} {"PASS" if ok else "FAIL"}  max_abs_lib_corr={m["max_abs_library_correlation"]:.3f} ({m["max_corr_library_id"]})')
