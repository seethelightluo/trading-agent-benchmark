"""miner_2 2034-02-16 screening: novel cross-asset factor candidates.

Validates on warm-up window (2020-01-01..2026-07-15) for admission gates
|IC|>=0.007 |ICIR|>=0.084 and reports recent post-warm-up IC/ICIR for drift.
Library correlation is computed against ALL effective factor artifacts.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')

from factor_common import (WATCHLIST, load_prices, factor_to_panel,
                           VAL_START, VAL_END, canonical_grid, forward_returns)
import miner2_common as m2

prices = load_prices(days=4000)
grid = canonical_grid(prices)
artifacts = m2.load_effective_artifacts()
print(f"prices: {len(prices)} assets; canonical grid {grid.min().date()}..{grid.max().date()} n={len(grid)}")
print(f"effective library artifacts for corr audit: {len(artifacts)} -> {sorted(artifacts.keys())}")

RECENT_START = pd.Timestamp('2026-07-16')
RECENT_END = max(dd.index.max() for dd in prices.values())


def rank_ic_series_win(panel, fwd, start, end, min_valid=8):
    common = panel.index.intersection(fwd.index)
    ic = {}
    for d in common:
        if d < start or d > end:
            continue
        x = panel.loc[d]; y = fwd.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            ic[d] = x[m].rank().corr(y[m].rank())
    return pd.Series(ic).sort_index()


def validate_win(panel, prices, start, end, horizon=10, min_valid=8):
    fwd = forward_returns(prices, horizon)
    ic10 = rank_ic_series_win(panel, fwd, start, end, min_valid)
    if len(ic10) < 60:
        return None
    ic_mean = float(ic10.mean())
    ic_std = float(ic10.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())
    fac = panel[(panel.index >= start) & (panel.index <= end)]
    total = fac.shape[0] * fac.shape[1]
    cov = float(fac.notna().sum().sum()) / total if total else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= min_valid).mean())
    return {'ic': ic_mean, 'icir': icir, 'hit': hit, 'n': len(ic10),
            'cov': cov, 'ge8': ge8}


# ---------- candidate factor functions ----------
def f_kaufman_eff_60(df, s):
    c = df['close']
    r = c.pct_change()
    num = (c / c.shift(60) - 1.0).abs()
    den = r.abs().rolling(60).sum()
    return (num / den).replace([np.inf, -np.inf], np.nan)


def f_ret_autocorr_20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 5 else np.nan, raw=False)


def f_downside_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    neg = r.where(r < 0, 0.0)
    sd_all = r.rolling(60).std()
    sd_neg = neg.rolling(60).std()
    return (sd_neg / sd_all).replace([np.inf, -np.inf], np.nan)


def f_ret_skew_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).skew()


def f_kurtosis_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).kurt()


def f_body_ratio_20(df, s):
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return (body / rng).rolling(20).mean()


def f_upper_shadow_20(df, s):
    top = df['high'] - df[['open', 'close']].max(axis=1)
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return (top / rng).rolling(20).mean()


def f_vol_ratio_5_60(df, s):
    r = df['close'].pct_change()
    v5 = r.rolling(5).std()
    v60 = r.rolling(60).std()
    return (v5 / v60).replace([np.inf, -np.inf], np.nan)


def f_ret_vol_corr_20(df, s):
    r = df['close'].pct_change()
    v = df['volume'].pct_change()
    z = pd.concat([r, v], axis=1)
    return z.rolling(20).corr().iloc[0::2, 1].reset_index(drop=True).reindex(z.index).astype(float) if False else _roll_corr(z)


def _roll_corr(z):
    out = pd.Series(np.nan, index=z.index)
    rc = z['r'].rolling(20).corr(z['v'])
    return rc


CANDIDATES = [
    ('kaufman_eff_60', f_kaufman_eff_60, 'trend efficiency ratio 60d'),
    ('ret_autocorr_20', f_ret_autocorr_20, 'lag-1 return autocorrelation 20d'),
    ('downside_vol_ratio_60', f_downside_vol_ratio_60, 'downside semivol / total vol 60d'),
    ('ret_skew_60', f_ret_skew_60, '60d skewness of daily returns'),
    ('kurtosis_60', f_kurtosis_60, '60d excess kurtosis of daily returns'),
    ('body_ratio_20', f_body_ratio_20, 'mean candle body proportion 20d'),
    ('upper_shadow_20', f_upper_shadow_20, 'mean upper shadow proportion 20d'),
    ('vol_ratio_5_60', f_vol_ratio_5_60, 'short/long realized vol ratio 5x60'),
    ('ret_vol_corr_20', f_ret_vol_corr_20, 'corr(daily ret, volume pct change) 20d'),
]

results = {}
for fid, fn, desc in CANDIDATES:
    panel = factor_to_panel(fn, prices)
    if panel.empty:
        print(f"{fid}: EMPTY PANEL")
        continue
    wm = validate_win(panel, prices, VAL_START, VAL_END)
    rm = validate_win(panel, prices, RECENT_START, RECENT_END)
    rho, rho_id, _ = m2.max_library_correlation(panel, artifacts, grid)
    if wm is None:
        print(f"{fid}: insufficient warm data")
        continue
    ok = abs(wm['ic']) >= 0.007 and abs(wm['icir']) >= 0.084
    results[fid] = dict(warm=wm, recent=rm, rho=rho, rho_id=rho_id, desc=desc, ok=ok)
    print('-' * 90)
    print(f"{fid} [{desc}]")
    print(f"  warm  : IC={wm['ic']:+.4f} ICIR={wm['icir']:+.4f} hit={wm['hit']:.3f} n={wm['n']} cov={wm['cov']:.3f} ge8={wm['ge8']:.3f}")
    if rm:
        print(f"  recent: IC={rm['ic']:+.4f} ICIR={rm['icir']:+.4f} hit={rm['hit']:.3f} n={rm['n']} cov={rm['cov']:.3f}")
    else:
        print("  recent: insufficient")
    print(f"  max|library rho|={rho:.3f} vs {rho_id}   ADMISSION={'PASS' if ok else 'FAIL'}")

print('=' * 90)
print("SUMMARY:")
for fid, r in results.items():
    w = r['warm']
    print(f"{fid:24s} warm IC={w['ic']:+.4f} ICIR={w['icir']:+.4f} rho={r['rho']:.3f} {'PASS' if r['ok'] else 'fail'}")
