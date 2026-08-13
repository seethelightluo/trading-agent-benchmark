"""miner_3 2034-03-02 screening: novel cross-asset beta factor family.

Admission on warm-up window (2020-01-01..2026-07-15) with |IC|>=0.007 |ICIR|>=0.084
at h=10; reports recent (2026-07-16..2034-03-01) IC/ICIR for drift.
Library correlation audited against ALL effective factor artifacts (same-shape only).
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')

from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           VAL_START, VAL_END, canonical_grid, forward_returns)
import miner2_common as m2

prices = load_prices(days=4000)
grid = canonical_grid(prices)
artifacts = m2.load_effective_artifacts()
RECENT_START = pd.Timestamp('2026-07-16')
RECENT_END = max(dd.index.max() for dd in prices.values())
print(f"prices: {len(prices)} assets; canonical grid {grid.min().date()}..{grid.max().date()} n={len(grid)}")
print(f"recent window end: {RECENT_END.date()}")
print(f"effective artifacts for corr audit: {len(artifacts)} -> {sorted(artifacts.keys())}")


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


def make_beta_fn(sig_series, window, cond=None, sign=None):
    """sig_series: pd.Series of signal close prices indexed by date."""
    def fn(df, s):
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), sig_series.pct_change().rename('s')], axis=1).dropna()
        if cond is not None:
            z = z[cond(z)]
        if len(z) < window:
            return pd.Series(np.nan, index=df.index)
        b = z['r'].rolling(window, min_periods=max(20, window // 2)).cov(z['s']) / \
            z['s'].rolling(window, min_periods=max(20, window // 2)).var()
        return b.reindex(df.index)
    return fn


# --- observation signals ---
usdjpy = load_index('USDJPY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
dxy = load_index('DXY', prices=prices)
print(f"signals: USDJPY n={0 if usdjpy is None else len(usdjpy)} "
      f"EURUSD n={0 if eurusd is None else len(eurusd)} DXY n={0 if dxy is None else len(dxy)}")

# --- candidate factor functions ---
def f_wti_beta_60(df, s):
    sig = prices['WTI']['close']
    return make_beta_fn(sig, 60)(df, s)

def f_btc_beta_60(df, s):
    sig = prices['BTC']['close']
    return make_beta_fn(sig, 60)(df, s)

def f_xau_beta_60(df, s):
    sig = prices['XAU']['close']
    return make_beta_fn(sig, 60)(df, s)

def f_us10y_beta_60(df, s):
    sig = prices['US10Y']['close']
    return make_beta_fn(sig, 60)(df, s)

def f_usdjpy_beta_cond_60x20(df, s):
    if usdjpy is None:
        return None
    sig = usdjpy['close']
    def cond(z): return z['s'] > 0  # yen depreciation (risk-on) days
    return make_beta_fn(sig, 60, cond=cond)(df, s)

def f_usdjpy_beta_down_60x20(df, s):
    if usdjpy is None:
        return None
    sig = usdjpy['close']
    def cond(z): return z['s'] < 0  # yen appreciation (risk-off) days
    return make_beta_fn(sig, 60, cond=cond)(df, s)

def f_rate_slope_beta_60(df, s):
    slope = (prices['US10Y']['close'] - prices['CN10Y']['close'])
    return make_beta_fn(slope, 60)(df, s)

def f_eurusd_beta_uncond_60(df, s):
    if eurusd is None:
        return None
    sig = eurusd['close']
    return make_beta_fn(sig, 60)(df, s)

CANDIDATES = [
    ('wti_beta_60', f_wti_beta_60, 'beta to WTI returns 60d'),
    ('btc_beta_60', f_btc_beta_60, 'beta to BTC returns 60d'),
    ('xau_beta_60', f_xau_beta_60, 'beta to XAU returns 60d'),
    ('us10y_beta_60', f_us10y_beta_60, 'beta to US10Y yield-change 60d'),
    ('usdjpy_beta_cond_60x20', f_usdjpy_beta_cond_60x20, 'beta to USDJPY on JPY-depreciation days 60d'),
    ('usdjpy_beta_down_60x20', f_usdjpy_beta_down_60x20, 'beta to USDJPY on JPY-appreciation days 60d'),
    ('rate_slope_beta_60', f_rate_slope_beta_60, 'beta to US10Y-CN10Y slope change 60d'),
    ('eurusd_beta_uncond_60', f_eurusd_beta_uncond_60, 'unconditional beta to EURUSD 60d'),
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
        print(f"  recent: IC={rm['ic']:+.4f} ICIR={rm['icir']:+.4f} hit={rm['hit']:.3f} n={rm['n']} cov={rm['cov']:.3f} ge8={rm['ge8']:.3f}")
    else:
        print(f"  recent: insufficient")
    print(f"  corr  : max_abs_rho={rho:.3f} vs {rho_id}")
    print(f"  ADMISSION (warm |IC|>=0.007 & |ICIR|>=0.084): {'PASS' if ok else 'FAIL'}")

print()
print("=" * 90)
print("SUMMARY")
for fid, r in results.items():
    w = r['warm']
    print(f"{fid:28s} warm IC={w['ic']:+.4f} ICIR={w['icir']:+.4f} | recent "
          f"IC={None if r['recent'] is None else f'{r[\"recent\"][\"ic\"]:+.4f}'} | rho={r['rho']:.3f} | {'PASS' if r['ok'] else 'FAIL'}")
