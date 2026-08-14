"""miner_3 2035-04-26 screening: risk-shape / statistical-moment factor family.

Admission on warm-up window (2020-01-01..2026-07-15) with |IC|>=0.007 |ICIR|>=0.084
at h=10; reports recent (2026-07-16..) IC/ICIR for drift.
Library correlation audited against ALL effective factor artifacts (same-shape only).
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')

from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           VAL_START, VAL_END, canonical_grid, forward_returns)
import miner2_common as m2

prices = load_prices(days=4200)
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


# ---------------- candidate factor definitions ----------------

def f_kurtosis_20(df, s):
    """Excess kurtosis of daily returns over 20d (fat-tail persistence)."""
    r = df['close'].pct_change()
    return r.rolling(20, min_periods=12).kurt()


def f_autocorr_20(df, s):
    """Lag-1 autocorrelation of daily returns over 20d (trend vs mean-reversion)."""
    r = df['close'].pct_change()
    return r.rolling(20, min_periods=12).apply(lambda z: z.autocorr() if len(z) > 3 else np.nan, raw=False)


def f_downside_vol_ratio_60(df, s):
    """Downside semideviation / total vol over 60d (loss asymmetry)."""
    r = df['close'].pct_change()
    neg = r.clip(upper=0.0)
    def f(z):
        z = np.asarray(z, dtype=float)
        z = z[np.isfinite(z)]
        if len(z) < 30:
            return np.nan
        total = np.std(z, ddof=1)
        if total <= 0:
            return np.nan
        d = z[z < 0]
        if len(d) < 5:
            return np.nan
        semi = np.sqrt(np.mean((d - np.mean(z)) ** 2))
        return semi / total
    return r.rolling(60, min_periods=30).apply(f, raw=True)


def f_parkinson_ratio_20(df, s):
    """Parkinson (high-low range) vol / close-close vol over 20d (intraday efficiency)."""
    r = df['close'].pct_change()
    cc_vol = r.rolling(20, min_periods=10).std()
    hl = np.log(df['high'] / df['low'])
    pk_vol = np.sqrt((hl ** 2).rolling(20, min_periods=10).mean() / (4 * np.log(2)))
    return (pk_vol / cc_vol).replace([np.inf, -np.inf], np.nan)


def f_max_dd_depth_60(df, s):
    """Max drawdown depth over 60d (negative, larger = deeper)."""
    c = df['close']
    roll_max = c.rolling(60, min_periods=30).max()
    dd = c / roll_max - 1.0
    return dd.rolling(60, min_periods=30).min()


def f_coskew_spx_60(df, s):
    """Coskewness of asset returns with SPX over 60d: E[r_i * r_m^2] style (higher-order systematic risk)."""
    spx = prices['SPX']['close'].pct_change().rename('m')
    r = df['close'].pct_change().rename('r')
    z = pd.concat([r, spx], axis=1).dropna()
    def f(rr, mm):
        if len(rr) < 40:
            return np.nan
        ri = rr - rr.mean()
        mi = mm - mm.mean()
        denom = (np.std(ri, ddof=1) * np.var(mi, ddof=1))
        if not np.isfinite(denom) or denom == 0:
            return np.nan
        return float(np.mean(ri * mi ** 2) / denom)
    return z['r'].rolling(60, min_periods=40).apply(lambda x: f(x, z['m'].loc[x.index]), raw=False)


def f_vol_trend_20_60(df, s):
    """Short vol (20d) / long vol (60d) ratio (vol regime momentum)."""
    r = df['close'].pct_change()
    v20 = r.rolling(20, min_periods=10).std()
    v60 = r.rolling(60, min_periods=30).std()
    return (v20 / v60).replace([np.inf, -np.inf], np.nan)


def f_tail_ratio_20(df, s):
    """Tail ratio: 95th pct abs return / median abs return over 20d (tail mass)."""
    r = df['close'].pct_change().abs()
    def f(z):
        z = np.asarray(z, dtype=float)
        z = z[np.isfinite(z)]
        if len(z) < 12:
            return np.nan
        q95 = np.percentile(z, 95)
        med = np.percentile(z, 50)
        if med <= 0:
            return np.nan
        return q95 / med
    return r.rolling(20, min_periods=12).apply(f, raw=True)


def f_range_pos_vol_adj_20(df, s):
    """Position within 20d range divided by 20d vol (trend location, vol-normalized)."""
    c = df['close']
    hi = df['high'].rolling(20, min_periods=10).max()
    lo = df['low'].rolling(20, min_periods=10).min()
    pos = (c - lo) / (hi - lo).replace(0, np.nan)
    v = df['close'].pct_change().rolling(20, min_periods=10).std()
    return (pos / v).replace([np.inf, -np.inf], np.nan)


CANDIDATES = [
    ('kurtosis_20', f_kurtosis_20, 'Excess kurtosis of 20d daily returns'),
    ('autocorr_20', f_autocorr_20, 'Lag-1 autocorrelation of 20d daily returns'),
    ('downside_vol_ratio_60', f_downside_vol_ratio_60, 'Downside semideviation / total vol 60d'),
    ('parkinson_ratio_20', f_parkinson_ratio_20, 'Parkinson range vol / close-close vol 20d'),
    ('max_dd_depth_60', f_max_dd_depth_60, 'Max drawdown depth over 60d'),
    ('coskew_spx_60', f_coskew_spx_60, 'Coskewness with SPX 60d'),
    ('vol_trend_20_60', f_vol_trend_20_60, 'Short/long vol ratio 20/60'),
    ('tail_ratio_20', f_tail_ratio_20, '95pct/median abs return ratio 20d'),
    ('range_pos_vol_adj_20', f_range_pos_vol_adj_20, '20d range position / 20d vol'),
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
    ric = 'n/a' if r['recent'] is None else f"{r['recent']['ic']:+.4f}"
    print(f"{fid:26s} warm IC={w['ic']:+.4f} ICIR={w['icir']:+.4f} | recent IC={ric} | rho={r['rho']:.3f} | {'PASS' if r['ok'] else 'FAIL'}")
