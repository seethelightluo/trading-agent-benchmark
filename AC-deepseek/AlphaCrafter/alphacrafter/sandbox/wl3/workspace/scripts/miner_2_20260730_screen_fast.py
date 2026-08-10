"""miner_2 2026-07-30 fast batch screen: numpy rank-IC validation for 14 candidates.

Speeds up the previous pandas .loc+rank().corr() loop by ~100x with numpy
per-date rank correlation. Run in slices:
    python scripts/miner_2_20260730_screen_fast.py <start> <end>
"""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
from factor_common import (load_prices, load_index, WATCHLIST, VAL_START, VAL_END,
                           canonical_grid, build_library_panels, factor_to_panel,
                           forward_returns)

prices = load_prices(days=2000)
dxy = load_index('DXY', prices=prices)
grid = canonical_grid(prices)
lib_panels = build_library_panels(prices)
lib_mat = {}
for fid, lp in lib_panels.items():
    m = lp.reindex(grid)
    lib_mat[fid] = m[WATCHLIST].values.astype(float)


def rankdata_avg(x):
    """Average ranks (1..n) for a 1D array, tie-aware, numpy only."""
    n = len(x)
    sorter = np.argsort(x, kind='mergesort')
    inv = np.empty(n, dtype=np.intp)
    inv[sorter] = np.arange(n)
    x_sorted = x[sorter]
    obs = np.r_[True, x_sorted[1:] != x_sorted[:-1]]
    dense = obs.cumsum() - 1
    starts = np.nonzero(obs)[0] + 1
    counts = np.diff(np.r_[np.nonzero(obs)[0], n])
    avg = starts + (counts - 1) / 2.0
    return avg[dense][inv]


def spearman_rows(xm, ym, min_valid=8):
    """Per-row Spearman between two (n_dates, n_assets) matrices. Returns list of floats."""
    out = []
    for i in range(xm.shape[0]):
        x = xm[i]
        y = ym[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < min_valid:
            out.append(np.nan)
            continue
        xr = rankdata_avg(x[m])
        yr = rankdata_avg(y[m])
        xr = xr - xr.mean()
        yr = yr - yr.mean()
        denom = np.sqrt((xr * xr).sum() * (yr * yr).sum())
        out.append(float((xr * yr).sum() / denom) if denom > 0 else np.nan)
    return out


def ic_metrics(ic_arr):
    ic = ic_arr[np.isfinite(ic_arr)]
    if len(ic) < 100:
        return None
    ic_mean = float(ic.mean())
    ic_std = float(ic.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic > 0).mean()) if ic_mean >= 0 else float((ic < 0).mean())
    return ic_mean, icir, hit, len(ic)


def evaluate_fast(fid, panel):
    fac = panel.reindex(grid)[WATCHLIST].values.astype(float)
    fwd = {h: forward_returns(prices, h).reindex(grid)[WATCHLIST].values.astype(float)
           for h in (1, 2, 3, 5, 10, 20)}
    ics = {h: np.array(spearman_rows(fac, fwd[h])) for h in fwd}
    m10 = ic_metrics(ics[10])
    if m10 is None:
        return None
    ic_mean, icir, hit, n = m10
    # coverage within window
    valid_cells = int(np.isfinite(fac).sum())
    total_cells = fac.shape[0] * fac.shape[1]
    ge8 = float((np.isfinite(fac).sum(axis=1) >= 8).mean())
    # turnover: mean abs rank change over 10-day steps
    fac_df = panel.reindex(grid)
    ranked = fac_df.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    # library correlation (mean daily cross-sectional Spearman)
    best, best_id = 0.0, None
    for fid_l, lm in lib_mat.items():
        c = np.array(spearman_rows(fac, lm))
        c = c[np.isfinite(c)]
        if len(c):
            r = float(np.mean(c))
            if abs(r) > best:
                best, best_id = abs(r), fid_l
    decay = {str(h): float(np.nanmean(ics[h])) for h in fwd}
    return {
        'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit, 'n_ic_dates': n,
        'coverage_asset_days': valid_cells / total_cells, 'coverage_dates_ge8': ge8,
        'turnover_10d_rank': turn, 'max_abs_library_correlation': best,
        'max_corr_library_id': best_id, 'decay_ic_by_horizon': decay,
    }


def f_rsi_14(df, s):
    close = df['close']
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).replace([np.inf], np.nan)


def f_bollinger_z_20(df, s):
    close = df['close']
    sma = close.rolling(20).mean()
    sd = close.rolling(20).std()
    return (close - sma) / sd.replace(0, np.nan)


def f_updown_ratio_60(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0).rolling(60).mean()
    dn = (-r.where(r < 0)).rolling(60).mean()
    return (up / dn.replace(0, np.nan)).replace([np.inf], np.nan)


def f_mom_ratio_120_20(df, s):
    close = df['close']
    m120 = close.shift(5) / close.shift(125) - 1.0
    m20 = close.shift(5) / close.shift(25) - 1.0
    return m120 / m20.abs().replace(0, np.nan)


def f_eff_ratio_60(df, s):
    close = df['close']
    num = (close - close.shift(60)).abs()
    den = close.pct_change().abs().rolling(60).sum()
    return num / den.replace(0, np.nan)


def f_downside_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    neg = r.clip(upper=0)
    return neg.rolling(60).std() / r.rolling(60).std().replace(0, np.nan)


def f_skew_60(df, s):
    return df['close'].pct_change().rolling(60).skew()


def f_dxy_beta_cond_60x20(df, s):
    if dxy is None:
        return None
    r = df['close'].pct_change()
    dr = dxy['close'].pct_change()
    z = pd.concat([r.rename('r'), dr.rename('d')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var().replace(0, np.nan)
    dxy_move = dxy['close'] / dxy['close'].shift(20) - 1.0
    return (b * dxy_move).reindex(z.index)


def f_gw_high_252(df, s):
    roll_max = df['close'].rolling(252, min_periods=60).max()
    return df['close'] / roll_max - 1.0


def f_vol_adj_mom_20_60(df, s):
    close = df['close']
    mom = close.shift(5) / close.shift(25) - 1.0
    vol = close.pct_change().rolling(60).std()
    return mom / vol.replace(0, np.nan)


def f_vol_term_20_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).std() / r.rolling(60).std().replace(0, np.nan) - 1.0


def f_ret_zscore_20(df, s):
    close = df['close']
    r = close.pct_change()
    mu = r.rolling(20).mean()
    sd = r.rolling(20).std()
    return ((r - mu) / sd.replace(0, np.nan)).shift(1)


def f_vol_ratio_20_120(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).std() / r.rolling(120).std().replace(0, np.nan) - 1.0


def f_high_low_range_pos_20(df, s):
    close = df['close']
    hi = df['high'].rolling(20).max()
    lo = df['low'].rolling(20).min()
    return (close - lo) / (hi - lo).replace(0, np.nan)


CANDIDATES = [
    ("rsi_14d", f_rsi_14),
    ("bollinger_z_20d", f_bollinger_z_20),
    ("updown_ratio_60d", f_updown_ratio_60),
    ("mom_ratio_120_20", f_mom_ratio_120_20),
    ("eff_ratio_60d", f_eff_ratio_60),
    ("downside_vol_ratio_60x20", f_downside_vol_ratio_60),
    ("skew_60d", f_skew_60),
    ("dxy_beta_cond_60x20", f_dxy_beta_cond_60x20),
    ("gw_high_252", f_gw_high_252),
    ("vol_adj_mom_20_60", f_vol_adj_mom_20_60),
    ("vol_term_20_60", f_vol_term_20_60),
    ("ret_zscore_20d", f_ret_zscore_20),
    ("vol_ratio_20_120", f_vol_ratio_20_120),
    ("high_low_range_pos_20", f_high_low_range_pos_20),
]

if __name__ == '__main__':
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(CANDIDATES)
    slice_ = CANDIDATES[start:end]
    print(f"grid dates: {len(grid)} ({grid.min().date()}..{grid.max().date()}), assets: {len(WATCHLIST)}")
    print(f"slice [{start}:{end}] of {len(CANDIDATES)} candidates")
    for fid, fn in slice_:
        try:
            panel = factor_to_panel(fn, prices)
            m = evaluate_fast(fid, panel)
        except Exception as exc:
            print(f"{fid:28s} ERROR {exc}")
            continue
        if m is None:
            print(f"{fid:28s} INSUFFICIENT")
            continue
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"{fid:28s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']:5d} cov={m['coverage_asset_days']:.2f} "
              f"ge8={m['coverage_dates_ge8']:.2f} turn={m['turnover_10d_rank']:.2f} "
              f"rho_lib={m['max_abs_library_correlation']:.2f} vs {m['max_corr_library_id']} -> {'PASS' if ok else 'FAIL'}")
        d = m['decay_ic_by_horizon']
        print(f"{'':28s} decay " + " ".join(f"h{h}:{d[str(h)]:+.4f}" for h in [1, 2, 3, 5, 10, 20]))
