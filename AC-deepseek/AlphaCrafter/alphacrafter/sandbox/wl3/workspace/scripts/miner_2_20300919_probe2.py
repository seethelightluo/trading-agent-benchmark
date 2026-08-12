"""miner_2 2030-09-19: follow-up probe - fix ups_beta_spread_60 + test more
distinct cross-asset candidates: gold_beta_60, usdjpy_sens_60, usdcny_sens_60.
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           validate_factor, canonical_grid, signal_matrix,
                           VAL_START, VAL_END)

prices = load_prices(days=3500)
spx = prices['SPX']['close']
btc = prices['BTC']['close']
xau = prices['XAU']['close']
usdjpy = load_index('USDJPY', prices=prices)
usdcny = load_index('USDCNY', prices=prices)

grid = canonical_grid(prices)
lib_art = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    fid = p.name.replace('_signal.npy', '')
    try:
        arr = np.load(p, allow_pickle=False)
        if arr.shape[0] == len(grid) and arr.shape[1] == 15:
            lib_art[fid] = arr
    except Exception:
        pass
print(f"library artifacts matched to grid: {len(lib_art)}")


def max_lib_corr(panel):
    mtx = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, arr in lib_art.items():
        if arr.shape != mtx.shape:
            continue
        corrs = []
        for i in range(len(grid)):
            x, y = mtx[i], arr[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                from scipy.stats import spearmanr
                c = spearmanr(x[m], y[m]).statistic
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


def f_ups_beta_spread_60(df, s):
    """Convexity: upside beta - downside beta vs SPX over 60d (min_periods=25)."""
    r = df['close'].pct_change()
    m = spx.pct_change()
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    up = z['m'] > 0
    dn = z['m'] < 0
    cov_up = z['r'].where(up).rolling(60, min_periods=25).cov(z['m'].where(up))
    var_up = z['m'].where(up).rolling(60, min_periods=25).var()
    cov_dn = z['r'].where(dn).rolling(60, min_periods=25).cov(z['m'].where(dn))
    var_dn = z['m'].where(dn).rolling(60, min_periods=25).var()
    b_up = cov_up / var_up.replace(0, np.nan)
    b_dn = cov_dn / var_dn.replace(0, np.nan)
    return (b_up - b_dn).reindex(df.index)


def f_gold_beta_60(df, s):
    """60d rolling beta of asset returns to XAU returns (safe-haven linkage)."""
    r = df['close'].pct_change()
    m = xau.pct_change()
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=40).cov(z['m']) / z['m'].rolling(60, min_periods=40).var().replace(0, np.nan)
    return b.reindex(df.index)


def f_usdjpy_sens_60(df, s):
    """60d rolling beta of asset returns to USDJPY changes (global risk-on proxy)."""
    if usdjpy is None:
        return None
    r = df['close'].pct_change()
    m = usdjpy['close'].pct_change()
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=40).cov(z['m']) / z['m'].rolling(60, min_periods=40).var().replace(0, np.nan)
    return b.reindex(df.index)


def f_usdcny_sens_60(df, s):
    """60d rolling beta of asset returns to USDCNY changes (China FX stress proxy)."""
    if usdcny is None:
        return None
    r = df['close'].pct_change()
    m = usdcny['close'].pct_change()
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=40).cov(z['m']) / z['m'].rolling(60, min_periods=40).var().replace(0, np.nan)
    return b.reindex(df.index)


CANDIDATES = [
    ('ups_beta_spread_60', f_ups_beta_spread_60, 'convexity'),
    ('gold_beta_60', f_gold_beta_60, 'cross_asset'),
    ('usdjpy_sens_60', f_usdjpy_sens_60, 'macro'),
    ('usdcny_sens_60', f_usdcny_sens_60, 'macro'),
]

results = {}
for fid, fn, tag in CANDIDATES:
    print(f"\n===== {fid} ({tag}) =====")
    panel = factor_to_panel(fn, prices)
    if panel.empty:
        print("empty panel")
        results[fid] = None
        continue
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: insufficient data -> None (panel dates {panel.index.min()}..{panel.index.max()})")
        results[fid] = None
        continue
    rho, rho_id = max_lib_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"panel {panel.shape}  dates {panel.index.min().date()}..{panel.index.max().date()}")
    print(f"IC10={m['ic']:+.4f}  ICIR10={m['icir']:+.3f}  hit={m['ic_hit_ratio']:.3f}  n={m['n_ic_dates']}")
    print(f"coverage={m['coverage_asset_days']:.3f}  ge8={m['coverage_dates_ge8']:.3f}  turnover={m['turnover_10d_rank']:.3f}")
    print(f"decay: " + json.dumps({k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()}))
    print(f"max_lib_corr={rho:.3f} (vs {rho_id})")
    print(f"ADMISSION: -> {'PASS' if ok else 'FAIL'}")
    results[fid] = m

print("\n===== SUMMARY =====")
for fid, m in results.items():
    if m is None:
        print(f"{fid:24s} INSUFFICIENT")
    else:
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"{fid:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.3f} rho={m['max_abs_library_correlation']:.3f} {'PASS' if ok else 'FAIL'}")
