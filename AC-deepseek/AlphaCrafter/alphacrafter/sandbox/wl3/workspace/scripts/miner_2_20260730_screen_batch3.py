"""miner_2 2026-07-30 batch-3 screen (fixed: load .npy signal artifacts for rho audit)."""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, load_index, WATCHLIST, canonical_grid,
                           factor_to_panel, forward_returns, build_library_panels)
import miner_2_20260730_screen_fast as scr

prices = load_prices(days=2000)
grid = canonical_grid(prices)
xau_close = prices['XAU']['close'] if 'XAU' in prices else None
spx_close = prices['SPX']['close'] if 'SPX' in prices else None

# ---- library for correlation audit: recompute 4 canonical + all persisted *.npy artifacts ----
lib_mat = {}
for fid, lp in build_library_panels(prices).items():
    lib_mat[fid] = lp.reindex(grid)[WATCHLIST].values.astype(float)
for f in sorted(Path('factors').glob('*_signal.npy')):
    fid = f.name.replace('_signal.npy', '')
    try:
        arr = np.load(f)
        if arr.shape == (len(grid), 15):
            lib_mat[fid] = arr.astype(float)
    except Exception as exc:
        print(f"skip artifact {f.name}: {exc}")
print(f"library for rho audit ({len(lib_mat)}): {sorted(lib_mat)}")


def max_rho(fac):
    best, best_id = 0.0, None
    for fid_l, lm in lib_mat.items():
        c = np.array(scr.spearman_rows(fac, lm))
        c = c[np.isfinite(c)]
        if len(c):
            r = float(np.mean(c))
            if abs(r) > best:
                best, best_id = abs(r), fid_l
    return best, best_id


def f_close_loc_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return ((df['close'] - df['open']) / rng).rolling(20).mean()


def f_lower_shadow_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return ((df['close'] - df['low']) / rng).rolling(20).mean()


def f_upper_shadow_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return ((df['high'] - df['close']) / rng).rolling(20).mean()


def f_gap_20(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    return gap.rolling(20).mean()


def f_vol_price_corr_60(df, s):
    r = df['close'].pct_change()
    v = df['volume']
    z = pd.concat([r.rename('r'), v.rename('v')], axis=1).replace([np.inf, -np.inf], np.nan)
    return z['r'].rolling(60).corr(z['v'])


def f_gold_beta_60(df, s):
    if xau_close is None:
        return None
    r = df['close'].pct_change()
    rr = xau_close.pct_change()
    z = pd.concat([r.rename('a'), rr.rename('b')], axis=1).dropna()
    return (z['a'].rolling(60).cov(z['b']) / z['b'].rolling(60).var().replace(0, np.nan)).reindex(z.index)


def f_mom_accel_20_60(df, s):
    close = df['close']
    m20 = close.shift(5) / close.shift(25) - 1.0
    m60 = close.shift(5) / close.shift(65) - 1.0
    return m20 - m60


def f_mdd_63(df, s):
    roll_max = df['close'].rolling(63, min_periods=30).max()
    return df['close'] / roll_max - 1.0


def f_downside_beta_60(df, s):
    if spx_close is None:
        return None
    r = df['close'].pct_change()
    rr = spx_close.pct_change()
    z = pd.concat([r.rename('a'), rr.rename('b')], axis=1).dropna()
    neg = z['b'] < 0
    a = z.loc[neg, 'a']
    b = z.loc[neg, 'b']
    cov = a.rolling(60, min_periods=15).cov(b)
    var = b.rolling(60, min_periods=15).var()
    return (cov / var.replace(0, np.nan)).reindex(z.index)


def f_rv_gap_5_20(df, s):
    r = df['close'].pct_change()
    return r.rolling(5).std() / r.rolling(20).std().replace(0, np.nan) - 1.0


def f_vwap_dist_20(df, s):
    close = df['close']
    v = df['volume'].replace(0, np.nan)
    vwap = (close * v).rolling(20).sum() / v.rolling(20).sum()
    return close / vwap - 1.0


CANDIDATES = [
    ("close_loc_20", f_close_loc_20),
    ("lower_shadow_20", f_lower_shadow_20),
    ("upper_shadow_20", f_upper_shadow_20),
    ("gap_20", f_gap_20),
    ("vol_price_corr_60", f_vol_price_corr_60),
    ("gold_beta_60", f_gold_beta_60),
    ("mom_accel_20_60", f_mom_accel_20_60),
    ("mdd_63", f_mdd_63),
    ("downside_beta_60", f_downside_beta_60),
    ("rv_gap_5_20", f_rv_gap_5_20),
    ("vwap_dist_20", f_vwap_dist_20),
]

for fid, fn in CANDIDATES:
    try:
        panel = factor_to_panel(fn, prices)
        m = scr.evaluate_fast(fid, panel)
    except Exception as exc:
        print(f"{fid:22s} ERROR {exc}")
        continue
    if m is None:
        print(f"{fid:22s} INSUFFICIENT")
        continue
    fac = panel.reindex(grid)[WATCHLIST].values.astype(float)
    rho, rho_id = max_rho(fac)
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"{fid:22s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:5d} cov={m['coverage_asset_days']:.2f} ge8={m['coverage_dates_ge8']:.2f} "
          f"turn={m['turnover_10d_rank']:.2f} rho={rho:.2f}({rho_id}) -> {'PASS' if ok else 'FAIL'}")
    d = m['decay_ic_by_horizon']
    print(f"{'':22s} decay " + " ".join(f"h{h}:{d[str(h)]:+.4f}" for h in [1, 3, 5, 10, 20]))
