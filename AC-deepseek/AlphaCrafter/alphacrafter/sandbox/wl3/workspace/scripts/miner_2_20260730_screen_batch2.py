"""miner_2 2026-07-30 batch-2 screen: 14 new factor ideas on the 15-asset universe.

Library correlation is computed against the 5 EFFECTIVE persisted factor
artifacts (factors/*_signal.npy on the canonical grid) plus legacy panels,
so rho estimates match what the deterministic gate will see.

Run: python scripts/miner_2_20260730_screen_batch2.py [start] [end]
"""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, load_index, WATCHLIST, VAL_START, VAL_END,
                           canonical_grid, factor_to_panel, forward_returns)
import miner_2_20260730_screen_fast as scr

prices = load_prices(days=2000)
grid = canonical_grid(prices)
usdjpy = load_index('USDJPY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
spx_close = prices['SPX']['close'] if 'SPX' in prices else None
btc_close = prices['BTC']['close'] if 'BTC' in prices else None
us10y_close = prices['US10Y']['close'] if 'US10Y' in prices else None

# --- library matrix: 5 persisted effective artifacts (canonical grid, WATCHLIST order) ---
lib_mat = {}
for f in sorted(Path('factors').glob('*_signal.npy')):
    fid = f.name.replace('_signal.npy', '')
    arr = np.load(f)
    if arr.shape == (len(grid), 15):
        lib_mat[fid] = arr
print(f"library artifacts for correlation: {list(lib_mat.keys())}")
print(f"grid: {len(grid)} dates ({grid.min().date()}..{grid.max().date()}), assets: {len(WATCHLIST)}")


def rolling_beta(r_asset, r_ref, window):
    z = pd.concat([r_asset.rename('a'), r_ref.rename('b')], axis=1).dropna()
    b = z['a'].rolling(window).cov(z['b']) / z['b'].rolling(window).var().replace(0, np.nan)
    return b.reindex(z.index)


def f_usdjpy_beta_cond_60x20(df, s):
    if usdjpy is None:
        return None
    r = df['close'].pct_change()
    rr = usdjpy['close'].pct_change()
    b = rolling_beta(r, rr, 60)
    move = usdjpy['close'] / usdjpy['close'].shift(20) - 1.0
    return (b * move).reindex(r.index)


def f_eurusd_beta_cond_60x20(df, s):
    if eurusd is None:
        return None
    r = df['close'].pct_change()
    rr = eurusd['close'].pct_change()
    b = rolling_beta(r, rr, 60)
    move = eurusd['close'] / eurusd['close'].shift(20) - 1.0
    return (b * move).reindex(r.index)


def f_spx_beta_60(df, s):
    if spx_close is None:
        return None
    r = df['close'].pct_change()
    rr = spx_close.pct_change()
    return rolling_beta(r, rr, 60)


def f_vol_level_20(df, s):
    vol = df['close'].pct_change().rolling(20).std()
    return -np.log(vol.replace(0, np.nan))


def f_range_vol_20(df, s):
    rng = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
    return -rng.rolling(20).mean()


def f_parkinson_vol_term_20_60(df, s):
    hl = np.log(df['high'] / df['low'].replace(0, np.nan))
    park = (hl ** 2).rolling(20).mean() / (4 * np.log(2))
    p20 = np.sqrt(park)
    park60 = (hl ** 2).rolling(60).mean() / (4 * np.log(2))
    p60 = np.sqrt(park60)
    return (p20 / p60.replace(0, np.nan) - 1.0)


def f_volume_trend_20_60(df, s):
    v = df['volume']
    return v.rolling(20).mean() / v.rolling(60).mean().replace(0, np.nan) - 1.0


def f_obv_slope_20(df, s):
    r = df['close'].pct_change()
    obv = (np.sign(r) * df['volume']).cumsum()
    flow = obv - obv.shift(20)
    scale = (r.abs() * df['volume']).rolling(20).sum()
    return (flow / scale.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_amihud_20(df, s):
    r = df['close'].pct_change().abs()
    illiq = (r / df['volume'].replace(0, np.nan)).rolling(20).mean()
    return np.log(illiq.replace(0, np.nan))


def f_kurtosis_60d(df, s):
    return df['close'].pct_change().rolling(60).kurt()


def f_mom_consistency_60(df, s):
    r = df['close'].pct_change()
    pos = (r > 0).astype(float).rolling(60).mean()
    return pos - 0.5


def f_pullback_in_trend_60x5(df, s):
    close = df['close']
    mom60 = close / close.shift(60) - 1.0
    ret5 = close / close.shift(5) - 1.0
    return np.sign(mom60) * (-ret5)


def f_beta_us10y_60(df, s):
    if us10y_close is None:
        return None
    r = df['close'].pct_change()
    rr = us10y_close.pct_change()
    return rolling_beta(r, rr, 60)


def f_btc_beta_60(df, s):
    if btc_close is None:
        return None
    r = df['close'].pct_change()
    rr = btc_close.pct_change()
    return rolling_beta(r, rr, 60)


CANDIDATES = [
    ("usdjpy_beta_cond_60x20", f_usdjpy_beta_cond_60x20),
    ("eurusd_beta_cond_60x20", f_eurusd_beta_cond_60x20),
    ("spx_beta_60", f_spx_beta_60),
    ("vol_level_20", f_vol_level_20),
    ("range_vol_20", f_range_vol_20),
    ("parkinson_vol_term_20_60", f_parkinson_vol_term_20_60),
    ("volume_trend_20_60", f_volume_trend_20_60),
    ("obv_slope_20", f_obv_slope_20),
    ("amihud_20", f_amihud_20),
    ("kurtosis_60d", f_kurtosis_60d),
    ("mom_consistency_60", f_mom_consistency_60),
    ("pullback_in_trend_60x5", f_pullback_in_trend_60x5),
    ("beta_us10y_60", f_beta_us10y_60),
    ("btc_beta_60", f_btc_beta_60),
]


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(CANDIDATES)
    for fid, fn in CANDIDATES[start:end]:
        try:
            panel = factor_to_panel(fn, prices)
            m = scr.evaluate_fast(fid, panel)
        except Exception as exc:
            print(f"{fid:28s} ERROR {exc}")
            continue
        if m is None:
            print(f"{fid:28s} INSUFFICIENT")
            continue
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"{fid:28s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']:5d} cov={m['coverage_asset_days']:.2f} ge8={m['coverage_dates_ge8']:.2f} "
              f"turn={m['turnover_10d_rank']:.2f} rho_lib={m['max_abs_library_correlation']:.2f} "
              f"vs {m['max_corr_library_id']} -> {'PASS' if ok else 'FAIL'}")
        d = m['decay_ic_by_horizon']
        print(f"{'':28s} decay " + " ".join(f"h{h}:{d[str(h)]:+.4f}" for h in [1, 3, 5, 10, 20]))


if __name__ == '__main__':
    main()
