"""miner_2 round-12 novel factor screen.

Tests 8 novel candidates against the current 13-factor library:
  - overnight_ret_20    : mean overnight return (open/prev_close - 1) over 20d
  - intraday_ret_20     : mean intraday return (close/open - 1) over 20d
  - gap_freq_dir_20     : (n_gap_up - n_gap_down)/20 over 20d
  - down_vol_ratio_20   : log(mean vol on down days / mean vol on up days) over 20d
  - parkinson_ratio_20  : Parkinson (high-low) vol / close-to-close vol over 20d
  - cvar_60             : 5% expected shortfall over 60d (mean of worst 5% daily rets)
  - downside_vol_ratio_60 : std(down-day rets) / std(up-day rets) over 60d
  - amihud_z_60         : rolling z-score of |ret|/volume over 60d (per-asset normalized)

Admission gate: |IC10| >= 0.007 and |ICIR10| >= 0.084, max |library rho| < 0.5.
"""
import sys, time
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, WATCHLIST)

t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---- load full library signal artifacts (13 effective factors) ----
lib = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    fid = p.name.replace('_signal.npy', '')
    lib[fid] = np.load(p, allow_pickle=False)
print(f'library factors: {len(lib)}', flush=True)


def lib_max_corr(panel):
    arr = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, larr in lib.items():
        corrs = []
        for i in range(arr.shape[0]):
            x, y = arr[i], larr[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xr = pd.Series(x[m]).rank().values
                yr = pd.Series(y[m]).rank().values
                c = np.corrcoef(xr, yr)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


# ---------------- candidates ----------------
def f_overnight_ret_20(df, s):
    ov = df['open'] / df['close'].shift(1) - 1.0
    return ov.rolling(20).mean()


def f_intraday_ret_20(df, s):
    intr = df['close'] / df['open'] - 1.0
    return intr.rolling(20).mean()


def f_gap_freq_dir_20(df, s):
    gap = np.sign(df['open'] / df['close'].shift(1) - 1.0)
    return gap.rolling(20).mean()


def f_down_vol_ratio_20(df, s):
    ret = df['close'].pct_change()
    vol = df['volume'].astype(float)
    down = ret < 0
    up = ret > 0
    md = vol[down].rolling(20).mean()
    mu = vol[up].rolling(20).mean()
    return np.log(md / mu)


def f_parkinson_ratio_20(df, s):
    hl = df['high'] / df['low']
    park = np.log(hl).rolling(20).std() * np.sqrt(1.0 / (4.0 * np.log(2.0)))
    c2c = df['close'].pct_change().rolling(20).std()
    return park / c2c


def f_cvar_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).apply(lambda a: np.mean(np.sort(a)[: max(1, int(0.05 * len(a)))]), raw=True)


def f_downside_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    down = r.where(r < 0)
    up = r.where(r > 0)
    ds = down.rolling(60).std()
    us = up.rolling(60).std()
    return ds / us


def f_amihud_z_60(df, s):
    ret = df['close'].pct_change().abs()
    vol = df['volume'].astype(float)
    ami = ret / (vol + 1e-12)
    mu = ami.rolling(60).mean()
    sd = ami.rolling(60).std()
    return (ami - mu) / sd


candidates = {
    'overnight_ret_20': f_overnight_ret_20,
    'intraday_ret_20': f_intraday_ret_20,
    'gap_freq_dir_20': f_gap_freq_dir_20,
    'down_vol_ratio_20': f_down_vol_ratio_20,
    'parkinson_ratio_20': f_parkinson_ratio_20,
    'cvar_60': f_cvar_60,
    'downside_vol_ratio_60': f_downside_vol_ratio_60,
    'amihud_z_60': f_amihud_z_60,
}

results = {}
for fid, fn in candidates.items():
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
        if panel.shape[0] < 100:
            print(f'{fid}: insufficient panel {panel.shape} -> skip', flush=True)
            continue
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f'{fid}: insufficient data -> None', flush=True)
            continue
        rho, rid = lib_max_corr(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rid
        ok_ic = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        ok_corr = rho < 0.5
        print(f"{fid}: ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rid}) "
              f"d1={m['decay_ic_by_horizon']['1']:+.4f} d5={m['decay_ic_by_horizon']['5']:+.4f} "
              f"d10={m['decay_ic_by_horizon']['10']:+.4f} d20={m['decay_ic_by_horizon']['20']:+.4f} "
              f"-> {'PASS' if (ok_ic and ok_corr) else 'skip'} [{time.time()-t1:.1f}s]", flush=True)
        results[fid] = (m, panel)
    except Exception as e:
        print(f'{fid}: ERROR {type(e).__name__}: {e}', flush=True)

print(f'\nTOTAL {time.time()-t0:.1f}s')
print('SUMMARY:')
for fid, (m, _) in sorted(results.items()):
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and m['max_abs_library_correlation'] < 0.5
    print(f"  {fid:24s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} rho={m['max_abs_library_correlation']:.3f} -> {'ADMIT' if ok else 'skip'}")
