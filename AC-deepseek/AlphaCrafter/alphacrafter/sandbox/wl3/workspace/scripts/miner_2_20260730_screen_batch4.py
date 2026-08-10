"""miner_2 2026-07-30 batch-4 screen: beta-asymmetry, tech beta, coskew, volume-flow,
semi-deviation, illiquidity, trend efficiency, autocorr, intraday/close vol ratio."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, load_index, WATCHLIST, canonical_grid,
                           factor_to_panel, build_library_panels)
import miner_2_20260730_screen_fast as scr

prices = load_prices(days=2000)
grid = canonical_grid(prices)
spx_close = prices['SPX']['close']
ndx_close = prices['NDX']['close']

# ---- library for correlation audit: recompute canonicals + all persisted *.npy artifacts ----
lib_mat = {}
for fid, lp in build_library_panels(prices).items():
    lib_mat[fid] = lp.reindex(grid)[WATCHLIST].values.astype(float)
for f in sorted(Path('factors').glob('*_signal.npy')):
    fid = f.name.replace('_signal.npy', '')
    try:
        arr = np.load(f)
        if arr.shape == (len(grid), 15):
            lib_mat[fid] = arr.astype(float)
    except Exception:
        pass
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


def f_beta_asym_60_spx(df, s):
    r = df['close'].pct_change()
    rm = spx_close.pct_change()
    z = pd.concat([r.rename('a'), rm.rename('b')], axis=1).dropna()
    up = z['b'] > 0
    dn = z['b'] < 0
    bu = (z.loc[up, 'a'].rolling(60, min_periods=15).cov(z.loc[up, 'b'])
          / z.loc[up, 'b'].rolling(60, min_periods=15).var().replace(0, np.nan))
    bd = (z.loc[dn, 'a'].rolling(60, min_periods=15).cov(z.loc[dn, 'b'])
          / z.loc[dn, 'b'].rolling(60, min_periods=15).var().replace(0, np.nan))
    return (bu - bd).reindex(z.index)


def f_ndx_beta_60(df, s):
    r = df['close'].pct_change()
    rm = ndx_close.pct_change()
    z = pd.concat([r.rename('a'), rm.rename('b')], axis=1).dropna()
    return (z['a'].rolling(60).cov(z['b']) / z['b'].rolling(60).var().replace(0, np.nan)).reindex(z.index)


def f_coskew_60_spx(df, s):
    r = df['close'].pct_change()
    rm = spx_close.pct_change()
    z = pd.concat([r.rename('a'), rm.rename('b')], axis=1).dropna()
    num = (z['a'] * z['b'] ** 2).rolling(60).mean()
    sd_a = z['a'].rolling(60).std()
    sd_b = z['b'].rolling(60).std()
    return (num / (sd_a * sd_b ** 2).replace(0, np.nan)).reindex(z.index)


def f_volume_trend_20_60(df, s):
    v = df['volume'].replace(0, np.nan)
    return v.rolling(20).mean() / v.rolling(60).mean() - 1.0


def f_volume_z_20(df, s):
    v = df['volume'].replace(0, np.nan)
    return (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)


def f_downside_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    neg = r.clip(upper=0)
    dv = (neg ** 2).rolling(60).mean() ** 0.5
    tv = r.rolling(60).std()
    return (dv / tv.replace(0, np.nan)).reindex(r.index)


def f_amihud_illiq_20(df, s):
    r = df['close'].pct_change().abs()
    v = df['volume'].replace(0, np.nan)
    return (r / v).rolling(20).mean() * 1e9


def f_efficiency_ratio_20(df, s):
    close = df['close']
    net = (close - close.shift(20)).abs()
    path = close.diff().abs().rolling(20).sum()
    return (net / path.replace(0, np.nan)).reindex(close.index)


def f_ret_autocorr_5(df, s):
    r = df['close'].pct_change()
    return r.rolling(5).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) >= 6 else np.nan, raw=True)


def f_hl_vol_ratio_60(df, s):
    hl = ((df['high'] - df['low']) / df['close'].replace(0, np.nan)).rolling(60).mean()
    cv = df['close'].pct_change().rolling(60).std()
    return (hl / cv.replace(0, np.nan)).reindex(hl.index)


CANDIDATES = [
    ("beta_asym_60_spx", f_beta_asym_60_spx),
    ("ndx_beta_60", f_ndx_beta_60),
    ("coskew_60_spx", f_coskew_60_spx),
    ("volume_trend_20_60", f_volume_trend_20_60),
    ("volume_z_20", f_volume_z_20),
    ("downside_vol_ratio_60", f_downside_vol_ratio_60),
    ("amihud_illiq_20", f_amihud_illiq_20),
    ("efficiency_ratio_20", f_efficiency_ratio_20),
    ("ret_autocorr_5", f_ret_autocorr_5),
    ("hl_vol_ratio_60", f_hl_vol_ratio_60),
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
          f"turn={m['turnover_10d_rank']:.2f} rho={rho:.2f}({rho_id}) -> {'PASS' if ok and rho < 0.5 else 'FAIL'}")
    d = m['decay_ic_by_horizon']
    print(f"{'':22s} decay " + " ".join(f"h{h}:{d[str(h)]:+.4f}" for h in [1, 3, 5, 10, 20]))
