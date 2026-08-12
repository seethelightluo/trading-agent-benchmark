"""miner_3 2030-07-25 batch-31 screen: novel cross-asset factors + revalidation.

Admission gates (warm-up 2020-01-01..2026-07-15, shared with library):
  |IC10| >= 0.007, |ICIR10| >= 0.084   (15-instrument cross-asset universe).

Candidates (all interpretable, one idea each):
  xs_zscore_mom_5     : cross-sectional z-score of 5d momentum (relative strength)
  up_day_ratio_60     : fraction of positive daily returns over 60d (trend consistency)
  sortino_60          : 60d mean daily return / downside deviation
  uscn_spread_beta_60 : 60d beta of asset returns vs (US10Y-CN10Y) spread change
  gap_freq_60         : frequency of |overnight gap| > 1% over 60d (gap risk)
  range_amplitude_60  : (max-min)/mean of close over 60d (trading amplitude)
  vol_volume_corr_20  : corr(|daily ret|, volume) over 20d (volume confirmation)
  hilo_pos_chg_20     : 20d change in (close-low)/(high-low) intraday position
  reversal_5d         : negative of 5d return (short-term mean reversion)
  max_dd_60           : REVALIDATE batch-30 PASS (60d max drawdown level, negative)
  night_ret_share_20  : REVALIDATE batch-30 PASS (overnight share of 20d return)
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, WATCHLIST, VAL_START,
                           VAL_END, factor_to_panel, forward_returns)

t0 = time.time()
prices = load_prices(days=2700)
print(f"assets loaded: {len(prices)}; max date: {max(d.index.max() for d in prices.values())} ({time.time()-t0:.1f}s)")

idx = set()
for s, df in prices.items():
    idx.update(df.index)
gidx = pd.DatetimeIndex(sorted(idx))
gidx = gidx[gidx >= VAL_START]
print(f"trading grid: {len(gidx)} dates, {gidx.min().date()}..{gidx.max().date()}")

cal_grid = pd.date_range(VAL_START, VAL_END, freq='D')
print(f"library calendar grid: {len(cal_grid)} dates, {cal_grid.min().date()}..{cal_grid.max().date()}")

r_all = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()}).sort_index()
cnt = r_all.notna().sum(axis=1)
r_ew = r_all.mean(axis=1)
r_ew[cnt < 8] = np.nan

us10y = prices.get('US10Y')
cn10y = prices.get('CN10Y')


def rolling_beta_series(df, mkt, window):
    r = df['close'].pct_change()
    mm = mkt.reindex(r.index).ffill()
    z = pd.concat([r.rename('r'), mm.rename('m')], axis=1).dropna()
    cov = z['r'].rolling(window).cov(z['m'])
    var = z['m'].rolling(window).var()
    return (cov / var).reindex(df.index)


def rolling_std_series(df, window):
    return df['close'].pct_change().rolling(window).std().reindex(df.index)


# ---- 1. cross-sectional z-score of 5d momentum ----
xs_m5 = pd.DataFrame({s: d['close'] / d['close'].shift(5) - 1.0 for s, d in prices.items()}).sort_index()
xs_m5_mean = xs_m5.mean(axis=1)
xs_m5_std = xs_m5.std(axis=1)
xs_m5_mean[cnt.reindex(xs_m5_mean.index) < 8] = np.nan
xs_m5_std[cnt.reindex(xs_m5_std.index) < 8] = np.nan


def f_xs_zscore_mom_5(df, s):
    m5 = df['close'] / df['close'].shift(5) - 1.0
    mu = xs_m5_mean.reindex(df.index).ffill()
    sd = xs_m5_std.reindex(df.index).ffill()
    return ((m5 - mu) / sd.replace(0, np.nan)).reindex(df.index)


# ---- 2. up-day ratio over 60d ----
def f_up_day_ratio_60(df, s):
    r = df['close'].pct_change()
    return (r > 0).rolling(60, min_periods=30).mean().reindex(df.index)


# ---- 3. sortino over 60d ----
def f_sortino_60(df, s):
    r = df['close'].pct_change()
    mu = r.rolling(60, min_periods=30).mean()
    dd = r.clip(upper=0).rolling(60, min_periods=30).apply(
        lambda x: np.sqrt(np.mean(np.square(x))) if np.isfinite(x).sum() >= 30 else np.nan, raw=True)
    return (mu / dd.replace(0, np.nan)).reindex(df.index)


# ---- 4. beta vs US10Y-CN10Y spread change ----
if us10y is not None and cn10y is not None:
    spread = us10y['close'] - cn10y['close'].reindex(us10y.index).ffill()
    d_spread = spread.diff()


def f_uscn_spread_beta_60(df, s):
    if us10y is None or cn10y is None:
        return None
    return rolling_beta_series(df, d_spread, 60)


# ---- 5. gap frequency over 60d ----
def f_gap_freq_60(df, s):
    o = df['open']; c = df['close']
    gap = o / c.shift(1) - 1.0
    return (gap.abs() > 0.01).rolling(60, min_periods=30).mean().reindex(df.index)


# ---- 6. range amplitude over 60d ----
def f_range_amplitude_60(df, s):
    c = df['close']
    rmax = c.rolling(60, min_periods=30).max()
    rmin = c.rolling(60, min_periods=30).min()
    rmean = c.rolling(60, min_periods=30).mean()
    return ((rmax - rmin) / rmean.replace(0, np.nan)).reindex(df.index)


# ---- 7. |ret|-volume correlation over 20d ----
def f_vol_volume_corr_20(df, s):
    r = df['close'].pct_change().abs()
    v = df['volume']
    if v is None or v.notna().sum() < 40:
        return None
    z = pd.concat([r.rename('r'), v.rename('v')], axis=1).dropna()
    return z['r'].rolling(20, min_periods=10).corr(z['v']).reindex(df.index)


# ---- 8. 20d change in intraday position ----
def f_hilo_pos_chg_20(df, s):
    hi = df['high']; lo = df['low']; c = df['close']
    rng = (hi - lo).replace(0, np.nan)
    hp = ((c - lo) / rng).reindex(df.index)
    return (hp - hp.shift(20)).reindex(df.index)


# ---- 9. 5d reversal ----
def f_reversal_5d(df, s):
    return -(df['close'] / df['close'].shift(5) - 1.0).reindex(df.index)


# ---- 10. max drawdown level over 60d (revalidate batch-30) ----
def f_max_dd_60(df, s):
    c = df['close']
    roll_max = c.rolling(60, min_periods=20).max()
    dd = c / roll_max - 1.0
    return dd.rolling(60, min_periods=20).min().reindex(df.index)


# ---- 11. overnight return share of 20d total (revalidate batch-30) ----
def f_night_ret_share_20(df, s):
    o = df['open']; c = df['close']
    night = o / c.shift(1) - 1.0
    total = c / c.shift(20) - 1.0
    night_sum = night.rolling(20).sum()
    tot_sum = total.rolling(20).sum()
    return (night_sum / tot_sum.replace(0, np.nan)).reindex(df.index)


candidates = {
    'xs_zscore_mom_5': f_xs_zscore_mom_5,
    'up_day_ratio_60': f_up_day_ratio_60,
    'sortino_60': f_sortino_60,
    'uscn_spread_beta_60': f_uscn_spread_beta_60,
    'gap_freq_60': f_gap_freq_60,
    'range_amplitude_60': f_range_amplitude_60,
    'vol_volume_corr_20': f_vol_volume_corr_20,
    'hilo_pos_chg_20': f_hilo_pos_chg_20,
    'reversal_5d': f_reversal_5d,
    'max_dd_60': f_max_dd_60,
    'night_ret_share_20': f_night_ret_share_20,
}

# ---------- vectorized IC engine ----------
def row_spearman(X, Y, min_valid=8):
    X = pd.DataFrame(X, dtype=float)
    Y = pd.DataFrame(Y, dtype=float)
    m = X.notna() & Y.notna()
    n = m.sum(axis=1)
    X2 = X.where(m); Y2 = Y.where(m)
    rx = X2.rank(axis=1)
    ry = Y2.rank(axis=1)
    rxm = rx.sub(rx.mean(axis=1), axis=0)
    rym = ry.sub(ry.mean(axis=1), axis=0)
    num = (rxm * rym).sum(axis=1)
    den = np.sqrt((rxm ** 2).sum(axis=1) * (rym ** 2).sum(axis=1))
    rho = (num / den.replace(0, np.nan)).to_numpy(dtype=float).copy()
    rho[n < min_valid] = np.nan
    return rho


fwd_mats = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd = forward_returns(prices, h).reindex(gidx)
    fwd_mats[h] = fwd[WATCHLIST].values.astype(float)

# ---- library artifacts with per-factor grids from JSON metadata ----
lib_artifacts = {}
for jp in sorted(Path('factors').glob('*.json')):
    try:
        p = json.loads(jp.read_text(encoding='utf-8'))
        art = p.get('signal_artifact')
        if not art:
            continue
        arr = np.load(Path('factors') / art, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] != 15:
            continue
        g = p.get('signal_artifact_grid', {})
        grid = None
        try:
            cand = pd.date_range(pd.Timestamp(g['start']), pd.Timestamp(g['end']), freq='D')
            if len(cand) == g.get('n_dates') and len(cand) == arr.shape[0]:
                grid = cand
        except Exception:
            pass
        if grid is None:
            if arr.shape[0] == len(cal_grid):
                grid = cal_grid
        if grid is not None:
            lib_artifacts[p.get('factor_id', jp.stem)] = (grid, arr)
    except Exception:
        pass
print(f"library artifacts with usable grids: {len(lib_artifacts)} ({time.time()-t0:.1f}s)")


def max_lib_corr(panel):
    best, best_id = 0.0, None
    for fid, (grid, la) in lib_artifacts.items():
        mc = panel.reindex(grid)[WATCHLIST].values.astype(float)
        c = row_spearman(mc, la)
        c = c[np.isfinite(c)]
        if len(c):
            r = float(np.abs(c).mean())
            if r > best:
                best, best_id = r, fid
    return best, best_id


warm = (gidx >= VAL_START) & (gidx <= VAL_END)
rstart = VAL_END + pd.Timedelta(days=1)
recent = gidx >= rstart
recent = recent & (gidx <= gidx.max() - pd.Timedelta(days=15))

results = {}
for fid, fn in candidates.items():
    t1 = time.time()
    panel = factor_to_panel(fn, prices)
    if panel.empty:
        print(f"{fid}: EMPTY panel"); continue
    mat = panel.reindex(gidx)[WATCHLIST].values.astype(float)
    ics = {}
    for h in (1, 2, 3, 5, 10, 20):
        ics[h] = row_spearman(mat, fwd_mats[h])
    ic10w = ics[10][warm]
    ic10w = ic10w[np.isfinite(ic10w)]
    if len(ic10w) < 100:
        print(f"{fid}: insufficient warm IC dates {len(ic10w)}"); continue
    ic = float(ic10w.mean()); sd = float(ic10w.std(ddof=1))
    icir = ic / sd if sd > 0 else 0.0
    hit = float((ic10w > 0).mean()) if ic >= 0 else float((ic10w < 0).mean())
    fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    cov = float(fac.notna().sum().sum()) / (fac.shape[0] * fac.shape[1]) if fac.shape[0] else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
    turn = float(fac.rank(axis=1).diff(10).abs().mean().mean()) if len(fac) > 10 else float('nan')
    decay = {str(h): float(np.nanmean(ics[h][warm])) for h in (1, 2, 3, 5, 10, 20)}
    icr = ics[10][recent]
    icr = icr[np.isfinite(icr)]
    ic_rmean = float(icr.mean()) if len(icr) >= 30 else float('nan')
    ic_rsd = float(icr.std(ddof=1)) if len(icr) >= 30 else float('nan')
    ic_ricir = ic_rmean / ic_rsd if len(icr) >= 30 and ic_rsd > 0 else float('nan')
    rho, fid_rho = max_lib_corr(panel)
    ok = abs(ic) >= 0.007 and abs(icir) >= 0.084
    results[fid] = {
        'ic': ic, 'icir': icir, 'hit': hit, 'cov': cov, 'ge8': ge8, 'turn': turn,
        'decay': decay, 'rho': rho, 'rho_id': fid_rho,
        'ic_recent': ic_rmean, 'icir_recent': ic_ricir,
        'n_recent': int(len(icr)), 'n_warm': int(len(ic10w)), 'PASS': ok,
    }
    print(f"{fid}: ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} cov={cov:.3f} ge8={ge8:.3f} "
          f"turn={turn:.2f} rho={rho:.3f}({fid_rho}) recent_ic={ic_rmean:+.4f} recent_icir={ic_ricir:+.4f} "
          f"n_rec={len(icr)} PASS={ok} ({time.time()-t1:.1f}s)")

with open('scripts/miner_3_20300725_results_batch31.json', 'w') as f:
    json.dump(results, f, indent=1, default=float)
print(f"DONE {time.time()-t0:.1f}s -> scripts/miner_3_20300725_results_batch31.json")
