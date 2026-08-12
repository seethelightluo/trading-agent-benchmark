"""miner_3 2030-06-13 batch-30 screen: novel cross-asset factors.

Admission gates (warm-up 2020-01-01..2026-07-15, shared with library):
  |IC10| >= 0.007, |ICIR10| >= 0.084   (15-instrument cross-asset universe).

Candidates (all interpretable, one idea each):
  us10y_beta_60         : 60d beta of asset returns vs US10Y yield changes
  us10y_cond_beta_60x20 : 60d beta vs US10Y * sign(US10Y 20d change)
  trend_ts_120          : t-stat of 120d linear trend of log close
  max_dd_60             : 60d rolling max drawdown level (negative)
  xs_rel_mom_20         : 20d return minus cross-sectional EW 20d return
  ret_autocorr5_60      : lag-5 autocorrelation of daily returns over 60d
  night_ret_share_20    : 20d overnight (c2o) return share of total return
  kurt_60               : 60d excess kurtosis of daily returns
  xs_vol_rank_60        : cross-sectional rank of 60d realized vol (low-vol)
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


us10y_r = us10y['close'].pct_change() if us10y is not None else None

candidates = {}

# 1. US10Y beta (yield-change beta) -- library has CN10Y beta, not US10Y
def f_us10y_beta_60(df, s):
    if us10y_r is None:
        return None
    return rolling_beta_series(df, us10y_r, 60)
candidates['us10y_beta_60'] = f_us10y_beta_60

# 2. Conditional US10Y beta * sign(US10Y 20d change)
def f_us10y_cond_beta_60x20(df, s):
    if us10y_r is None:
        return None
    b = rolling_beta_series(df, us10y_r, 60)
    d20 = us10y['close'] / us10y['close'].shift(20) - 1.0
    return (b * np.sign(d20.reindex(df.index).ffill())).reindex(df.index)
candidates['us10y_cond_beta_60x20'] = f_us10y_cond_beta_60x20

# 3. Trend t-stat over 120d of log close
def f_trend_ts_120(df, s):
    lc = np.log(df['close'])
    x = np.arange(len(lc))
    def tstat(y):
        m = np.isfinite(y)
        if m.sum() < 40:
            return np.nan
        yy = y[m]; xx = x[m]
        if np.std(yy) == 0:
            return np.nan
        A = np.vstack([np.ones_like(xx), xx]).T
        coef, res, _, _ = np.linalg.lstsq(A, yy, rcond=None)
        resid = yy - A @ coef
        se = np.sqrt((resid ** 2).sum() / (len(xx) - 2)) / np.sqrt(((xx - xx.mean()) ** 2).sum())
        return coef[1] / se if se > 0 else np.nan
    vals = lc.rolling(120, min_periods=40).apply(tstat, raw=True)
    return vals.reindex(df.index)
candidates['trend_ts_120'] = f_trend_ts_120

# 4. Max drawdown level over 60d (negative)
def f_max_dd_60(df, s):
    c = df['close']
    roll_max = c.rolling(60, min_periods=20).max()
    dd = c / roll_max - 1.0
    return dd.rolling(60, min_periods=20).min().reindex(df.index)
candidates['max_dd_60'] = f_max_dd_60

# 5. Cross-sectional relative momentum: 20d return minus EW 20d return
xs_mom_20 = r_all.rolling(20).apply(lambda x: np.prod(1 + x) - 1, raw=True)
xs_ew_20 = xs_mom_20.mean(axis=1)
xs_ew_20[cnt.reindex(xs_ew_20.index) < 8] = np.nan

def f_xs_rel_mom_20(df, s):
    m20 = df['close'] / df['close'].shift(20) - 1.0
    base = xs_ew_20.reindex(df.index).ffill()
    return (m20 - base).reindex(df.index)
candidates['xs_rel_mom_20'] = f_xs_rel_mom_20

# 6. Lag-5 autocorrelation of daily returns over 60d
def f_ret_autocorr5_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), r.shift(5).rename('rl')], axis=1).dropna()
    c = z['r'].rolling(60).corr(z['rl'])
    return c.reindex(df.index)
candidates['ret_autocorr5_60'] = f_ret_autocorr5_60

# 7. Overnight (close-to-open) return share of total 20d return
def f_night_ret_share_20(df, s):
    o = df['open']; c = df['close']
    night = o / c.shift(1) - 1.0
    total = c / c.shift(20) - 1.0
    night_sum = night.rolling(20).sum()
    tot_sum = total.rolling(20).sum()
    return (night_sum / tot_sum.replace(0, np.nan)).reindex(df.index)
candidates['night_ret_share_20'] = f_night_ret_share_20

# 8. Excess kurtosis of 60d daily returns
def f_kurt_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=30).kurt().reindex(df.index)
candidates['kurt_60'] = f_kurt_60

# 9. Cross-sectional rank of 60d realized vol (low-vol factor)
def f_xs_vol_rank_60(df, s):
    v60 = rolling_std_series(df, 60)
    allv = pd.DataFrame({ss: rolling_std_series(prices[ss], 60) for ss in WATCHLIST}).reindex(df.index)
    nv = allv.notna().sum(axis=1)
    rank = allv.rank(axis=1, pct=True)
    rank[nv < 8] = np.nan
    return rank[s].reindex(df.index)
candidates['xs_vol_rank_60'] = f_xs_vol_rank_60

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

with open('scripts/miner_3_20300613_results_batch30.json', 'w') as f:
    json.dump(results, f, indent=1, default=float)
print(f"DONE {time.time()-t0:.1f}s -> scripts/miner_3_20300613_results_batch30.json")
