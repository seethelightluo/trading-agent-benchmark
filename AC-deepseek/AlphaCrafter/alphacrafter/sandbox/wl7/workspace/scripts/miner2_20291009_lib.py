"""Shared research harness for miner_2 (2029-10-09).
Loads 15-instrument universe + macro signals, computes factor values,
forward returns, and IC/ICIR/coverage/turnover/decay metrics.
Usage: import via runpy or exec; main entry `run_validation(factor_fn, params)`.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'VIX', 'EURUSD', 'USDJPY', 'USDCNY']


def load_prices(days=3000):
    prices = {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is None:
            df = get_index_daily_data(symbol=s, days=days)
        if df is None:
            continue
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        prices[s] = df[['open', 'high', 'low', 'close', 'volume']]
    px = pd.Panel if False else None
    # build wide close frame
    closes = pd.DataFrame({s: prices[s]['close'] for s in prices})
    return prices, closes


def load_macro(days=3000):
    """Macro observation-only signals (may extend beyond current sim date; clip to sim)."""
    out = {}
    for m in MACRO:
        p = f'../persistent/index_data/{m}.csv'
        df = pd.read_csv(p)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        out[m] = df['close']
    mac = pd.DataFrame(out)
    return mac


def forward_ret(closes, horizon=10):
    """Forward return over horizon trading days per asset (aligned on close dates)."""
    fwd = closes.shift(-horizon) / closes - 1.0
    return fwd


def cross_sectional_ic(factor_df, fwd_df, min_assets=8):
    """Daily cross-sectional Spearman IC between factor and forward return."""
    dates, ics = [], []
    for dt in factor_df.index:
        if dt not in fwd_df.index:
            continue
        f = factor_df.loc[dt]
        r = fwd_df.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_assets:
            continue
        fi = f[mask].astype(float)
        ri = r[mask].astype(float)
        if fi.nunique() < 2 or ri.nunique() < 2:
            continue
        ic = fi.corr(ri, method='spearman')
        if np.isfinite(ic):
            dates.append(dt)
            ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def ic_stats(ic_series, direction=1):
    ics = ic_series * direction
    if len(ics) == 0:
        return dict(n=0, ic=np.nan, icir=np.nan, hit=np.nan)
    ic = float(ics.mean())
    icir = float(ics.mean() / ics.std(ddof=1)) if ics.std(ddof=1) > 0 else np.nan
    hit = float((ics > 0).mean())
    return dict(n=len(ics), ic=ic, icir=icir, hit=hit)


def coverage_stats(factor_df):
    valid = factor_df.notna().sum(axis=1)
    dates_ge8 = float((valid >= 8).mean())
    asset_days = float(valid.mean() / factor_df.shape[1])
    return dict(coverage_dates_ge8=dates_ge8, coverage_asset_days=asset_days,
                n_dates=int(factor_df.shape[0]))


def turnover_rank(factor_df):
    """Mean abs change of cross-sectional ranks between consecutive valid dates."""
    r = factor_df.rank(axis=1, pct=True)
    d = r.diff().abs().mean(axis=1)
    return float(d.mean())


def decay_profile(factor_df, closes, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fwd = forward_ret(closes, h)
        ic = cross_sectional_ic(factor_df, fwd)
        out[str(h)] = round(float(ic.mean()), 4) if len(ic) else np.nan
    return out


def library_corr(factor_df, factors_dir='factors', active_ids=None, prefix=''):
    """Max abs pairwise correlation of factor signals with library artifacts.
    Uses .signal.npy files if present, else recomputes via factor files (best effort).
    Returns max_abs over compared library factors."""
    import json, os, glob
    if active_ids is None:
        active_ids = ['rel_mom_20d_skip5', 'downside_vol_ratio_20', 'beta_ew_60d',
                      'max_ret_20d', 'corr_ew_60', 'kurt_20d_skip5', 'dxy_beta_cond_60x20']
    # align factor_df to a common date grid for corr
    sig = factor_df.copy()
    res = {}
    for fid in active_ids:
        npy = os.path.join(factors_dir, f'{fid}.signal.npy')
        if os.path.exists(npy):
            try:
                arr = np.load(npy, allow_pickle=True)
                # try to align: assume same instrument order; use last available shape
                lib = pd.DataFrame(arr, index=sig.index[:arr.shape[0]], columns=sig.columns[:arr.shape[1]])
                common = sig.index.intersection(lib.index)
                if len(common) > 30:
                    a = sig.loc[common].rank(axis=1).values.astype(float)
                    b = lib.loc[common].rank(axis=1).values.astype(float)
                    mask = np.isfinite(a) & np.isfinite(b)
                    if mask.sum() > 100:
                        aa = a[mask]; bb = b[mask]
                        r = float(np.corrcoef(aa, bb)[0, 1]) if aa.std() > 0 and bb.std() > 0 else np.nan
                        res[fid] = r
            except Exception as e:
                pass
    if not res:
        return None, {}
    maxabs = max(abs(v) for v in res.values() if np.isfinite(v)) if any(np.isfinite(v) for v in res.values()) else None
    return maxabs, res


GATES = dict(ic=0.0070, icir=0.0840)


def run_validation(factor_id, factor_name, factor_df, closes, direction=1,
                   horizon=10, params=None, description='', tags=None,
                   regime_notes='', factors_dir='factors'):
    fwd = forward_ret(closes, horizon)
    ic_s = cross_sectional_ic(factor_df, fwd)
    st = ic_stats(ic_s, direction)
    cov = coverage_stats(factor_df)
    turn = turnover_rank(factor_df)
    decay = decay_profile(factor_df, closes)
    maxabs, pair = library_corr(factor_df, factors_dir=factors_dir)
    # yearly breakdown (direction-adjusted)
    yearly = {}
    for yr in sorted(set(ic_s.index.year)):
        sub = ic_s[ic_s.index.year == yr]
        if len(sub):
            yearly[str(yr)] = dict(ic=round(float((sub * direction).mean()), 4),
                                   icir=round(float((sub * direction).mean() / (sub * direction).std(ddof=1)), 4) if (sub * direction).std(ddof=1) > 0 else np.nan,
                                   n=int(len(sub)))
    passed = (abs(st['ic']) >= GATES['ic']) and (abs(st['icir']) >= GATES['icir'])
    res = dict(factor_id=factor_id, factor_name=factor_name, direction=direction,
               horizon=horizon, metrics=dict(ic=st['ic'], icir=st['icir'], hit=st['hit'],
                                             n_ic_dates=st['n'], **cov, turnover_10d_rank=turn,
                                             decay_ic_by_horizon=decay,
                                             max_abs_library_correlation=maxabs),
               yearly=yearly, passed=passed, gates=GATES,
               last_date=str(ic_s.index[-1])[:10] if len(ic_s) else None,
               first_date=str(ic_s.index[0])[:10] if len(ic_s) else None,
               params=params, description=description, tags=tags or [],
               regime_notes=regime_notes)
    print(f'=== {factor_id} (dir {direction}) ===')
    print(f'  IC={st["ic"]:.4f} ICIR={st["icir"]:.4f} hit={st["hit"]:.3f} n={st["n"]} '
          f'[{res["first_date"]}..{res["last_date"]}]')
    print(f'  coverage_dates_ge8={cov["coverage_dates_ge8"]:.3f} asset_days={cov["coverage_asset_days"]:.3f} '
          f'turnover={turn:.3f} max_lib_corr={maxabs}')
    print(f'  decay={decay}')
    print(f'  PASS={passed} (gates ic>={GATES["ic"]} icir>={GATES["icir"]})')
    return res
