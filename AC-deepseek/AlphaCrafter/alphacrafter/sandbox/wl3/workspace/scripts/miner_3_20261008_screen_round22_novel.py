"""Round 22 (2026-10-08): novel factor screen for the 15-asset cross-asset universe.

Candidate families (avoiding overlap with the 18-factor library):
  1. roll_kurt_60            : rolling 60d kurtosis of daily returns (fat-tail risk)
  2. jump_intensity_60       : fraction of days with |r| > 2*std20 over 60d (jump frequency)
  3. us10y_corr_60           : rolling 60d corr(asset ret, d(US10Y)) (bond-equity correlation)
  4. btc_down_beta_60x20     : beta on BTC on BTC-down days x BTC 20d ret (crypto tail)
  5. weekday_effect_120      : mean Monday return - mean Friday return (120d)
  6. tom_effect_120          : turn-of-month: mean ret d1-3 minus d26-end (120d)
  7. overnight_intraday_corr_20 : corr(gap_t, intraday_t) over 20d (within-day reversal)
  8. trend_r2_60             : R^2 of 60d linear trend on close (trend quality)
  9. dd_depth_60             : close/rolling_max(close,60)-1 (drawdown depth)
 10. tech_beta_diff_60       : beta(asset,NDX,60) - beta(asset,SPX,60) (tech sensitivity)
 11. wti_beta_cond_60x20     : beta on WTI on WTI-up days x WTI 20d ret (oil sensitivity)
 12. jpy_beta_cond_60x20     : beta on USDJPY on JPY-strengthening days x USDJPY 20d ret
 13. vol_skew_20_60          : skew(r,20) - skew(r,60) (skew term structure)
 14. autocorr_abs_ret_20     : lag-1 autocorrelation of |r| over 20d (vol clustering)

Gate: |IC10| >= 0.007, |ICIR10| >= 0.084, max_abs_library_correlation < 0.5 (gate-style
per-date Spearman rho vs all 18 effective library signal artifacts on the canonical grid).
"""
import sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, INDEX_SIGNALS, load_prices, load_index,
                           canonical_grid, factor_to_panel, validate_factor,
                           signal_matrix, VAL_START, VAL_END, forward_returns,
                           rank_ic_series)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)}", flush=True)
print(f"last price date per asset:", {s: str(df.index.max().date()) for s, df in prices.items()}, flush=True)

# ---------------- library panels from real signal artifacts (gate-style) ----------------
lib_panels = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        fid = d['factor_id']
        art = d.get('signal_artifact')
        if not art:
            continue
        arr = np.load('factors/' + art, allow_pickle=False)
        if arr.shape == (len(grid), len(WATCHLIST)):
            lib_panels[fid] = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
    except Exception as e:
        print(f"  artifact ERR {f}: {e}", flush=True)
print(f"library panels: {len(lib_panels)} -> {sorted(lib_panels.keys())}", flush=True)


def max_lib_corr(panel):
    """Per-date Spearman rho vs each library factor; report max |mean rho| (gate-style)."""
    pm = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    per_id = {}
    for fid, lp in lib_panels.items():
        lm = lp.values
        corrs = []
        for t in range(len(grid)):
            x = pm[t]; y = lm[t]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xr = pd.Series(x[m]).rank().values
                yr = pd.Series(y[m]).rank().values
                xc = xr - xr.mean(); yc = yr - yr.mean()
                den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
                if den > 0:
                    corrs.append((xc * yc).sum() / den)
        if corrs:
            r = float(np.mean(corrs))
            per_id[fid] = r
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id, per_id


# ---------------- macro / market inputs ----------------
spx_ret = prices['SPX']['close'].pct_change()
ndx_ret = prices['NDX']['close'].pct_change() if 'NDX' in prices else None
btc_ret = prices['BTC']['close'].pct_change() if 'BTC' in prices else None
wti_ret = prices['WTI']['close'].pct_change() if 'WTI' in prices else None
us10y = prices['US10Y']['close']
jpy = load_index('USDJPY', prices=prices)


def _cond_beta(r, m, cond, w):
    """Rolling beta of r on m restricted to observations where cond(m) holds;
    window w is in terms of filtered observations. Returns series aligned to r."""
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    z = z[cond(z['m'])]
    if len(z) < w + 5:
        return pd.Series(np.nan, index=r.index)
    b = z['r'].rolling(w).cov(z['m']) / z['m'].rolling(w).var().replace(0, np.nan)
    return b.reindex(r.index)


# ---------------- candidate constructions ----------------
def f_roll_kurt_60(df, s):
    return df['close'].pct_change().rolling(60).kurt()


def f_jump_intensity_60(df, s):
    r = df['close'].pct_change()
    sd = r.rolling(20).std()
    ind = (r.abs() > 2.0 * sd).astype(float)
    return ind.rolling(60).mean()


def f_us10y_corr_60(df, s):
    r = df['close'].pct_change()
    dy = us10y.diff()
    z = pd.concat([r.rename('r'), dy.rename('y')], axis=1).dropna()
    return z['r'].rolling(60).corr(z['y']).reindex(z.index)


def f_btc_down_beta(df, s):
    if btc_ret is None:
        return None
    r = df['close'].pct_change()
    b = _cond_beta(r, btc_ret, lambda m: m < 0, 60)
    return (-b * (btc_ret.rolling(20).mean() * 20)).reindex(r.index)


def f_weekday_effect_120(df, s):
    r = df['close'].pct_change()
    dow = r.index.dayofweek
    mon = r.where(dow == 0).rolling(120, min_periods=20).mean()
    fri = r.where(dow == 4).rolling(120, min_periods=20).mean()
    return (mon - fri)


def f_tom_effect_120(df, s):
    r = df['close'].pct_change()
    dom = r.index.day
    early = r.where(dom <= 3).rolling(120, min_periods=15).mean()
    late = r.where(dom >= 26).rolling(120, min_periods=15).mean()
    return (early - late)


def f_overnight_intraday_corr_20(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    intra = df['close'] / df['open'] - 1.0
    z = pd.concat([gap.rename('g'), intra.rename('i')], axis=1).dropna()
    return z['g'].rolling(20).corr(z['i']).reindex(z.index)


def f_trend_r2_60(df, s):
    c = df['close']
    idx = pd.Series(np.arange(len(c)), index=c.index)
    corr = c.rolling(60).corr(idx)
    return (corr ** 2)


def f_dd_depth_60(df, s):
    return df['close'] / df['close'].rolling(60).max() - 1.0


def f_tech_beta_diff_60(df, s):
    if ndx_ret is None:
        return None
    r = df['close'].pct_change()
    b_ndx = r.rolling(60).cov(ndx_ret) / ndx_ret.rolling(60).var().replace(0, np.nan)
    b_spx = r.rolling(60).cov(spx_ret) / spx_ret.rolling(60).var().replace(0, np.nan)
    return (b_ndx - b_spx)


def f_wti_beta_cond(df, s):
    if wti_ret is None:
        return None
    r = df['close'].pct_change()
    b = _cond_beta(r, wti_ret, lambda m: m > 0, 60)
    return (b * (wti_ret.rolling(20).mean() * 20)).reindex(r.index)


def f_jpy_beta_cond(df, s):
    if jpy is None:
        return None
    r = df['close'].pct_change()
    jr = jpy['close'].pct_change()
    # JPY-strengthening days: USDJPY falls
    b = _cond_beta(r, jr, lambda m: m < 0, 60)
    return (b * (jr.rolling(20).mean() * 20)).reindex(r.index)


def f_vol_skew_20_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).skew() - r.rolling(60).skew()


def f_autocorr_abs_ret_20(df, s):
    ar = df['close'].pct_change().abs()
    return ar.rolling(20).corr(ar.shift(1))


cands = {
    'roll_kurt_60': (f_roll_kurt_60, 'rolling 60d kurtosis of daily returns', 'higher-moment fat-tail risk'),
    'jump_intensity_60': (f_jump_intensity_60, 'share of days with |r|>2*std20 in 60d', 'jump frequency / tail activity'),
    'us10y_corr_60': (f_us10y_corr_60, 'rolling 60d corr(asset ret, dUS10Y)', 'bond-equity correlation regime'),
    'btc_down_beta_60x20': (f_btc_down_beta, '-downside beta on BTC x BTC 20d trend', 'crypto tail sensitivity'),
    'weekday_effect_120': (f_weekday_effect_120, 'mean Monday minus Friday return (120d)', 'calendar weekday seasonality'),
    'tom_effect_120': (f_tom_effect_120, 'mean early-month minus late-month return (120d)', 'turn-of-month calendar effect'),
    'overnight_intraday_corr_20': (f_overnight_intraday_corr_20, 'corr(overnight gap, intraday ret) over 20d', 'within-day gap continuation/reversal'),
    'trend_r2_60': (f_trend_r2_60, 'R^2 of 60d linear trend on close', 'trend quality/consistency'),
    'dd_depth_60': (f_dd_depth_60, 'close/rolling_max(60)-1 (drawdown depth)', 'drawdown depth risk'),
    'tech_beta_diff_60': (f_tech_beta_diff_60, 'beta(NDX,60) - beta(SPX,60)', 'relative tech sensitivity'),
    'wti_beta_cond_60x20': (f_wti_beta_cond, 'beta on WTI on WTI-up days x WTI 20d trend', 'oil sensitivity conditional'),
    'jpy_beta_cond_60x20': (f_jpy_beta_cond, 'beta on USDJPY on JPY-up days x USDJPY 20d trend', 'JPY carry sensitivity'),
    'vol_skew_20_60': (f_vol_skew_20_60, 'skew(r,20)-skew(r,60)', 'skew term structure'),
    'autocorr_abs_ret_20': (f_autocorr_abs_ret_20, 'lag-1 autocorr of |r| over 20d', 'volatility clustering persistence'),
}

fwd_ret = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}


def full_validate(fid, panel):
    m = validate_factor(fid, panel, prices)
    if m is None:
        return None
    ic10 = rank_ic_series(panel, fwd_ret[10], 8)
    ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    for nm, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                     ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                     ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
        sub = ic10[(ic10.index >= pd.Timestamp(a)) & (ic10.index <= pd.Timestamp(b))]
        m[nm] = float(sub.mean()) if len(sub) > 30 else float('nan')
    recent = ic10[(ic10.index >= pd.Timestamp('2025-07-15')) & (ic10.index <= pd.Timestamp('2026-07-15'))]
    if len(recent) > 30:
        m['recent_1y_ic'] = float(recent.mean())
        m['recent_1y_icir'] = float(recent.mean() / recent.std(ddof=1)) if recent.std(ddof=1) > 0 else 0.0
    return m


results = {}
for fid, (fn, desc, tag) in cands.items():
    try:
        panel = factor_to_panel(fn, prices)
        if panel is None or len(panel) == 0:
            print(f"{fid}: EMPTY panel -> skip", flush=True)
            continue
        m = full_validate(fid, panel)
        if m is None:
            print(f"{fid}: insufficient data -> None", flush=True)
            continue
        rho, rho_id, per_id = max_lib_corr(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rho_id
        m['per_factor_rho'] = {k: round(v, 3) for k, v in sorted(per_id.items(), key=lambda kv: -abs(kv[1]))[:4]}
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
        results[fid] = {'ok': bool(ok), 'metrics': {k: v for k, v in m.items()}, 'desc': desc, 'tag': tag}
        dec = {h: round(v, 4) for h, v in m['decay_ic_by_horizon'].items()}
        print(f"\n{fid}: IC10={m['ic']:.4f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
              f"coverage={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rho_id})", flush=True)
        print(f"  decay: {dec}", flush=True)
        print(f"  top rho: {m['per_factor_rho']}", flush=True)
        for nm in ['ic_2020_2022', 'ic_2023_2024', 'ic_2025_2026']:
            print(f"  {nm}: {m.get(nm, float('nan')):.4f}", flush=True)
        if 'recent_1y_ic' in m:
            print(f"  recent_1y: ic={m['recent_1y_ic']:.4f} icir={m['recent_1y_icir']:.4f}", flush=True)
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007, |ICIR|={abs(m['icir']):.4f}/0.084, rho={rho:.3f}/0.5)", flush=True)
    except Exception as e:
        print(f"{fid}: EXCEPTION {e}", flush=True)
        results[fid] = {'ok': False, 'error': str(e), 'desc': desc, 'tag': tag}

with open('scripts/miner_3_20261008_results_round22.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)

print("\n=== SUMMARY ===")
for fid, r in sorted(results.items()):
    if 'metrics' in r:
        m = r['metrics']
        print(f"{fid:24s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m.get('max_abs_library_correlation', float('nan')):.3f} ({m.get('max_corr_library_id')})")
    else:
        print(f"{fid:24s} ERROR {r.get('error', '')[:80]}")
