"""Round 11 (miner_1): novel factor families vs the 14-factor library artifacts.

Structurally new ideas (not previously screened in repo):
 1. weekday_ret_12/20/40  - same-weekday average return over trailing N weeks (weekly seasonality)
 2. month_season_prior    - prior-year same-calendar-month average monthly return (monthly seasonality)
 3. ret_entropy_20        - Shannon entropy of 20d return distribution (distributional randomness)
 4. amihud_illiq_20       - mean(|ret|/volume) over 20d (Amihud illiquidity)
 5. up_streak_20 / down_streak_20 / streak_asym_20 - max consecutive up/down-day runs (streak structure)
 6. turn_of_month_12      - average 5-day window return around trailing 12 month-ends (calendar effect)
 7. slope_beta_60         - rolling beta of asset returns to (CN10Y-US10Y) spread change
 8. time_since_low_260    - log days since the 260d rolling low (duration structure)
 9. lag5_autocorr_60      - weekly (lag-5) return autocorrelation over 60d
10. open_to_open_mom_20   - 20d momentum computed from open-to-open returns (gap-adjusted)
11. ndx_beta_60 / xau_beta_60 - rolling beta to NDX / XAU (new beta targets)

Admission gate: |IC|>=0.007, |ICIR|>=0.084, max library rho < 0.5 (artifact-based).
"""
import sys, json, glob, traceback
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, factor_to_panel, validate_factor)

np.seterr(all='ignore')

prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices loaded: {len(prices)} assets; canonical grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})")

# ---------- library artifacts (real signal matrices from EFFECTIVE factors) ----------
lib = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            art = d.get('signal_artifact')
            if art and Path('factors', art).exists():
                lib[d['factor_id']] = np.load(Path('factors', art))
    except Exception as e:
        print("lib skip", f, e)
print(f"library artifacts loaded: {len(lib)} -> {sorted(lib)}")


def max_lib_corr(panel):
    arr = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, la in lib.items():
        if la.shape[0] < arr.shape[0]:
            arr_use = arr[-la.shape[0]:]
        else:
            arr_use = arr
        corrs = []
        for i in range(arr_use.shape[0]):
            x, y = arr_use[i], la[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                r = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())
                if np.isfinite(r):
                    corrs.append(r)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


def rolling_beta(r_asset, r_sig, win=60):
    b = r_asset.rolling(win).cov(r_sig) / r_sig.rolling(win).var().replace(0, np.nan)
    return b


def f_weekday_ret(df, s, n):
    r = df['close'].pct_change()
    out = None
    for k in range(1, n + 1):
        sh = r.shift(7 * k)
        out = sh if out is None else out.add(sh, fill_value=np.nan)
    return out / n


def f_weekday_12(df, s): return f_weekday_ret(df, s, 12)
def f_weekday_20(df, s): return f_weekday_ret(df, s, 20)
def f_weekday_40(df, s): return f_weekday_ret(df, s, 40)


def f_month_season_prior(df, s):
    """Prior-year same-calendar-month avg monthly return (fully causal: only years < current)."""
    c = df['close']
    y = df.index.year.values
    m = df.index.month.values
    g = df.groupby([pd.Series(y, index=df.index, name='year'),
                    pd.Series(m, index=df.index, name='month')])['close'].last()
    g = g.reset_index()
    g['prev_close'] = g['last_close'].shift(1)
    g['mret'] = g['last_close'] / g['prev_close'] - 1.0
    g['key'] = g['year'] * 12 + g['month']
    mret_by_key = dict(zip(g['key'], g['mret']))
    vals = np.full(len(df), np.nan)
    keys = y * 12 + m
    for i in range(len(df)):
        yrs = np.arange(2020, y[i])
        ks = yrs * 12 + m[i]
        arr = [mret_by_key.get(k) for k in ks]
        arr = [a for a in arr if a is not None and np.isfinite(a)]
        if len(arr) >= 2:
            vals[i] = float(np.mean(arr))
    return pd.Series(vals, index=df.index)


def f_ret_entropy_20(df, s):
    r = df['close'].pct_change()
    sig = r.rolling(60, min_periods=20).std().replace(0, np.nan)
    b1 = (r < -sig).astype(float)
    b2 = ((r >= -sig) & (r < 0)).astype(float)
    b3 = ((r >= 0) & (r < sig)).astype(float)
    b4 = (r >= sig).astype(float)
    ent = None
    for b in [b1, b2, b3, b4]:
        p = b.rolling(20, min_periods=15).mean()
        term = (-p * np.log(p)).where(p > 0, 0.0)
        ent = term if ent is None else ent.add(term, fill_value=0.0)
    return ent


def f_amihud_illiq_20(df, s):
    r = df['close'].pct_change().abs()
    v = df['volume'].replace(0, np.nan)
    return (r / v).rolling(20, min_periods=10).mean()


def _max_run(x):
    best, run = 0, 0
    for v in x:
        run = run + 1 if v > 0 else 0
        best = max(best, run)
    return best


def f_streak(df, s, side):
    r = df['close'].pct_change()
    sgn = (r > 0).astype(int) if side == 'up' else (r < 0).astype(int)
    return sgn.rolling(20).apply(_max_run, raw=True)


def f_up_streak(df, s): return f_streak(df, s, 'up')
def f_down_streak(df, s): return f_streak(df, s, 'down')


def f_streak_asym(df, s):
    r = df['close'].pct_change()
    up = (r > 0).astype(int).rolling(20).apply(_max_run, raw=True)
    dn = (r < 0).astype(int).rolling(20).apply(_max_run, raw=True)
    return up - dn


def f_turn_of_month_12(df, s):
    """Avg 5-trading-day return around trailing 12 completed month-ends (value known at e+2)."""
    c = df['close']
    idx = df.index
    me_mask = idx.month != idx.shift(-1).month
    me_dates = idx[me_mask]
    win = {}
    for e in me_dates:
        pos = df.index.get_loc(e)
        e3 = pos - 3
        e2 = pos + 2
        if e3 >= 0 and e2 < len(df):
            win[e2] = c.iloc[e2] / c.iloc[e3] - 1.0  # window return known at e+2
    wser = pd.Series(win).sort_index()
    out = {}
    seen = []
    for dt, wv in wser.items():
        seen.append(wv)
        if len(seen) > 12:
            seen = seen[-12:]
        out[dt] = float(np.mean(seen[:-1])) if len(seen) > 1 else np.nan
    o = pd.Series(out).reindex(df.index).ffill()
    return o


def f_slope_beta_60(df, s):
    cn = prices['CN10Y']['close'].pct_change()
    us = prices['US10Y']['close'].pct_change()
    sp = (cn - us).replace(0, np.nan)
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), sp.rename('sp')], axis=1).dropna()
    return rolling_beta(z['r'], z['sp'], 60).reindex(z.index)


def f_time_since_low_260(df, s):
    c = df['close']
    argmin = c.rolling(260, min_periods=60).apply(lambda x: int(np.argmin(x)), raw=True)
    days = (259 - argmin).clip(lower=0)
    return np.log1p(days)


def f_lag5_autocorr_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=30).apply(
        lambda x: np.corrcoef(x[:-5], x[5:])[0, 1] if len(x) > 10 else np.nan, raw=True)


def f_open_to_open_mom_20(df, s):
    o = df['open']
    return o.shift(5) / o.shift(25) - 1.0


def f_ndx_beta_60(df, s):
    nd = prices['NDX']['close'].pct_change()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), nd.rename('nd')], axis=1).dropna()
    return rolling_beta(z['r'], z['nd'], 60).reindex(z.index)


def f_xau_beta_60(df, s):
    xa = prices['XAU']['close'].pct_change()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), xa.rename('xa')], axis=1).dropna()
    return rolling_beta(z['r'], z['xa'], 60).reindex(z.index)


candidates = {
    'weekday_ret_12': f_weekday_12,
    'weekday_ret_20': f_weekday_20,
    'weekday_ret_40': f_weekday_40,
    'month_season_prior': f_month_season_prior,
    'ret_entropy_20': f_ret_entropy_20,
    'amihud_illiq_20': f_amihud_illiq_20,
    'up_streak_20': f_up_streak,
    'down_streak_20': f_down_streak,
    'streak_asym_20': f_streak_asym,
    'turn_of_month_12': f_turn_of_month_12,
    'slope_beta_60': f_slope_beta_60,
    'time_since_low_260': f_time_since_low_260,
    'lag5_autocorr_60': f_lag5_autocorr_60,
    'open_to_open_mom_20': f_open_to_open_mom_20,
    'ndx_beta_60': f_ndx_beta_60,
    'xau_beta_60': f_xau_beta_60,
}

results = {}
for fid, fn in candidates.items():
    try:
        panel = factor_to_panel(fn, prices)
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid}: INSUFFICIENT DATA (panel {panel.shape})")
            results[fid] = dict(ok=False, error='insufficient_data', panel_shape=list(panel.shape))
            continue
        rho, fid_lib = max_lib_corr(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = fid_lib
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
        results[fid] = dict(ok=ok, metrics=m)
        print(f"\n=== {fid} === panel {panel.shape} "
              f"range {panel.index.min().date()}..{panel.index.max().date()}")
        print(f"IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
              f"ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
              f"maxlibrho={rho:.3f}({fid_lib})")
        print("decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()})
        print(f"ADMISSION: |IC|={abs(m['ic']):.4f} {'PASS' if abs(m['ic'])>=0.007 else 'FAIL'} | "
              f"|ICIR|={abs(m['icir']):.4f} {'PASS' if abs(m['icir'])>=0.084 else 'FAIL'} | "
              f"rho={rho:.3f} {'PASS' if rho<0.5 else 'FAIL'} -> {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"\n=== {fid} === ERROR: {e}")
        traceback.print_exc()
        results[fid] = dict(ok=False, error=str(e))

json.dump({k: v for k, v in results.items()},
          open('scripts/miner_1_20260730_results_round11.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_1_20260730_results_round11.json")
