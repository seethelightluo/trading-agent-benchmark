"""miner_2: refined crisis-beta factor with full-date coverage.

crisis_beta_120: rolling 120-calendar-day beta of asset to SPX, computed using
ONLY returns observed on crisis days (SPX 20d vol > its 1y rolling median),
requiring >=20 crisis observations in the window. Values defined on all dates
(forward-filled from the most recent window estimate), not just crisis days.
"""
import sys, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel,
                           validate_factor, WATCHLIST, VAL_START, VAL_END,
                           canonical_grid, signal_matrix)

prices = load_prices(days=2000)
spx = prices.get('SPX')

def crisis_beta_120(df, s, min_obs=20, win=120):
    if spx is None:
        return None
    r = df['close'].pct_change()
    rs = spx['close'].pct_change()
    spx_vol = rs.rolling(20).std()
    med = spx_vol.rolling(252).median()
    crisis = (spx_vol > med).astype(float)
    z = pd.concat([r.rename('r'), rs.rename('s'), crisis.rename('c')], axis=1).dropna()
    rv = z['r'].values.astype(float)
    sv = z['s'].values.astype(float)
    cv = z['c'].values.astype(float)
    n = len(z)
    out = np.full(n, np.nan)
    idx = z.index
    # rolling window beta using only crisis rows inside window
    for t in range(win, n):
        w0 = t - win
        m = cv[w0:t] > 0.5
        if m.sum() < min_obs:
            continue
        x = rv[w0:t][m]
        y = sv[w0:t][m]
        if y.std(ddof=1) < 1e-12:
            continue
        b = np.cov(x, y, ddof=1)[0, 1] / np.var(y, ddof=1)
        out[t] = b
    out_series = pd.Series(out, index=idx)
    # forward-fill to cover non-crisis days with the most recent crisis-beta estimate
    out_series = out_series.ffill()
    return out_series

# --- library correlation vs real signal artifacts of the 12 effective factors ---
EFFECTIVE = ['dd_duration_120_resid', 'down_beta_60', 'dxy_beta_cond_60x20',
             'eurusd_beta_cond_60x20', 'hilo_pos_60', 'hs300_beta_60', 'max_ret_20d',
             'skew_term_20_60', 'spx_beta_60', 'vix_beta_cond_60x20',
             'vol_adj_mom_20_60', 'vol_of_vol20x60']

def lib_corr(panel):
    best, best_id = 0.0, None
    for fid in EFFECTIVE:
        try:
            arr = np.load(f'factors/{fid}_signal.npy', allow_pickle=False)
        except Exception:
            print(f'  (no artifact for {fid})')
            continue
        grid = canonical_grid(prices)
        lib = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
        common = panel.index.intersection(lib.index)
        corrs = []
        for d in common:
            x, y = panel.loc[d], lib.loc[d]
            m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                c = x[m].rank().corr(y[m].rank())
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

for min_obs in (15, 20, 25):
    fid = f'crisis_beta_120_mo{min_obs}'
    panel = factor_to_panel(lambda df, s, mo=min_obs: crisis_beta_120(df, s, min_obs=mo), prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data')
        continue
    rho, best = lib_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = best
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    print(f'{fid}: ic={m["ic"]:+.4f} icir={m["icir"]:+.4f} hit={m["ic_hit_ratio"]:.3f} '
          f'n={m["n_ic_dates"]} cov={m["coverage_asset_days"]:.3f} ge8={m["coverage_dates_ge8"]:.3f} '
          f'turn={m["turnover_10d_rank"]:.2f} rho={rho:.3f} vs {best} '
          f'decay10={m["decay_ic_by_horizon"]["10"]:+.4f} decay20={m["decay_ic_by_horizon"]["20"]:+.4f} '
          f'-> {"ADMIT" if ok else "skip"}')
