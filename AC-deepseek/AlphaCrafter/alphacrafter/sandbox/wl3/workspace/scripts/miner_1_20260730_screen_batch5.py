"""miner_1 2026-07-30 batch-5 screening: NEW orthogonal factor families.

Library (12 EFFECTIVE) already covers: beta-to-anchor (spx/hs300/copper/usdcny/
ndx/btc/wti/us10y/cn10y), conditional beta (dxy/eurusd/usdjpy/xau/wti/vix/cn10y/
us10y), vol-adj momentum, hilo range position, skew term-structure, vol-of-vol,
max ret, drawdown-duration residual, RSI/bollinger/brkout (rejected).

THIS batch (all NEW, interpretable):
  1. cvar_20           : 20d 5% CVaR of daily returns (tail-risk level)
  2. coskew_60         : 60d co-skewness of asset ret with SPX ret (sys tail)
  3. down_beta_60      : SPX beta estimated on market-down days only
  4. beta_asym_60      : downside-beta minus upside-beta (corr asymmetry)
  5. parkinson_vol_20  : 20d high-low range volatility estimator
  6. sharpe_60         : 60d mean daily ret / std (risk-adjusted return)
  7. accel_mom_60_20   : 60d momentum minus 20d momentum (trend acceleration)
  8. min_ret_20d       : worst daily return over 20d (tail-loss memory)
  9. kurtosis_20       : realized excess kurtosis of 20d daily returns
 10. treynor_60        : 60d return / 60d SPX beta (sys-risk-adjusted ret)

Gate: |IC(h=10)| >= 0.007 AND |ICIR| >= 0.084 on 2020-01-01..2026-07-15;
rho vs current 12-factor artifact library < 0.5 recommended for gate survival.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel, WATCHLIST,
                           canonical_grid, VAL_START, VAL_END, validate_factor)

prices = load_prices(days=2200)
spx = load_index('SPX', prices=prices) or prices.get('SPX')
grid = canonical_grid(prices)
print(f'[load] {len(prices)} assets; grid n={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---------------- library panels from artifacts (current effective set) -------
EFFECTIVE = ['copper_beta_60', 'dd_duration_120_resid', 'dxy_beta_cond_60x20',
             'eurusd_beta_cond_60x20', 'hilo_pos_60', 'hs300_beta_60',
             'max_ret_20d', 'skew_term_20_60', 'spx_beta_60',
             'vix_beta_cond_60x20', 'vol_adj_mom_20_60', 'vol_of_vol20x60']
EURUSD = load_index('EURUSD', prices=prices)
VIX = load_index('VIX', prices=prices)


def artifact_panel(fid):
    """Reconstruct wide panel from persisted .npy (or rebuild if missing)."""
    p = f'factors/{fid}_signal.npy'
    try:
        arr = np.load(p, allow_pickle=False)
        return pd.DataFrame(arr, index=grid, columns=WATCHLIST)
    except Exception:
        pass
    # rebuild the 3 library factors lacking artifacts
    if fid == 'eurusd_beta_cond_60x20':
        def f(df, s):
            if EURUSD is None: return None
            r = df['close'].pct_change(); vr = EURUSD['close'].pct_change()
            z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
            b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
            return (b * (EURUSD['close'] / EURUSD['close'].shift(20) - 1.0)).reindex(z.index)
    elif fid == 'hilo_pos_60':
        def f(df, s):
            return (df['close'] - df['low'].rolling(60).min()) / (df['high'].rolling(60).max() - df['low'].rolling(60).min())
    elif fid == 'vix_beta_cond_60x20':
        def f(df, s):
            if VIX is None: return None
            r = df['close'].pct_change(); vr = VIX['close'].pct_change()
            z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
            b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
            return (-b * (VIX['close'] / VIX['close'].shift(20) - 1.0)).reindex(z.index)
    else:
        return None
    return factor_to_panel(f, prices)


lib_panels = {}
for fid in EFFECTIVE:
    pan = artifact_panel(fid)
    if pan is not None and len(pan):
        lib_panels[fid] = pan
print(f'[lib] {len(lib_panels)} effective library panels ready', flush=True)


def max_lib_corr(panel):
    best, best_id = 0.0, None
    for fid, lp in lib_panels.items():
        idx = panel.index.intersection(lp.index)
        corrs = []
        for d in idx:
            x, y = panel.loc[d], lp.loc[d]
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


# ---------------- candidate definitions ---------------------------------------
def f_cvar20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).quantile(0.05)


def f_coskew60(df, s):
    r = df['close'].pct_change()
    rm = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
    mu_r = z['r'].rolling(60).mean(); mu_m = z['m'].rolling(60).mean()
    sr = z['r'].rolling(60).std(); sm = z['m'].rolling(60).std()
    num = ((z['r'] - mu_r) * (z['m'] - mu_m) ** 2).rolling(60).mean()
    return (num / (sr * sm ** 2)).reindex(z.index)


def f_down_beta60(df, s):
    r = df['close'].pct_change()
    rm = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
    down = z[z['m'] < 0]
    b = down['r'].rolling(60).cov(down['m']) / down['m'].rolling(60).var()
    return b.reindex(z.index)


def f_beta_asym60(df, s):
    r = df['close'].pct_change()
    rm = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
    down = z[z['m'] < 0]
    up = z[z['m'] >= 0]
    bd = down['r'].rolling(60).cov(down['m']) / down['m'].rolling(60).var()
    bu = up['r'].rolling(60).cov(up['m']) / up['m'].rolling(60).var()
    return (bd - bu).reindex(z.index)


def f_parkinson20(df, s):
    h, l, c = df['high'], df['low'], df['close']
    rng = (np.log(h / l) ** 2 / (4 * np.log(2))).rolling(20).mean()
    return np.sqrt(rng)


def f_sharpe60(df, s):
    r = df['close'].pct_change()
    mu = r.rolling(60).mean()
    sd = r.rolling(60).std()
    return (mu / sd).reindex(r.index)


def f_accel_mom(df, s):
    m60 = df['close'].shift(5) / df['close'].shift(65) - 1.0
    m20 = df['close'].shift(5) / df['close'].shift(25) - 1.0
    return m60 - m20


def f_min_ret20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).min()


def f_kurt20(df, s):
    r = df['close'].pct_change()
    mu = r.rolling(20).mean(); sd = r.rolling(20).std()
    return ((r - mu) ** 4).rolling(20).mean() / sd ** 4 - 3.0


def f_treynor60(df, s):
    r = df['close'].pct_change()
    rm = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['m']) / z['m'].rolling(60).var()
    ret60 = z['r'].rolling(60).sum()
    return (ret60 / b).reindex(z.index)


CANDIDATES = [
    ('cvar_20', f_cvar20, '20d 5% CVaR tail risk'),
    ('coskew_60', f_coskew60, '60d co-skewness with SPX'),
    ('down_beta_60', f_down_beta60, 'SPX downside-only beta 60d'),
    ('beta_asym_60', f_beta_asym60, 'down minus up beta 60d'),
    ('parkinson_vol_20', f_parkinson20, '20d Parkinson (high-low) vol'),
    ('sharpe_60', f_sharpe60, '60d mean/std risk-adjusted return'),
    ('accel_mom_60_20', f_accel_mom, '60d-20d momentum acceleration'),
    ('min_ret_20d', f_min_ret20, '20d worst daily return'),
    ('kurtosis_20', f_kurt20, '20d realized excess kurtosis'),
    ('treynor_60', f_treynor60, '60d return / 60d SPX beta'),
]

results = {}
for fid, fn, desc in CANDIDATES:
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient -> None', flush=True)
        continue
    rho, rho_id = max_lib_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    results[fid] = {'ok': ok, 'metrics': m}
    print(f"\n=== {fid} ({desc}) shape={panel.shape} ===", flush=True)
    print(f"  IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
          f"cov8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.3f} "
          f"maxlibrho={rho:.3f}({rho_id})", flush=True)
    print(f"  decay={ {h: round(v,4) for h, v in m['decay_ic_by_horizon'].items()} }", flush=True)
    print(f"  ADMISSION: {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f} "
          f"{'ok' if abs(m['ic'])>=0.007 else 'NO'}, |ICIR|={abs(m['icir']):.4f} "
          f"{'ok' if abs(m['icir'])>=0.084 else 'NO'})", flush=True)

with open('scripts/miner_1_20260730_results_batch5.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)
print('\n[done] saved scripts/miner_1_20260730_results_batch5.json')
