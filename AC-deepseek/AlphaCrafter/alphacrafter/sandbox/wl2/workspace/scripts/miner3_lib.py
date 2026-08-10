"""Shared validation framework for miner_3.

Loads tradable asset closes through visible_through, computes factor values,
daily cross-sectional rank IC vs forward returns, ICIR, turnover, coverage,
decay, and max abs correlation vs existing library factors.

Methodology mirrors library factors: daily Spearman rank IC across the
15-asset cross-section (>=8 valid instruments per date), forward-return
horizon 10d for admission, decay table over horizons 1/2/3/5/10/20.
"""
import json
import glob
import numpy as np
import pandas as pd

VISIBLE = '2026-07-29'
ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
HORIZONS = [1, 2, 3, 5, 10, 20]
ADMISSION_HORIZON = 10
MIN_ASSETS = 8


def load_close(asset, folder='../persistent/stock_data'):
    df = pd.read_csv(f'{folder}/{asset}.csv', parse_dates=['date'])
    df = df[df['date'] <= pd.Timestamp(VISIBLE)].set_index('date')['close'].astype(float)
    return df


def load_obs(asset):
    df = pd.read_csv(f'../persistent/index_data/{asset}.csv', parse_dates=['date'])
    df = df[df['date'] <= pd.Timestamp(VISIBLE)].set_index('date')['close'].astype(float)
    return df


def build_panel():
    return {a: load_close(a) for a in ASSETS}


def forward_returns(prices, horizon):
    """Per-asset forward returns on each asset's own calendar, reindexed."""
    fwd = {}
    for a, s in prices.items():
        f = (s.shift(-horizon) / s - 1.0)
        fwd[a] = f
    return pd.DataFrame(fwd)


def spearman_ic(factor_df, fwd_df):
    """Daily Spearman rank IC between factor values and forward returns."""
    dates, ics = [], []
    idx = factor_df.index.intersection(fwd_df.index)
    for d in idx:
        f = factor_df.loc[d]
        r = fwd_df.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < MIN_ASSETS:
            continue
        ic = f[mask].corr(r[mask], method='spearman')
        if np.isfinite(ic):
            dates.append(d)
            ics.append(ic)
    return pd.Series(ics, index=dates)


def mean_rank_turnover(factor_df, step=10):
    """Mean absolute rank change over `step` days (rank in [0,1])."""
    ranks = factor_df.rank(axis=1, pct=True)
    chg = (ranks - ranks.shift(step)).abs()
    return float(chg.stack().mean())


def max_abs_library_correlation(factor_df):
    """Pooled Pearson correlation of raw factor values vs each library factor."""
    best = 0.0
    names = []
    for fn in sorted(glob.glob('factors/*.json')):
        try:
            with open(fn) as f:
                meta = json.load(f)
        except Exception:
            continue
        fid = meta.get('factor_id')
        if not fid or fid == 'factor_ensemble':
            continue
        exp = meta.get('calculation', {}).get('expression', '')
        try:
            other = compute_library_signal(exp)
        except Exception:
            continue
        if other is None:
            continue
        both = pd.concat([factor_df.stack().rename('x'),
                          other.stack().rename('y')], axis=1).dropna()
        if len(both) < 50:
            continue
        rho = float(both['x'].corr(both['y']))
        if abs(rho) > best:
            best = abs(rho)
            names = [fid]
        elif abs(rho) == best:
            names.append(fid)
    return best, names


def compute_library_signal(expression):
    """Recompute a library factor's daily signal from its pandas expression."""
    prices = build_panel()
    panel = pd.DataFrame(prices)
    ret = panel.pct_change()
    df = panel.copy()
    ns = {'close': panel, 'pct_change': ret}
    env = {'pd': pd, 'np': np}
    env.update(ns)
    try:
        sig = eval(expression, {'__builtins__': {}}, env)
    except Exception:
        return None
    if isinstance(sig, pd.Series):
        sig = sig.to_frame()
    return sig


def run_validation(factor_fn, factor_id, tag):
    """factor_fn: prices -> factor DataFrame (index=date, columns=asset)."""
    prices = build_panel()
    factor_df = factor_fn(pd.DataFrame(prices))
    fwd = forward_returns(prices, ADMISSION_HORIZON)
    ic_series = spearman_ic(factor_df, fwd)
    ic = float(ic_series.mean())
    icir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
    hit = float((ic_series > 0).mean()) if ic >= 0 else float((ic_series < 0).mean())

    # decay at other horizons
    decay = {}
    for h in HORIZONS:
        fh = forward_returns(prices, h)
        s = spearman_ic(factor_df, fh)
        decay[str(h)] = round(float(s.mean()), 4)

    # coverage
    valid = factor_df.notna().sum().sum()
    total = factor_df.shape[0] * factor_df.shape[1]
    cov_asset_days = valid / total if total else 0.0
    n_ge8 = sum(1 for d in factor_df.index if factor_df.loc[d].notna().sum() >= MIN_ASSETS)
    cov_dates_ge8 = n_ge8 / len(factor_df) if len(factor_df) else 0.0

    turnover = mean_rank_turnover(factor_df)
    maxrho, rho_names = max_abs_library_correlation(factor_df)

    # regime split (rough calendar blocks)
    regime = {}
    blocks = [('2020-01-01', '2021-12-31'), ('2022-01-01', '2022-12-31'),
              ('2023-01-01', '2024-12-31'), ('2025-01-01', '2026-12-31')]
    for b0, b1 in blocks:
        sub = ic_series[(ic_series.index >= b0) & (ic_series.index <= b1)]
        if len(sub) >= 30:
            regime[f'{b0[:4]}-{b1[:4]}'] = {
                'ic': round(float(sub.mean()), 4),
                'icir': round(float(sub.mean() / sub.std()), 4) if sub.std() > 0 else 0.0,
                'n_dates': int(len(sub)),
            }

    print(f'==== {factor_id} [{tag}] ====')
    print(f'n_ic_dates={len(ic_series)}  ic={ic:.4f}  icir={icir:.4f}  hit={hit:.3f}')
    print(f'coverage_asset_days={cov_asset_days:.3f}  dates_ge8={cov_dates_ge8:.3f}  turnover_10d_rank={turnover:.3f}')
    print(f'decay_ic_by_horizon={decay}')
    print(f'max_abs_library_correlation={maxrho:.4f}  (vs {rho_names})')
    print(f'regime={json.dumps(regime)}')
    return {'factor_id': factor_id, 'ic': ic, 'icir': icir, 'hit': hit,
            'n_ic_dates': len(ic_series), 'cov_asset_days': cov_asset_days,
            'cov_dates_ge8': cov_dates_ge8, 'turnover': turnover, 'decay': decay,
            'maxrho': maxrho, 'rho_names': rho_names, 'regime': regime}
