"""Shared IC validation harness for miner_3 (2035-06-29 cycle).
Loads 15-asset panel through visible date 2035-06-28, computes rank IC of a
factor vs forward returns at horizons 1/5/10, and reports gate metrics.
Usage: import ic_harness; ic_harness.evaluate(factor_fn, params)
"""
import pandas as pd
import numpy as np

SYMS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
        'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
VIS = pd.Timestamp('2035-06-28')
MIN_ASSETS = 8
GATE_IC = 0.007
GATE_ICIR = 0.084


def load_panel():
    closes = {}
    for s in SYMS:
        df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= VIS].set_index('date')['close']
        closes[s] = df
    panel = pd.DataFrame(closes).sort_index()
    # restrict to dates where at least 8 assets have data
    panel = panel.dropna(thresh=MIN_ASSETS)
    return panel


def forward_returns(panel, h):
    rets = panel.shift(-h) / panel - 1.0
    return rets


def spearman_ic(factor, fwd):
    """Date-wise Spearman IC between factor cross-section and fwd-return cross-section."""
    dates = factor.index
    ics = []
    for dt in dates:
        if dt not in fwd.index:
            continue
        x = factor.loc[dt]
        y = fwd.loc[dt]
        mask = x.notna() & y.notna()
        if mask.sum() < MIN_ASSETS:
            continue
        ics.append(x[mask].rank().corr(y[mask].rank()))
    return np.array(ics)


def evaluate(factor_df, label, horizons=(1, 5, 10), n_obs_min=200):
    """factor_df: DataFrame dates x symbols of factor values.
    Returns dict of metrics per horizon."""
    panel = factor_df
    results = {}
    for h in horizons:
        fwd = forward_returns(panel, h)
        ics = spearman_ic(panel, fwd)
        ics = ics[np.isfinite(ics)]
        if len(ics) < n_obs_min:
            results[h] = {'n': len(ics), 'pass': False, 'reason': f'n={len(ics)} < {n_obs_min}'}
            continue
        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics, ddof=1))
        icir = float(ic_mean / ic_std * np.sqrt(len(ics))) if ic_std > 0 else 0.0
        hit = float(np.mean(ics > 0))
        results[h] = {
            'ic': ic_mean, 'icir': icir, 'hit': hit, 'n': len(ics),
            'pass': abs(ic_mean) >= GATE_IC and abs(icir) >= GATE_ICIR,
        }
    return results


def print_results(label, results):
    print(f"\n=== {label} ===")
    for h, r in results.items():
        if 'reason' in r:
            print(f"  h={h}: {r['reason']}")
        else:
            print(f"  h={h}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} "
                  f"n={r['n']} {'PASS' if r['pass'] else 'fail'}")
