"""miner2 2030-02-15: reusable factor validation library (cross-asset 15-name panel).
Computes daily rank IC / ICIR / hit / coverage / turnover / decay.
Admission gates: abs(IC1) >= 0.0070, abs(ICIR1) >= 0.0840.
"""
import pandas as pd, numpy as np

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']


def load_panel(path='scripts/panel_cache_20300215.pkl'):
    with open(path, 'rb') as f:
        return pd.read_pickle(f)


def fwd_ret(close, h):
    return close.shift(-h) / close - 1.0


def daily_rank_ic(signal, fwd, min_n=8):
    """Row-wise Spearman rank IC between signal and forward return."""
    ics, dates = [], []
    idx = signal.index.intersection(fwd.index)
    for t in idx:
        s = signal.loc[t]
        f = fwd.loc[t]
        m = s.notna() & f.notna()
        if m.sum() < min_n:
            continue
        ic = s[m].rank().corr(f[m].rank())
        if np.isfinite(ic):
            ics.append(ic)
            dates.append(t)
    return np.array(ics), np.array(dates)


def eval_factor(signal, close, horizons=(1, 2, 3, 5, 10), min_n=8, start=None, end=None):
    if start is not None:
        signal = signal[signal.index >= start]
    if end is not None:
        signal = signal[signal.index <= end]
    out = {}
    for h in horizons:
        fwd = fwd_ret(close, h)
        ics, dates = daily_rank_ic(signal, fwd, min_n=min_n)
        if len(ics) == 0:
            out[h] = dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
            continue
        ic = float(np.mean(ics))
        sd = float(np.std(ics, ddof=1))
        icir = ic / sd if sd > 0 else np.nan
        hit = float(np.mean(ics > 0))
        out[h] = dict(ic=ic, icir=icir, hit=hit, n=len(ics))
    # coverage
    cov = float(signal.notna().mean().mean()) if signal.shape[0] else 0.0
    # turnover: mean absolute daily change of cross-sectional rank percentile
    rp = signal.rank(axis=1, pct=True)
    to = float(rp.diff().abs().mean().mean()) if rp.shape[0] > 1 else 0.0
    out['coverage'] = cov
    out['turnover_1d_rank'] = to
    out['n_dates'] = int(signal.shape[0])
    return out


def summarize(res, label=''):
    h1 = res.get(1, {})
    h5 = res.get(5, {})
    h10 = res.get(10, {})
    print(f"[{label}] IC1={h1.get('ic', float('nan')):.4f} ICIR1={h1.get('icir', float('nan')):.3f} "
          f"hit1={h1.get('hit', float('nan')):.3f} n1={h1.get('n', 0)} | "
          f"IC5={h5.get('ic', float('nan')):.4f} ICIR5={h5.get('icir', float('nan')):.3f} | "
          f"IC10={h10.get('ic', float('nan')):.4f} ICIR10={h10.get('icir', float('nan')):.3f} | "
          f"cov={res.get('coverage', float('nan')):.3f} turn1d={res.get('turnover_1d_rank', float('nan')):.3f} "
          f"dates={res.get('n_dates', 0)}")
    return h1
