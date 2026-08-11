"""Shared validation helper for miner_3.
Loads panel cache, computes factor signals, and reports IC/ICIR/hit/turnover/coverage
plus max abs correlation vs current library factor signals.
Usage: import validate_lib  (module-level helpers)
"""
import numpy as np
import pandas as pd
import pickle, json, glob

CACHE_PATH = 'scripts/panel_cache.pkl'

def load_cache():
    with open(CACHE_PATH, 'rb') as f:
        return pickle.load(f)

def library_signals():
    """Reconstruct library factor signal matrices for decorrelation check."""
    cache = load_cache()
    close = cache['close']
    ret = cache['ret']
    macro = cache['macro']
    signals = {}
    # mom_10d_skip5: close.shift(5)/close.shift(15)-1  (10d momentum skipping 5)
    signals['mom_10d_skip5'] = close.shift(5) / close.shift(15) - 1.0
    signals['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1.0
    # vol_of_vol20x60: std(pct_change,20).rolling(60).std()
    pct = close.pct_change()
    signals['vol_of_vol20x60'] = pct.rolling(20).std().rolling(60).std()
    # vix_beta_cond_60x20: -beta(asset_ret, VIX_ret, 60)*(VIX/VIX.shift(20)-1)
    vix = macro['VIX']
    vix_ret = vix.pct_change()
    def rolling_beta(x, y, w):
        out = pd.Series(np.nan, index=x.index)
        for i in range(w, len(x)):
            a = x.iloc[i-w:i].values
            b = y.iloc[i-w:i].values
            if np.std(b) > 0:
                out.iloc[i] = np.cov(a, b)[0, 1] / np.var(b)
        return out
    # vectorized beta via rolling cov
    cov = ret.rolling(60).cov(vix_ret)
    var = vix_ret.rolling(60).var()
    beta = cov / var
    vix_mom = vix / vix.shift(20) - 1.0
    signals['vix_beta_cond_60x20'] = -beta.multiply(vix_mom, axis=0)
    return signals

def max_abs_library_corr(signal_df, lib_signals, max_dates=2000):
    """Cross-sectional demeaned signal correlation with library factors."""
    s = signal_df.copy()
    s = s.sub(s.mean(axis=1), axis=0)
    best = 0.0
    for name, lf in lib_signals.items():
        lf = lf.copy()
        lf = lf.sub(lf.mean(axis=1), axis=0)
        common_idx = s.index.intersection(lf.index)
        if len(common_idx) < 60:
            continue
        s2 = s.loc[common_idx]
        l2 = lf.loc[common_idx]
        # sample dates to keep it fast
        step = max(1, len(common_idx) // max_dates)
        s2 = s2.iloc[::step]
        l2 = l2.iloc[::step]
        corrs = []
        for i in range(len(s2)):
            a = s2.iloc[i].values
            b = l2.iloc[i].values
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() >= 8:
                c = np.corrcoef(a[m], b[m])[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
        if corrs:
            best = max(best, np.nanmean(np.abs(corrs)))
    return best

def validate_factor(signal_df, forward_ret_df, horizons=(1, 2, 3, 5, 10, 20),
                    min_instruments=8, name='factor'):
    """Compute IC / ICIR / hit ratio / turnover / coverage for a signal.
    forward_ret_df: DataFrame of forward returns aligned on same index (value at t = ret t->t+h).
    """
    results = {}
    sig = signal_df.copy()
    for h in horizons:
        fwd = forward_ret_df[h]
        common = sig.index.intersection(fwd.index)
        ic_list = []
        for dt in common:
            a = sig.loc[dt]
            b = fwd.loc[dt]
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() >= min_instruments:
                c = np.corrcoef(a[m], b[m])[0, 1]
                if not np.isnan(c):
                    ic_list.append((dt, c))
        if len(ic_list) == 0:
            results[h] = dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
            continue
        ics = np.array([x[1] for x in ic_list])
        ic_mean = ics.mean()
        ic_std = ics.std(ddof=1) if len(ics) > 1 else np.nan
        icir = ic_mean / ic_std if ic_std and ic_std > 0 else 0.0
        hit = (np.sign(ic_mean) * ics > 0).mean() if ic_mean != 0 else 0.5
        results[h] = dict(ic=round(float(ic_mean), 4), icir=round(float(icir), 4),
                          hit=round(float(hit), 3), n=len(ics))
    # coverage
    valid = (~signal_df.isna()).sum().sum()
    total = signal_df.shape[0] * signal_df.shape[1]
    coverage = valid / total if total else 0.0
    # dates with >= 8 valid instruments
    n_dates_ge8 = (signal_df.notna().sum(axis=1) >= min_instruments).sum()
    n_dates_total = signal_df.shape[0]
    # turnover: rank change per 10d
    r = signal_df.rank(axis=1)
    r10 = r.shift(10)
    turn = (r - r10).abs().mean().mean() if len(r) > 10 else np.nan
    return dict(results=results, coverage=round(float(coverage), 3),
                dates_ge8=int(n_dates_ge8), dates_total=int(n_dates_total),
                turnover=round(float(turn), 3) if not np.isnan(turn) else None)

def build_forward_returns(cache, horizons=(1, 2, 3, 5, 10, 20)):
    close = cache['close']
    out = {}
    for h in horizons:
        out[h] = close.shift(-h) / close - 1.0
    return out

if __name__ == '__main__':
    cache = load_cache()
    fwd = build_forward_returns(cache)
    lib = library_signals()
    for name, s in lib.items():
        res = validate_factor(s, fwd, name=name)
        mac = max_abs_library_corr(s, lib)
        print(name, '-> ic10:', res['results'][10]['ic'], 'icir10:', res['results'][10]['icir'],
              'cov:', res['coverage'], 'maxcorr:', round(mac, 3))
