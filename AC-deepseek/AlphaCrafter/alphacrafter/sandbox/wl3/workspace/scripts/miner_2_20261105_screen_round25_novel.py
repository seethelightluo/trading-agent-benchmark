"""Round 25 novel factor screen (miner_2) - 2026-11-05.

Fresh ideas (cross-checked against full explored-id list incl. evicted/rejected/
quarantine/effective and all miner scripts; no duplicates):
- perm_entropy_20        : order-3 permutation entropy of returns over 20d (pattern complexity)
- lz_sign_complexity_60  : Lempel-Ziv complexity of return sign sequence over 60d (normalized)
- vol_regime_switch_20x60: crossings of 20d realized vol vs its 60d median over trailing 60d
- defensive_mom_20       : 20d momentum penalized by downside beta to SPX (quality/defensive momentum)
- range_skew_20          : skew of daily (high-low)/close over 20d
- volume_entropy_20      : entropy of 20d volume share distribution (activity concentration)
- candle_pos_20          : mean close position inside daily range over 20d
- mom_flip_60x20         : sign-flip frequency of 20d momentum over trailing 60d (trend stability)
- cvar_var_ratio_60      : CVaR5% / |VaR5%| ratio over 60d (tail curvature)
- max_drawup_dd_ratio_120: max drawup depth / |max drawdown| over 120d (reward-risk shape)
- gap_autocorr_20        : lag-1 autocorr of overnight gaps over 20d
- gap_follow_through_20  : corr(gap_t, ret_{t+1}) over 20d (do gaps continue or reverse?)

Validates vs full EFFECTIVE library (recoverable signal artifacts), per-date
Spearman rho. Gates: |IC|>=0.007, |ICIR|>=0.084, max_abs_library_correlation<0.5.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import math
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, WATCHLIST,
                           VAL_START, VAL_END)

TODAY = '2026-11-05'
t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---- library: effective factors with recoverable signal artifacts ----
lib = {}
for p in sorted(Path('factors').glob('*.json')):
    if p.name.endswith('.bak') or 'deprecated' in p.name or 'ensemble' in p.name:
        continue
    try:
        payload = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    if payload.get('validation', {}).get('status') != 'EFFECTIVE':
        continue
    art = payload.get('signal_artifact')
    art_path = p.parent / str(art) if art else None
    if art_path is not None and art_path.exists():
        lib[payload['factor_id']] = np.load(art_path, allow_pickle=False)
print(f'library factors with artifacts: {len(lib)} {sorted(lib.keys())}', flush=True)


def spearman_rho_vs_library(panel):
    """Per-date Spearman rho vs each library artifact; report max |rho| (gate-style)."""
    pm = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    per_id = {}
    for fid, lm in lib.items():
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


# ---- benchmark series ----
spx_ret = prices['SPX']['close'].pct_change()


def rolling_beta_down(asset_df, window=60, min_obs=12):
    """Beta to SPX computed only on days where SPX return < 0."""
    r = asset_df['close'].pct_change()
    z = pd.concat([r.rename('r'), spx_ret.rename('b')], axis=1).dropna()
    up = z[z['b'] < 0]
    cov = up['r'].rolling(window, min_periods=min_obs).cov(up['b'])
    var = up['b'].rolling(window, min_periods=min_obs).var()
    return (cov / var).reindex(z.index)


def _perm_entropy(x, order=3, lag=1):
    """Permutation entropy of a 1d array (order 3)."""
    n = len(x)
    if n < order * lag + 1:
        return np.nan
    patterns = {}
    for i in range(n - order * lag):
        idx = [i + j * lag for j in range(order)]
        vals = x[idx]
        perm = tuple(np.argsort(vals))
        patterns[perm] = patterns.get(perm, 0) + 1
    tot = sum(patterns.values())
    if tot == 0:
        return np.nan
    p = np.array(list(patterns.values()), dtype=float) / tot
    ent = -np.sum(p * np.log(p))
    return ent / math.log(math.factorial(order))


def _lz_complexity(bits):
    """Lempel-Ziv (LZ76) complexity of a binary sequence, normalized by n/log2(n)."""
    n = len(bits)
    if n < 2:
        return np.nan
    c, i, k, l = 1, 0, 1, 1
    while True:
        if i + k >= n:
            c += 1
            break
        sub = bits[i:i + k]
        if sub in bits[i + k:i + k + k]:
            k += 1
        else:
            c += 1
            i += k
            k = 1
        if i >= n:
            break
    norm = n / np.log2(n) if n > 1 else 1.0
    return c / norm


# ---- candidate factor functions ----
def f_perm_entropy_20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20, min_periods=12).apply(_perm_entropy, raw=True)


def f_lz_sign_complexity_60(df, s):
    r = df['close'].pct_change()
    def fn(x):
        bits = (x > 0).astype(int)
        return _lz_complexity(bits)
    return r.rolling(60, min_periods=30).apply(fn, raw=True)


def f_vol_regime_switch_20x60(df, s):
    v = df['close'].pct_change().rolling(20, min_periods=10).std()
    med = v.rolling(60, min_periods=30).median()
    ind = (v > med).astype(float)
    def fn(x):
        if len(x) < 8 or np.all(np.isnan(x)):
            return np.nan
        d = np.diff(np.nan_to_num(x))
        return np.mean(d != 0)
    return ind.rolling(60, min_periods=30).apply(fn, raw=True)


def f_defensive_mom_20(df, s):
    mom = df['close'] / df['close'].shift(20) - 1.0
    db = rolling_beta_down(df, 60, 12)
    w = (1.0 / (1.0 + db.clip(lower=0))).reindex(mom.index)
    return (mom * w).reindex(mom.index)


def f_range_skew_20(df, s):
    rr = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
    return rr.rolling(20, min_periods=12).skew()


def f_volume_entropy_20(df, s):
    v = df['volume'].replace(0, np.nan)
    def fn(x):
        x = x[~np.isnan(x)]
        if len(x) < 10 or x.sum() <= 0:
            return np.nan
        p = x / x.sum()
        return -np.sum(p * np.log(p)) / np.log(len(p))
    return v.rolling(20, min_periods=10).apply(fn, raw=True)


def f_candle_pos_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    pos = (df['close'] - df['low']) / rng
    return pos.rolling(20, min_periods=10).mean()


def f_mom_flip_60x20(df, s):
    m = df['close'] / df['close'].shift(20) - 1.0
    sgn = np.sign(m)
    def fn(x):
        x = x[~np.isnan(x)]
        if len(x) < 20:
            return np.nan
        flips = np.sum(np.diff(x) != 0)
        return flips / len(x)
    return sgn.rolling(60, min_periods=30).apply(fn, raw=True)


def f_cvar_var_ratio_60(df, s):
    r = df['close'].pct_change()
    def fn(x):
        x = x[~np.isnan(x)]
        if len(x) < 30:
            return np.nan
        q = np.percentile(x, 5)
        cvar = x[x <= q].mean()
        return cvar / q if q < 0 else np.nan
    return r.rolling(60, min_periods=30).apply(fn, raw=True)


def f_max_drawup_dd_ratio_120(df, s):
    c = np.log(df['close'])
    def fn(x):
        if len(x) < 60:
            return np.nan
        cum = x - x[0]
        dd = np.minimum(cum - np.maximum.accumulate(cum), 0)
        du = cum - np.minimum.accumulate(cum)
        mdd = -dd.min()
        mdu = du.max()
        return mdu / (mdd + 1e-12) if mdd > 0 else np.nan
    return c.rolling(120, min_periods=60).apply(fn, raw=True)


def f_gap_autocorr_20(df, s):
    g = df['open'] / df['close'].shift(1) - 1.0
    def fn(x):
        x = x[~np.isnan(x)]
        if len(x) < 8 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
            return np.nan
        return np.corrcoef(x[:-1], x[1:])[0, 1]
    return g.rolling(20, min_periods=8).apply(fn, raw=True)


def f_gap_follow_through_20(df, s):
    g = df['open'] / df['close'].shift(1) - 1.0
    r = df['close'].pct_change()
    def fn(x):
        if len(x) < 8:
            return np.nan
        a = x[:-1]; b = x[1:]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 8 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            return np.nan
        return np.corrcoef(a[m], b[m])[0, 1]
    return g.rolling(20, min_periods=8).apply(fn, raw=True)


CANDIDATES = [
    ('perm_entropy_20', f_perm_entropy_20),
    ('lz_sign_complexity_60', f_lz_sign_complexity_60),
    ('vol_regime_switch_20x60', f_vol_regime_switch_20x60),
    ('defensive_mom_20', f_defensive_mom_20),
    ('range_skew_20', f_range_skew_20),
    ('volume_entropy_20', f_volume_entropy_20),
    ('candle_pos_20', f_candle_pos_20),
    ('mom_flip_60x20', f_mom_flip_60x20),
    ('cvar_var_ratio_60', f_cvar_var_ratio_60),
    ('max_drawup_dd_ratio_120', f_max_drawup_dd_ratio_120),
    ('gap_autocorr_20', f_gap_autocorr_20),
    ('gap_follow_through_20', f_gap_follow_through_20),
]

results = {}
for fid, fn in CANDIDATES:
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
    except Exception as e:
        print(f'{fid}: factor_fn EXCEPTION {e}', flush=True)
        results[fid] = {'ok': False, 'error': str(e)}
        continue
    if panel.shape[0] < 100 or panel.shape[1] < 8:
        print(f'{fid}: panel too small {panel.shape} -> skip', flush=True)
        results[fid] = {'ok': False, 'metrics': None}
        continue
    try:
        m = validate_factor(fid, panel, prices)
    except Exception as e:
        print(f'{fid}: validate_factor EXCEPTION {type(e).__name__} {e}', flush=True)
        results[fid] = {'ok': False, 'error': str(e)}
        continue
    if m is None:
        print(f'{fid}: insufficient data -> None', flush=True)
        results[fid] = {'ok': False, 'metrics': None}
        continue
    rho, rho_id, per_id = spearman_rho_vs_library(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ic_ok = abs(m['ic']) >= 0.007
    icir_ok = abs(m['icir']) >= 0.084
    corr_ok = rho < 0.5
    ok = ic_ok and icir_ok and corr_ok
    results[fid] = {'ok': ok, 'metrics': m}
    print(f'Factor {fid}: panel {panel.shape} [{time.time()-t1:.1f}s]', flush=True)
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=1, default=str), flush=True)
    print('decay:', json.dumps(m['decay_ic_by_horizon'], default=str), flush=True)
    print(f"top rho: { {k: round(v,3) for k,v in sorted(per_id.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]} }", flush=True)
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {ic_ok} | |ICIR|={abs(m['icir']):.4f}>=0.084 {icir_ok} | corr={rho:.3f}<0.5 {corr_ok} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print('-' * 80, flush=True)

out = Path(f'scripts/miner_2_20261105_results_round25.json')
out.write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')
print(f'elapsed={time.time()-t0:.1f}s results -> {out}', flush=True)
