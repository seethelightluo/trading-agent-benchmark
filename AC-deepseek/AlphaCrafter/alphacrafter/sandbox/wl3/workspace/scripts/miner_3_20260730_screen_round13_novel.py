"""Round 13: fresh novel factor candidates for the 16-factor library.

Ideas (single-idea constructs, none previously screened in this repo):
 1. hurst_vr_100       - variance-ratio Hurst proxy (long-memory trend persistence)
 2. entropy_sign_20    - 1 - normalized Shannon entropy of return signs (directional balance)
 3. kurt_20            - excess kurtosis of daily returns (tail risk)
 4. autocorr_ret_20    - lag-1 autocorrelation of daily returns (magnitude-weighted persistence)
 5. up_capture_60      - upside beta vs equal-weight market (positive market days only)
 6. conviction_20      - |20d return| / 20d realized vol (trend signal-to-noise)
 7. gap_freq_20        - frequency of large overnight gaps over 20d (gap risk activity)
 8. rs_vs_gold_20      - 20d momentum minus XAU 20d momentum (risk-on/off relative strength)
 9. herfindahl_ret_20  - return concentration: HHI of |daily ret| shares over 20d
10. downside_dev_ratio_20_60 - short/long downside semi-deviation ratio (tail acceleration)
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, factor_to_panel, validate_factor,
                           forward_returns, rank_ic_series)

np.seterr(all='ignore')

prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices loaded: {len(prices)} assets; canonical grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})")

# equal-weight market return panel (for upside-capture)
ret_panel = pd.DataFrame({s: prices[s]['close'].pct_change() for s in WATCHLIST})
mkt_ret = ret_panel.mean(axis=1)
mkt_ret.name = 'MKT'

# ---------- library artifacts (real signal matrices) ----------
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
        arr_use = arr if la.shape[0] >= arr.shape[0] else arr[-la.shape[0]:]
        n = min(arr_use.shape[0], la.shape[0])
        if n < 60:
            continue
        x, y = arr_use[-n:], la[-n:]
        corrs = []
        for i in range(n):
            a, b = x[i], y[i]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= 8:
                r = pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())
                if np.isfinite(r):
                    corrs.append(r)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


# ---------- candidate factor functions ----------
def hurst_vr_100(df, s):
    """Variance-ratio Hurst: H = 0.5*ln(VR(10))/ln(10) + 0.5 over 100d window."""
    r = df['close'].pct_change()
    w = 100
    v1 = r.rolling(w).var()
    r10 = r.rolling(10).sum()
    v10 = r10.rolling(w).var()
    vr = v10 / (10.0 * v1.replace(0, np.nan))
    h = 0.5 * np.log(vr) / np.log(10.0) + 0.5
    return h.clip(0.0, 1.0)


def entropy_sign_20(df, s):
    pos = (df['close'].pct_change() > 0).astype(float)
    n_up = pos.rolling(20).sum()
    p = n_up / 20.0
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0)
    return (1.0 - ent).clip(0.0, 1.0)


def kurt_20(df, s):
    return df['close'].pct_change().rolling(20).kurt()


def autocorr_ret_20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).corr(r.shift(1))


def up_capture_60(df, s):
    r = df['close'].pct_change()
    m = mkt_ret.reindex(df.index)
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1)
    up = z[z['m'] > 0]
    cov = up['r'].rolling(60).cov(up['m'])
    var = up['m'].rolling(60).var()
    return (cov / var.replace(0, np.nan)).reindex(z.index)


def conviction_20(df, s):
    r = df['close'].pct_change()
    mom20 = df['close'] / df['close'].shift(20) - 1.0
    vol20 = r.rolling(20).std() * np.sqrt(20)
    return (mom20 / vol20.replace(0, np.nan)).abs()


def gap_freq_20(df, s):
    pc = df['close'].shift(1)
    gap = df['open'] / pc - 1.0
    tr = pd.concat([(df['high'] - df['low']),
                    (df['high'] - pc).abs(),
                    (df['low'] - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    big = (gap.abs() > 0.5 * atr14).astype(float)
    return big.rolling(20).mean()


def rs_vs_gold_20(df, s):
    gold = prices['XAU']['close'].reindex(df.index)
    a = df['close'].shift(5) / df['close'].shift(25) - 1.0
    g = gold.shift(5) / gold.shift(25) - 1.0
    return a - g


def herfindahl_ret_20(df, s):
    r = df['close'].pct_change().abs()
    w = r.rolling(20).sum()
    share = r / w.replace(0, np.nan)
    return (share ** 2).rolling(20).mean()


def downside_dev_ratio_20_60(df, s):
    r = df['close'].pct_change()
    neg = r.clip(upper=0.0)
    dd20 = (neg ** 2).rolling(20).mean().apply(np.sqrt)
    dd60 = (neg ** 2).rolling(60).mean().apply(np.sqrt)
    return dd20 / dd60.replace(0, np.nan)


candidates = {
    'hurst_vr_100': hurst_vr_100,
    'entropy_sign_20': entropy_sign_20,
    'kurt_20': kurt_20,
    'autocorr_ret_20': autocorr_ret_20,
    'up_capture_60': up_capture_60,
    'conviction_20': conviction_20,
    'gap_freq_20': gap_freq_20,
    'rs_vs_gold_20': rs_vs_gold_20,
    'herfindahl_ret_20': herfindahl_ret_20,
    'downside_dev_ratio_20_60': downside_dev_ratio_20_60,
}

results = {}
for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: INSUFFICIENT -> skip", flush=True)
        results[fid] = {'ok': False, 'metrics': {'error': 'insufficient'}}
        continue
    rho, lib_id = max_lib_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = lib_id
    # recent 1y IC (timeliness / drift)
    ic_s = rank_ic_series(panel, forward_returns(prices, 10))
    ic_s = ic_s[(ic_s.index >= pd.Timestamp('2025-07-15')) & (ic_s.index <= pd.Timestamp('2026-07-15'))]
    if len(ic_s) > 30:
        m['recent_1y_ic'] = float(ic_s.mean())
        m['recent_1y_icir'] = float(ic_s.mean() / ic_s.std(ddof=1)) if ic_s.std(ddof=1) > 0 else 0.0
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = {'ok': ok, 'metrics': m}
    print(f"{fid}: IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
          f"ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} "
          f"rho={rho:.3f}({lib_id}) 1yIC={m.get('recent_1y_ic', float('nan')):+.4f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print("   decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()}, flush=True)

json.dump(results, open('scripts/miner_3_20260730_results_round13.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_3_20260730_results_round13.json")
