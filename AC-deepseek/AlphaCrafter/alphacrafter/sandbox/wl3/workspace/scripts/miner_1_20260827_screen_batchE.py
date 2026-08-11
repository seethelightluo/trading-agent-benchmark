"""miner_1 2026-08-27 exploration batch E: final conditional-beta sweep on
untried drivers + overnight/intraday risk-share factor.

The conditional-beta family (beta x driver move) is the most productive vein
(vix/dxy/eurusd passed with rho<0.1). Untried drivers: BTC, ETH, SOX, COPPER,
N225. Plus a novel risk-allocation factor: share of total vol coming from
intraday (open->close) vs overnight (prev close->open) moves.

Admission: |IC|>=0.007, |ICIR|>=0.084, max full-lib rho < 0.5.
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           WATCHLIST, canonical_grid, signal_matrix)

prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"loaded {len(prices)} assets; canonical grid {len(grid)} dates")

def drv_ret(name):
    return prices[name]['close'].pct_change().rename(name)

DRIVERS = {d: drv_ret(d) for d in ['BTC', 'ETH', 'SOX', 'COPPER', 'N225']}

def make_cond_beta(dname):
    dr = DRIVERS[dname]
    def f(df, s):
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), dr.rename('y')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()
        y_move = prices[dname]['close'] / prices[dname]['close'].shift(20) - 1.0
        return (b * y_move).reindex(z.index)
    f.__name__ = f'f_{dname}_cond'
    return f

def f_intraday_vol_share_20(df, s):
    """Share of total (open->close + overnight) vol coming from intraday moves.
    High share => risk concentrated in session trading; low => gap risk."""
    if 'open' not in df.columns:
        return None
    c = df['close']; o = df['open']
    intra = (c / o - 1.0)
    overn = (o / c.shift(1) - 1.0)
    iv = intra.rolling(20, min_periods=10).std()
    ov = overn.rolling(20, min_periods=10).std()
    return iv / (iv + ov + 1e-12)

# ---- evaluate ----
candidates = []
for d in ['BTC', 'ETH', 'SOX', 'COPPER', 'N225']:
    candidates.append((f'{d.lower()}_beta_cond_60x20', make_cond_beta(d)))
candidates.append(('intraday_vol_share_20', f_intraday_vol_share_20))

lib_artifacts = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    lib_artifacts[p.name.replace('_signal.npy', '')] = np.load(p, allow_pickle=False)
print(f"loaded {len(lib_artifacts)} library artifacts for rho audit")

def full_lib_rho(cand_arr, lib_artifacts):
    out = {}
    for fid, arr in lib_artifacts.items():
        if arr.shape != cand_arr.shape:
            continue
        corrs = []
        for i in range(cand_arr.shape[0]):
            x = cand_arr[i]; y = arr[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xs = pd.Series(x[m]).rank().values
                ys = pd.Series(y[m]).rank().values
                c = np.corrcoef(xs, ys)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            out[fid] = float(np.mean(corrs))
    return out

results = {}
for fid, fn in candidates:
    try:
        panel = factor_to_panel(fn, prices)
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid}: insufficient -> None")
            results[fid] = {'ok': False, 'metrics': {'error': 'insufficient'}}
            continue
        cand_arr = signal_matrix(panel, grid)
        rhos = full_lib_rho(cand_arr, lib_artifacts)
        rho_max = max(abs(v) for v in rhos.values()) if rhos else 0.0
        rho_max_id = max(rhos, key=lambda k: abs(rhos[k])) if rhos else None
        top = sorted(rhos.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho_max < 0.5
        print(f"=== {fid}: panel {panel.shape} | IC={m['ic']:.4f} ICIR={m['icir']:.4f} "
              f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
              f"ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.3f}")
        print(f"    max full-lib rho={rho_max:.3f} ({rho_max_id}) -> {'PASS' if ok else 'FAIL'}")
        for k, v in top:
            print(f"      rho({k})={v:.3f}")
        print("    decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()})
        m['max_abs_library_correlation'] = rho_max
        m['max_corr_library_id'] = rho_max_id
        m['full_lib_rho_top4'] = {k: round(v, 4) for k, v in top}
        results[fid] = {'ok': ok, 'metrics': m}
    except Exception as e:
        print(f"{fid}: ERROR {e}")
        results[fid] = {'ok': False, 'metrics': {'error': str(e)}}

with open('scripts/miner_1_20260827_results_batchE.json', 'w') as f:
    json.dump(results, f, default=str, indent=1)
print("\nDONE saved scripts/miner_1_20260827_results_batchE.json")
