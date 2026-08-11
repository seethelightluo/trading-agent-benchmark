"""
miner_1 2026-08-27 exploration batch F: final sweep of untested families.

Library already covers: plain betas (spx/sx5e/hs300/cn10y/comm_basket/down),
conditional betas (vix/dxy/eurusd/bad), vol-scaled momentum, skew families,
amihud level, hilo position/vol-ratio, adx, streak, sign-persist, dd-duration.

Untested in batch F:
  - up_beta_60: beta to market-EW computed on up-market days only
  - beta_asym_60: down_beta - up_beta vs market-EW (coskewness proxy)
  - xau_beta_cond_60x20 / wti_beta_cond_60x20: conditional beta, commodity drivers
  - hs300_beta_cond_60x20: conditional beta, China equity driver
  - mean_intraday_ret_20 / mean_overnight_ret_20 / io_spread_20: session drift split
  - illiq_change_60_20: amihud trend (level already in library, trend not)
  - beta_instability_60: std of 20d market beta over 60d
  - usdcny_beta_cond_60x10: USDCNY cond-beta variant (60x20 was ICIR=0.083, just
    under gate; shorter driver move may strengthen it)

Admission: |IC|>=0.007, |ICIR|>=0.084, max full-lib rho < 0.5. Library rho audit
uses ONLY currently-EFFECTIVE persisted factors (excludes evicted leftovers).
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, load_index, factor_to_panel, validate_factor,
                           WATCHLIST, canonical_grid, signal_matrix)

prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"loaded {len(prices)} assets; canonical grid {len(grid)} dates")

def ew_ret(symbols, prices):
    df = None
    for s in symbols:
        r = prices[s]['close'].pct_change().rename(s)
        df = r if df is None else pd.concat([df, r], axis=1)
    return df.mean(axis=1).rename('ew')

mkt_r = ew_ret(WATCHLIST, prices)
xau_r = prices['XAU']['close'].pct_change().rename('xau')
wti_r = prices['WTI']['close'].pct_change().rename('wti')
hs300_r = prices['000300.SH']['close'].pct_change().rename('hs300')

# ---------- candidate factor functions ----------

def f_up_beta_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), mkt_r.rename('y')], axis=1).dropna()
    z = z[z['y'] > 0]
    if len(z) < 30:
        return pd.Series(np.nan, index=df.index)
    b = (z['r'].rolling(60, min_periods=20).cov(z['y']) /
         z['y'].rolling(60, min_periods=20).var())
    return b.reindex(df.index)

def f_beta_asym_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), mkt_r.rename('y')], axis=1).dropna()
    idx = z.index
    up_b = pd.Series(np.nan, index=idx)
    dn_b = pd.Series(np.nan, index=idx)
    up = z[z['y'] > 0]
    dn = z[z['y'] < 0]
    if len(up) >= 30:
        up_b.loc[up.index] = (up['r'].rolling(60, min_periods=20).cov(up['y']) /
                              up['y'].rolling(60, min_periods=20).var())
    if len(dn) >= 30:
        dn_b.loc[dn.index] = (dn['r'].rolling(60, min_periods=20).cov(dn['y']) /
                              dn['y'].rolling(60, min_periods=20).var())
    up_b = up_b.ffill()
    dn_b = dn_b.ffill()
    return (dn_b - up_b).reindex(df.index)

def make_cond_beta(dname, dr):
    def f(df, s):
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), dr.rename('y')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()
        y_move = prices[dname]['close'] / prices[dname]['close'].shift(20) - 1.0
        return (b * y_move).reindex(z.index)
    f.__name__ = f'f_{dname}_cond'
    return f

def f_mean_intraday_ret_20(df, s):
    if 'open' not in df.columns:
        return None
    ir = df['close'] / df['open'] - 1.0
    return ir.rolling(20, min_periods=10).mean()

def f_mean_overnight_ret_20(df, s):
    if 'open' not in df.columns:
        return None
    orn = df['open'] / df['close'].shift(1) - 1.0
    return orn.rolling(20, min_periods=10).mean()

def f_io_spread_20(df, s):
    if 'open' not in df.columns:
        return None
    ir = (df['close'] / df['open'] - 1.0).rolling(20, min_periods=10).mean()
    orn = (df['open'] / df['close'].shift(1) - 1.0).rolling(20, min_periods=10).mean()
    return ir - orn

def f_illiq_change_60_20(df, s):
    v = df['volume'].replace(0, np.nan) if 'volume' in df.columns else None
    if v is None or v.isna().all():
        return None
    amihud = df['close'].pct_change().abs() / v
    a20 = amihud.rolling(20, min_periods=10).mean()
    a60 = amihud.rolling(60, min_periods=30).mean()
    return a20 / a60 - 1.0

def f_beta_instability_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), mkt_r.rename('y')], axis=1).dropna()
    b20 = z['r'].rolling(20, min_periods=15).cov(z['y']) / z['y'].rolling(20, min_periods=15).var()
    return b20.rolling(60, min_periods=30).std().reindex(z.index)

# USDCNY cond-beta with 10d driver move (60x20 was borderline)
usdcny = load_index('USDCNY', prices=prices)
usdcny_r = usdcny['close'].pct_change().rename('usdcny') if usdcny is not None else None

def f_usdcny_beta_cond_60x10(df, s):
    if usdcny_r is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), usdcny_r.rename('y')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()
    y_move = usdcny['close'] / usdcny['close'].shift(10) - 1.0
    return (b * y_move).reindex(z.index)

# ---------- evaluate ----------
candidates = [
    ('up_beta_60', f_up_beta_60),
    ('beta_asym_60', f_beta_asym_60),
    ('xau_beta_cond_60x20', make_cond_beta('XAU', xau_r)),
    ('wti_beta_cond_60x20', make_cond_beta('WTI', wti_r)),
    ('hs300_beta_cond_60x20', make_cond_beta('000300.SH', hs300_r)),
    ('mean_intraday_ret_20', f_mean_intraday_ret_20),
    ('mean_overnight_ret_20', f_mean_overnight_ret_20),
    ('io_spread_20', f_io_spread_20),
    ('illiq_change_60_20', f_illiq_change_60_20),
    ('beta_instability_60', f_beta_instability_60),
    ('usdcny_beta_cond_60x10', f_usdcny_beta_cond_60x10),
]

# library = currently-EFFECTIVE persisted factors only
lib_artifacts = {}
for p in sorted(Path('factors').glob('*.json')):
    if p.stem == 'factor_ensemble':
        continue
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if d.get('validation', {}).get('status') != 'EFFECTIVE':
        continue
    art = d.get('signal_artifact')
    ap = Path('factors') / art if art else None
    if ap is not None and ap.exists():
        lib_artifacts[p.stem] = np.load(ap, allow_pickle=False)
print(f"loaded {len(lib_artifacts)} EFFECTIVE library artifacts for rho audit")

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

with open('scripts/miner_1_20260827_results_batchF.json', 'w') as f:
    json.dump(results, f, default=str, indent=1)
print("\nDONE saved scripts/miner_1_20260827_results_batchF.json")
