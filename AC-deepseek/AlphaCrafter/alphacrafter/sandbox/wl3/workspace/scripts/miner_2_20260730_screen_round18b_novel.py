"""Round 18b novel factor screen (miner_2). Beta/conditional family on fresh anchors:
US10Y, USDCNY, XAU-up days, risk-off spread (US10Y-SPX), SPX up/down beta gap,
NDX, plus volume up/down asymmetry. Library = all EFFECTIVE factors with artifacts."""
import sys, time, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, load_index, factor_to_panel,
                           validate_factor, canonical_grid, signal_matrix,
                           WATCHLIST)

t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

lib = {}
for p in sorted(Path('factors').glob('*.json')):
    if p.name.endswith('.bak') or 'deprecated' in p.name:
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
print(f'library factors: {len(lib)} {sorted(lib.keys())}', flush=True)

def lib_max_corr(panel):
    arr = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, larr in lib.items():
        corrs = []
        for i in range(arr.shape[0]):
            x, y = arr[i], larr[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xr = pd.Series(x[m]).rank().values
                yr = pd.Series(y[m]).rank().values
                c = np.corrcoef(xr, yr)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

ret_panel = pd.DataFrame({s: df['close'].pct_change() for s, df in prices.items()}).sort_index()
spx = ret_panel['SPX'] if 'SPX' in ret_panel else None
ndx = ret_panel['NDX'] if 'NDX' in ret_panel else None
xau = ret_panel['XAU'] if 'XAU' in ret_panel else None
us10y = ret_panel['US10Y'] if 'US10Y' in ret_panel else None

usdcny = load_index('USDCNY', prices=prices)
usdcny_ret = usdcny['close'].pct_change() if usdcny is not None else None

def beta_series(r, anchor, window=60, min_periods=30, cond_mask=None):
    z = pd.concat([r.rename('r'), anchor.rename('a')], axis=1).dropna()
    if cond_mask is not None:
        cm = cond_mask.reindex(z.index).fillna(False)
        z = z[cm]
        if len(z) < 10:
            return pd.Series(index=r.index, dtype=float)
    b = z['r'].rolling(window, min_periods=min_periods).cov(z['a']) / \
        z['a'].rolling(window, min_periods=min_periods).var()
    return b.reindex(r.index)

def f_us10y_beta_60(df, s):
    if us10y is None:
        return None
    return beta_series(df['close'].pct_change(), us10y, 60, 40)

def f_usdcny_beta_60(df, s):
    if usdcny_ret is None:
        return None
    return beta_series(df['close'].pct_change(), usdcny_ret, 60, 40)

def f_xau_upbeta_60(df, s):
    if xau is None:
        return None
    return beta_series(df['close'].pct_change(), xau, 60, 40, cond_mask=(xau > 0))

def f_risk_off_beta_20(df, s):
    if us10y is None or spx is None:
        return None
    spread = us10y - spx
    return beta_series(df['close'].pct_change(), spread, 20, 15)

def f_spx_beta_gap_60(df, s):
    if spx is None:
        return None
    r = df['close'].pct_change()
    up = beta_series(r, spx, 60, 40, cond_mask=(spx > 0))
    dn = beta_series(r, spx, 60, 40, cond_mask=(spx < 0))
    return up - dn

def f_vol_updown_asym_20(df, s):
    r = df['close'].pct_change()
    v = df['volume'].astype(float)
    up_v = v.where(r > 0).rolling(20, min_periods=6).mean()
    dn_v = v.where(r < 0).rolling(20, min_periods=6).mean()
    return up_v / (dn_v + 1e-12)

def f_ndx_beta_60(df, s):
    if ndx is None:
        return None
    return beta_series(df['close'].pct_change(), ndx, 60, 40)

CANDIDATES = [
    ('us10y_beta_60', f_us10y_beta_60),
    ('usdcny_beta_60', f_usdcny_beta_60),
    ('xau_upbeta_60', f_xau_upbeta_60),
    ('risk_off_beta_20', f_risk_off_beta_20),
    ('spx_beta_gap_60', f_spx_beta_gap_60),
    ('vol_updown_asym_20', f_vol_updown_asym_20),
    ('ndx_beta_60', f_ndx_beta_60),
]

results = {}
for fid, fn in CANDIDATES:
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data -> None', flush=True)
        results[fid] = {'ok': False, 'metrics': None}
        continue
    rho, rho_id = lib_max_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ic_ok = abs(m['ic']) >= 0.007
    icir_ok = abs(m['icir']) >= 0.084
    corr_ok = rho < 0.5
    ok = ic_ok and icir_ok and corr_ok
    results[fid] = {'ok': ok, 'metrics': m}
    print(f'Factor {fid}: panel {panel.shape} range {panel.index.min()}..{panel.index.max()}', flush=True)
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=2, default=str), flush=True)
    print('decay:', json.dumps(m['decay_ic_by_horizon'], default=str), flush=True)
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {ic_ok} | |ICIR|={abs(m['icir']):.4f}>=0.084 {icir_ok} | corr={rho:.3f}<0.5 {corr_ok} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print('-' * 80, flush=True)

out = Path('scripts/miner_2_20260730_results_round18b.json')
out.write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')
print(f'elapsed={time.time()-t0:.1f}s results -> {out}', flush=True)
