"""Round 17 novel factor screen (miner_2). Fresh cross-asset ideas not previously
tested in the library: yield-curve beta, safe-haven beta, oil-up beta,
drawdown depth, downside frequency, crypto-spread beta, gain/loss asymmetry,
vol acceleration, copper-gold rotation beta, semis-leadership beta."""
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

# ---- library: only currently effective factors (JSON present, status EFFECTIVE) ----
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
us10y = ret_panel['US10Y'] if 'US10Y' in ret_panel else None
cn10y = ret_panel['CN10Y'] if 'CN10Y' in ret_panel else None
# yield curve spread = CN10Y level - US10Y level (both are yields in %)
cn_level = prices['CN10Y']['close'] if 'CN10Y' in prices else None
us_level = prices['US10Y']['close'] if 'US10Y' in prices else None
if cn_level is not None and us_level is not None:
    curve = cn_level - us_level
    d_curve = curve.diff()
else:
    d_curve = None
wti = ret_panel['WTI'] if 'WTI' in ret_panel else None
xau = ret_panel['XAU'] if 'XAU' in ret_panel else None
btc = ret_panel['BTC'] if 'BTC' in ret_panel else None
eth = ret_panel['ETH'] if 'ETH' in ret_panel else None
copper = ret_panel['COPPER'] if 'COPPER' in ret_panel else None
sox = ret_panel['SOX'] if 'SOX' in ret_panel else None
ndx = ret_panel['NDX'] if 'NDX' in ret_panel else None

def f_curve_beta_60(df, s):
    if d_curve is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), d_curve.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['c']) / z['c'].rolling(60).var()
    return b

def f_xau_beta_60(df, s):
    if xau is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), xau.rename('x')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['x']) / z['x'].rolling(60).var()
    return b

def f_wti_upbeta_60(df, s):
    if wti is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), wti.rename('w')], axis=1).dropna()
    up = z[z['w'] > 0]
    b = up['r'].rolling(60).cov(up['w']) / up['w'].rolling(60).var()
    return b.reindex(z.index)

def f_dd_depth_60(df, s):
    roll_max = df['close'].rolling(60, min_periods=20).max()
    return df['close'] / roll_max - 1.0

def f_downside_freq_60(df, s):
    r = df['close'].pct_change()
    sig = r.rolling(60).std()
    z = (r < -sig).astype(float)
    return z.rolling(60).mean()

def f_crypto_spread_beta_20(df, s):
    if btc is None or eth is None:
        return None
    spread = btc - eth
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), spread.rename('s')], axis=1).dropna()
    b = z['r'].rolling(20).cov(z['s']) / z['s'].rolling(20).var()
    return b

def f_gain_loss_asym_20(df, s):
    r = df['close'].pct_change()
    up = r.clip(lower=0).rolling(20).mean()
    dn = (-r.clip(upper=0)).rolling(20).mean()
    return up / (dn + 1e-9)

def f_vol_accel_20_60(df, s):
    r = df['close'].pct_change()
    v20 = r.rolling(20).std()
    v60 = r.rolling(60).std()
    return v60 / v20 - 1.0

def f_copper_gold_beta_20(df, s):
    if copper is None or xau is None:
        return None
    spread = copper - xau
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), spread.rename('s')], axis=1).dropna()
    b = z['r'].rolling(20).cov(z['s']) / z['s'].rolling(20).var()
    return b

def f_semis_beta_20(df, s):
    if sox is None or ndx is None:
        return None
    spread = sox - ndx
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), spread.rename('s')], axis=1).dropna()
    b = z['r'].rolling(20).cov(z['s']) / z['s'].rolling(20).var()
    return b

CANDIDATES = [
    ('curve_beta_60', f_curve_beta_60),
    ('xau_beta_60', f_xau_beta_60),
    ('wti_upbeta_60', f_wti_upbeta_60),
    ('dd_depth_60', f_dd_depth_60),
    ('downside_freq_60', f_downside_freq_60),
    ('crypto_spread_beta_20', f_crypto_spread_beta_20),
    ('gain_loss_asym_20', f_gain_loss_asym_20),
    ('vol_accel_20_60', f_vol_accel_20_60),
    ('copper_gold_beta_20', f_copper_gold_beta_20),
    ('semis_beta_20', f_semis_beta_20),
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

out = Path('scripts/miner_2_20260730_results_round17.json')
out.write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')
print(f'elapsed={time.time()-t0:.1f}s results -> {out}', flush=True)
