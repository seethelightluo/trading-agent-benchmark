"""Round 18 novel factor screen (miner_2). Fresh ideas not previously persisted:
overnight-return share, volume expansion trend, return autocorrelation (reversal
speed), return kurtosis (tail thickness), short/long vol ratio, wick asymmetry,
up/down volatility asymmetry. Validates vs the full EFFECTIVE library (now 12)."""
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

# ---- candidate factor functions (each takes df, symbol -> pd.Series) ----
def f_overnight_share_20(df, s):
    prev_close = df['close'].shift(1)
    over = (df['open'] / prev_close - 1.0).abs()
    intra = (df['close'] / df['open'] - 1.0).abs()
    a = over.rolling(20, min_periods=10).mean()
    b = intra.rolling(20, min_periods=10).mean()
    return a / (a + b + 1e-12)

def f_vol_trend_5_60(df, s):
    v = df['volume'].astype(float)
    return v.rolling(5, min_periods=3).mean() / v.rolling(60, min_periods=20).mean() - 1.0

def f_ret_autocorr_20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20, min_periods=15).apply(lambda x: x.autocorr() if len(x) > 2 else np.nan, raw=False)

def f_kurt_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=40).kurt()

def f_vol_ratio_5_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(5, min_periods=3).std() / r.rolling(60, min_periods=20).std() - 1.0

def f_wick_ratio_20(df, s):
    up = df['high'] - pd.concat([df['open'], df['close']], axis=1).max(axis=1)
    dn = pd.concat([df['open'], df['close']], axis=1).min(axis=1) - df['low']
    su = up.rolling(20, min_periods=10).sum()
    sd = dn.rolling(20, min_periods=10).sum()
    return su / (sd + 1e-12)

def f_updown_vol_asym_20(df, s):
    r = df['close'].pct_change()
    up = r.clip(lower=0)
    dn = (-r.clip(upper=0))
    su = up.rolling(20, min_periods=10).std()
    sd = dn.rolling(20, min_periods=10).std()
    return su / (sd + 1e-12)

CANDIDATES = [
    ('overnight_share_20', f_overnight_share_20),
    ('vol_trend_5_60', f_vol_trend_5_60),
    ('ret_autocorr_20', f_ret_autocorr_20),
    ('kurt_60', f_kurt_60),
    ('vol_ratio_5_60', f_vol_ratio_5_60),
    ('wick_ratio_20', f_wick_ratio_20),
    ('updown_vol_asym_20', f_updown_vol_asym_20),
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

out = Path('scripts/miner_2_20260730_results_round18.json')
out.write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')
print(f'elapsed={time.time()-t0:.1f}s results -> {out}', flush=True)
