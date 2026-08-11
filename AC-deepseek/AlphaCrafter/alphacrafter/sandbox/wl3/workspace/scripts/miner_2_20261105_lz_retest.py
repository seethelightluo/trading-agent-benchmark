"""miner_2 2026-11-05: re-test lz_sign_complexity_60 with fixed LZ76 implementation.

The round25 screen hit a numpy broadcast ValueError inside _lz_complexity for
short rolling windows (sub in bits[i+k:i+k+k] when i+2k > n). This script uses a
bounded, element-wise comparison LZ76 and re-runs the full validation battery.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, WATCHLIST)

TODAY = '2026-11-05'
t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

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
print(f'library factors with artifacts: {len(lib)}', flush=True)


def rank_matrix(arr):
    out = np.full(arr.shape, np.nan)
    for t in range(arr.shape[0]):
        row = arr[t]
        m = np.isfinite(row)
        n = int(m.sum())
        if n >= 8:
            r = np.full(arr.shape[1], np.nan)
            r[m] = pd.Series(row[m]).rank().values
            out[t] = r
    return out


lib_rank = {fid: rank_matrix(arr) for fid, arr in lib.items()}


def lib_max_corr_fast(panel):
    arr = signal_matrix(panel, grid)
    cand_rank = rank_matrix(arr)
    best, best_id = 0.0, None
    for fid, lr in lib_rank.items():
        cs = np.full(cand_rank.shape[0], np.nan)
        for t in range(cand_rank.shape[0]):
            a, b = cand_rank[t], lr[t]
            m = np.isfinite(a) & np.isfinite(b)
            n = int(m.sum())
            if n >= 8:
                a2, b2 = a[m], b[m]
                ma, mb = a2.mean(), b2.mean()
                sa, sb = a2.std(ddof=1), b2.std(ddof=1)
                if sa > 1e-12 and sb > 1e-12:
                    cs[t] = ((a2 - ma) * (b2 - mb)).mean() / (sa * sb)
        if np.isfinite(cs).any():
            r = float(np.nanmean(cs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


def _lz_complexity(bits):
    """LZ76 complexity of a binary sequence, normalized by n/log2(n). Bounded search."""
    n = len(bits)
    if n < 2:
        return np.nan
    c, i, k = 1, 0, 1
    while True:
        if i + k >= n:
            c += 1
            break
        sub = bits[i:i + k]
        hi = min(i + 2 * k, n) - k          # last valid start for a length-k window
        found = False
        for j in range(i + k, hi + 1):
            if np.array_equal(sub, bits[j:j + k]):
                found = True
                break
        if found:
            k += 1
        else:
            c += 1
            i += k
            k = 1
        if i >= n:
            break
    norm = n / np.log2(n) if n > 1 else 1.0
    return c / norm


def f_lz_sign_complexity_60(df, s):
    r = df['close'].pct_change()
    def fn(x):
        bits = (x > 0).astype(int)
        return _lz_complexity(bits)
    return r.rolling(60, min_periods=30).apply(fn, raw=True)


fid = 'lz_sign_complexity_60'
panel = factor_to_panel(f_lz_sign_complexity_60, prices)
print(f'panel {panel.shape}', flush=True)
if panel.shape[0] < 100 or panel.shape[1] < 8:
    print('panel too small -> skip', flush=True)
    sys.exit(0)
m = validate_factor(fid, panel, prices)
if m is None:
    print('insufficient data -> None', flush=True)
    sys.exit(0)
rho, rho_id = lib_max_corr_fast(panel)
m['max_abs_library_correlation'] = rho
m['max_corr_library_id'] = rho_id
ic_ok = abs(m['ic']) >= 0.007
icir_ok = abs(m['icir']) >= 0.084
corr_ok = rho < 0.5
ok = ic_ok and icir_ok and corr_ok
print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=1, default=str), flush=True)
print('decay:', json.dumps(m['decay_ic_by_horizon'], default=str), flush=True)
print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {ic_ok} | |ICIR|={abs(m['icir']):.4f}>=0.084 {icir_ok} | corr={rho:.3f}<0.5 {corr_ok} -> {'PASS' if ok else 'FAIL'}", flush=True)
print(f'elapsed={time.time()-t0:.1f}s', flush=True)
