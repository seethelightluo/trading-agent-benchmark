"""Batch A (2026-11-05): volume/flow + time-series dynamics factor screen.

Validation on warm-up 2020-01-01..2026-07-15 (canonical grid, matches library).
Gate: |IC10| >= 0.007, |ICIR10| >= 0.084, max_abs_library_correlation < 0.5
vs real signal artifacts of EFFECTIVE library factors.
Supplementary: recent online-window IC (2026-07-16..2026-11-04) reported separately.
"""
import sys, json, glob, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, VAL_START, VAL_END, forward_returns)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
T, N = len(grid), len(WATCHLIST)
val_mask = (grid >= VAL_START) & (grid <= VAL_END)
print(f"grid {T} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)} | {time.time()-t0:.1f}s", flush=True)
print(f"full data end: {max(d.index.max() for d in prices.values()).date()}", flush=True)

# ---------------- library rank matrices from real signal artifacts ----------------
lib_raw = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        fid = d['factor_id']
        art = d.get('signal_artifact')
        if not art:
            print(f"  no artifact: {fid}", flush=True)
            continue
        arr = np.load('factors/' + art, allow_pickle=False)
        if arr.shape == (T, N):
            lib_raw[fid] = arr
    except Exception as e:
        print(f"  artifact ERR {f}: {e}", flush=True)
print(f"library artifacts: {len(lib_raw)} -> {sorted(lib_raw.keys())}", flush=True)

def rank_matrix(df):
    return df.rank(axis=1).values.astype(float)

def spearman_from_ranks(xr, yr):
    valid = np.isfinite(xr) & np.isfinite(yr)
    nv = valid.sum(axis=1)
    ok = nv >= 8
    out = np.full(T, np.nan)
    xc = np.where(valid, xr, 0.0); yc = np.where(valid, yr, 0.0)
    xc -= xc.sum(axis=1, keepdims=True) / np.maximum(nv, 1)[:, None]
    yc -= yc.sum(axis=1, keepdims=True) / np.maximum(nv, 1)[:, None]
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    out[ok] = num[ok] / den[ok]
    return out

# sanity check
rng = np.random.default_rng(42)
xa = rng.normal(size=(50, 15)); xa[rng.random((50, 15)) < 0.2] = np.nan
ya = rng.normal(size=(50, 15)); ya[rng.random((50, 15)) < 0.2] = np.nan
xdf = pd.DataFrame(xa); ydf = pd.DataFrame(ya)
fast = spearman_from_ranks(rank_matrix(xdf), rank_matrix(ydf))
ref = np.array([xdf.iloc[i].corr(ydf.iloc[i], method='spearman') for i in range(50)])
print(f"SPEARMAN SANITY: max|fast-pandas|={np.nanmax(np.abs(fast-ref)):.2e} -> {'OK' if np.nanmax(np.abs(fast-ref))<1e-10 else 'MISMATCH'}", flush=True)

fwd_raw = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}
fwd_rank = {h: rank_matrix(fwd_raw[h].reindex(grid)) for h in fwd_raw}
lib_rank = {fid: rank_matrix(pd.DataFrame(arr, index=grid, columns=WATCHLIST)) for fid, arr in lib_raw.items()}

# ---------------- candidate factors ----------------
def f_volume_trend_20_60(df, s):
    v = df['volume']
    return v.rolling(20).mean() / v.rolling(60).mean() - 1.0

def f_volume_concentration_20(df, s):
    v = df['volume']
    sv = v.rolling(20).sum()
    conc = (v ** 2).rolling(20).sum() / sv ** 2
    return conc

def f_max_volume_day_20(df, s):
    v = df['volume']
    return v.rolling(20).max() / v.rolling(20).mean()

def f_vwp_mom_60_20(df, s):
    r = df['close'].pct_change()
    v = df['volume']
    num = (r * v).rolling(60).sum()
    den = v.rolling(60).sum()
    return num / den.replace(0, np.nan)

def f_vol_volume_corr_20(df, s):
    v = df['volume']
    ar = df['close'].pct_change().abs()
    z = pd.concat([v.rename('v'), ar.rename('a')], axis=1)
    return z['v'].rolling(20).corr(z['a'])

def f_variance_ratio_20_60(df, s):
    r = df['close'].pct_change()
    s20 = r.rolling(20).sum()
    vr = s20.rolling(60).var() / (20 * r.rolling(60).var())
    return vr

def f_autocorr_ret_20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if np.std(x) > 0 else np.nan, raw=True)

def f_zscore_120(df, s):
    c = df['close']
    mu = c.rolling(120).mean()
    sd = c.rolling(120).std()
    return (c - mu) / sd.replace(0, np.nan)

def f_below_high_frac_60(df, s):
    c = df['close']
    hh = c.rolling(60).max()
    return (c < hh).astype(float)

def f_max_dd_60(df, s):
    c = df['close']
    hh = c.rolling(60).max()
    return c / hh - 1.0

cands = {
    'volume_trend_20_60': (f_volume_trend_20_60, '20d/60d mean volume ratio - 1', 'volume expansion/contraction'),
    'volume_concentration_20': (f_volume_concentration_20, 'Herfindahl of daily volume shares over 20d', 'volume flow concentration'),
    'max_volume_day_20': (f_max_volume_day_20, 'max daily volume / 20d mean volume', 'volume spike intensity'),
    'vwp_mom_60_20': (f_vwp_mom_60_20, 'volume-weighted 60d momentum (sum r*v / sum v)', 'volume-confirmed momentum'),
    'vol_volume_corr_20': (f_vol_volume_corr_20, 'rolling 20d corr(volume, |ret|)', 'volume-activity coupling'),
    'variance_ratio_20_60': (f_variance_ratio_20_60, 'Var(20d sum)/(20*Var(1d)) over 60d', 'trend vs mean-reversion (VR)'),
    'autocorr_ret_20': (f_autocorr_ret_20, 'lag-1 autocorrelation of daily returns over 20d', 'return autocorrelation'),
    'zscore_120': (f_zscore_120, '(close - MA120)/std120', '120d z-score / distance from mean'),
    'below_high_frac_60': (f_below_high_frac_60, 'fraction of days below rolling 60d high', 'drawdown time persistence'),
    'max_dd_60': (f_max_dd_60, 'close/60d max - 1 (drawdown depth)', 'drawdown depth'),
}

def factor_to_panel(fn, prices):
    cols = {}
    for s, df in prices.items():
        try:
            ser = fn(df, s)
            if ser is not None and len(ser) > 0:
                cols[s] = ser.astype(float)
        except Exception as e:
            print(f"    {s} ERR {e}", flush=True)
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols)
    return panel[~panel.index.duplicated(keep='last')].sort_index()

def fast_validate(fid, panel):
    pm = signal_matrix(panel, grid)
    pr = rank_matrix(pd.DataFrame(pm, index=grid, columns=WATCHLIST))
    ic10 = spearman_from_ranks(pr, fwd_rank[10])[val_mask]
    if np.isfinite(ic10).sum() < 100:
        return None
    ic_mean = float(np.nanmean(ic10))
    ic_std = float(np.nanstd(ic10, ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())
    fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    total_cells = fac.shape[0] * fac.shape[1]
    coverage = float(fac.notna().sum().sum()) / total_cells if total_cells else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    decay = {}
    for h in (1, 2, 3, 5, 10, 20):
        s = spearman_from_ranks(pr, fwd_rank[h])[val_mask]
        decay[str(h)] = float(np.nanmean(s))
    m = {'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit,
         'n_ic_dates': int(np.isfinite(ic10).sum()), 'coverage_asset_days': coverage,
         'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn,
         'decay_ic_by_horizon': decay}
    gv = grid[val_mask]
    for nm, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                     ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                     ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
        sub = ic10[(gv >= pd.Timestamp(a)) & (gv <= pd.Timestamp(b))]
        m[nm] = float(np.nanmean(sub)) if np.isfinite(sub).sum() > 30 else float('nan')
    recent = ic10[(gv >= pd.Timestamp('2025-07-15')) & (gv <= pd.Timestamp('2026-07-15'))]
    if np.isfinite(recent).sum() > 30:
        m['recent_1y_ic'] = float(np.nanmean(recent))
        sd = float(np.nanstd(recent, ddof=1))
        m['recent_1y_icir'] = float(np.nanmean(recent) / sd) if sd > 0 else 0.0
    return m

def max_lib_corr(panel):
    pm = signal_matrix(panel, grid)
    pr = rank_matrix(pd.DataFrame(pm, index=grid, columns=WATCHLIST))
    best, best_id, per_id = 0.0, None, {}
    for fid, lr in lib_rank.items():
        r = spearman_from_ranks(pr, lr)
        rr = r[np.isfinite(r)]
        if len(rr) > 0:
            per_id[fid] = float(np.mean(rr))
            if abs(per_id[fid]) > best:
                best, best_id = abs(per_id[fid]), fid
    return best, best_id, per_id

results = {}
for fid, (fn, desc, tag) in cands.items():
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
        if panel is None or len(panel) == 0:
            print(f"{fid}: EMPTY panel -> skip", flush=True)
            continue
        m = fast_validate(fid, panel)
        if m is None:
            print(f"{fid}: insufficient data -> None", flush=True)
            continue
        rho, rho_id, per_id = max_lib_corr(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rho_id
        m['per_factor_rho'] = {k: round(v, 3) for k, v in sorted(per_id.items(), key=lambda kv: -abs(kv[1]))[:4]}
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
        results[fid] = {'ok': bool(ok), 'metrics': {k: v for k, v in m.items()}, 'desc': desc, 'tag': tag}
        dec = {h: round(v, 4) for h, v in m['decay_ic_by_horizon'].items()}
        print(f"\n{fid}: IC10={m['ic']:.4f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
              f"cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rho_id}) [{time.time()-t1:.1f}s]", flush=True)
        print(f"  decay: {dec}", flush=True)
        print(f"  top rho: {m['per_factor_rho']}", flush=True)
        for nm in ['ic_2020_2022', 'ic_2023_2024', 'ic_2025_2026']:
            print(f"  {nm}: {m.get(nm, float('nan')):.4f}", flush=True)
        if 'recent_1y_ic' in m:
            print(f"  recent_1y: ic={m['recent_1y_ic']:.4f} icir={m['recent_1y_icir']:.4f}", flush=True)
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007 |ICIR|={abs(m['icir']):.4f}/0.084 rho={rho:.3f}/0.5)", flush=True)
    except Exception as e:
        print(f"{fid}: EXCEPTION {e}", flush=True)
        results[fid] = {'ok': False, 'error': str(e), 'desc': desc, 'tag': tag}

with open('scripts/miner_1_20261105_results_batchA.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)

print("\n=== SUMMARY ===")
for fid, r in sorted(results.items()):
    if 'metrics' in r:
        m = r['metrics']
        print(f"{fid:24s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m.get('max_abs_library_correlation', float('nan')):.3f} ({m.get('max_corr_library_id')})")
    else:
        print(f"{fid:24s} ERROR {r.get('error', '')[:80]}")
print(f"total time {time.time()-t0:.1f}s")
