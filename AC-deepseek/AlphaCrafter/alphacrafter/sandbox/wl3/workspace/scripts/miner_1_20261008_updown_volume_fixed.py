"""miner_1 2026-10-08: re-test volume-price asymmetry factor with fixed rolling
min_periods.

Crash root cause (found earlier): up.rolling(w).mean() uses default
min_periods=w; a 20d window never contains 20 up-days (or down-days), so the
whole panel was NaN -> empty IC series -> crash.

Fix: min_periods = max(4, w//5) so the up/down volume averages are estimated from
a reasonable subset of days. Also 6/15 assets (SOX, XAU, COPPER, WTI, US10Y,
CN10Y) have all-zero volume -> factor is NaN there by construction (report
coverage explicitly).

Idea: average volume on up-days / average volume on down-days over trailing w
days. High values = buying pressure dominates on up-moves (volume confirmation
of advances); low values = heavy volume on declines (distribution / fear).
Cross-sectional IC vs 10d forward return.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid, factor_to_panel,
                           validate_factor, signal_matrix, VAL_START, VAL_END,
                           forward_returns, rank_ic_series)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)}", flush=True)

# volume availability audit
vol_assets = []
for s in WATCHLIST:
    df = prices[s]
    has_vol = 'volume' in df.columns and df['volume'].notna().sum() > 0 and (df['volume'] > 0).any()
    vol_assets.append(s) if has_vol else None
    nz = int((df['volume'] > 0).sum()) if 'volume' in df.columns else 0
    print(f"  {s:12s} volume_positive_days={nz}", flush=True)
print(f"assets with real volume: {len(vol_assets)} -> {vol_assets}", flush=True)


def make_updown_volume(w, mp=None, thresh=0.0):
    """avg volume up-days / avg volume down-days over trailing w days."""
    if mp is None:
        mp = max(4, w // 5)
    def f(df, s):
        r = df['close'].pct_change()
        v = df['volume'].replace(0, np.nan)
        up = v.where(r > thresh)
        dn = v.where(r < -thresh)
        upm = up.rolling(w, min_periods=mp).mean()
        dnm = dn.rolling(w, min_periods=mp).mean()
        return upm / dnm.replace(0, np.nan)
    return f


def max_lib_corr_from_artifacts(panel, grid):
    """rho vs the 18 effective library artifacts (same method as batch A)."""
    best, best_id = 0.0, None
    pm = signal_matrix(panel, grid)
    for fid, lp in lib_panels.items():
        lm = lp.values
        corrs = []
        for t in range(len(grid)):
            x = pm[t]; y = lm[t]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xv = x[m]; yv = y[m]
                xc = xv - xv.mean(); yc = yv - yv.mean()
                den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
                if den > 0:
                    corrs.append((xc * yc).sum() / den)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


# library artifact panels (18 effective)
import glob
lib_panels = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        art = d.get('signal_artifact')
        if art:
            arr = np.load('factors/' + art, allow_pickle=False)
            if arr.shape == (len(grid), len(WATCHLIST)):
                lib_panels[d['factor_id']] = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
    except Exception as e:
        print(f"  artifact ERR {f}: {e}", flush=True)
print(f"library panels for rho: {len(lib_panels)} -> {sorted(lib_panels.keys())}", flush=True)

fwd10 = forward_returns(prices, 10)

results = {}
for w in [20, 40, 60]:
    fid = f"updown_volume_{w}"
    panel = factor_to_panel(make_updown_volume(w), prices)
    if panel is None or len(panel) == 0:
        print(f"{fid}: EMPTY panel -> skip", flush=True)
        continue
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: insufficient data -> None", flush=True)
        continue
    rho, rho_id = max_lib_corr_from_artifacts(panel, grid)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    results[fid] = {'ok': bool(ok), 'metrics': {k: v for k, v in m.items()}}
    dec = {h: round(v, 4) for h, v in m['decay_ic_by_horizon'].items()}
    print(f"\n{fid}: IC10={m['ic']:.4f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"coverage={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rho_id})", flush=True)
    print(f"  decay: {dec}", flush=True)
    ic10 = rank_ic_series(panel, fwd10, 8)
    ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    for nm, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                     ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                     ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
        sub = ic10[(ic10.index >= a) & (ic10.index <= b)]
        print(f"  {nm}: {sub.mean():.4f} (n={len(sub)})", flush=True)
    rec = ic10[ic10.index >= '2025-07-16']
    if len(rec) > 30:
        r_ic = rec.mean(); r_icir = r_ic / rec.std(ddof=1) if rec.std(ddof=1) > 0 else 0.0
        print(f"  recent_1y: ic={r_ic:.4f} icir={r_icir:.4f} (n={len(rec)})", flush=True)
    print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007, "
          f"|ICIR|={abs(m['icir']):.4f}/0.084, rho={rho:.3f}/0.5)", flush=True)

with open('scripts/miner_1_20261008_results_updown_volume.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)
print("\n=== SUMMARY ===")
for fid, r in results.items():
    m = r['metrics']
    print(f"{fid:22s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} "
          f"rho={m['max_abs_library_correlation']:.3f} ({m.get('max_corr_library_id')}) "
          f"cov={m['coverage_asset_days']:.3f}", flush=True)
