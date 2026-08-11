"""miner_1 2026-10-08: persist coexceed_down_60 (PASS in batch A screen) and
debug updown_volume_20 crash (volume column availability)."""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid, factor_to_panel,
                           validate_factor, signal_matrix, VAL_START, VAL_END,
                           persist_factor, build_library_panels, max_library_correlation)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)}", flush=True)

spx_ret = prices['SPX']['close'].pct_change()

# ---------------- 1) coexceed_down_60 ----------------
def make_coexceed_down(w, thresh=0.01):
    def f(df, s):
        r = df['close'].pct_change()
        joint = (r < -thresh) & (spx_ret < -thresh)
        return joint.rolling(w).mean()
    return f

panel = factor_to_panel(make_coexceed_down(60), prices)
print(f"coexceed_down_60 panel: {panel.shape} range {panel.index.min()}..{panel.index.max()}", flush=True)
m = validate_factor('coexceed_down_60', panel, prices)
lib_panels = build_library_panels(prices)
rho, fid = max_library_correlation(panel, lib_panels)
m['max_abs_library_correlation'] = rho
m['max_corr_library_id'] = fid
ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
print(f"\ncoexceed_down_60: IC10={m['ic']:.4f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
      f"coverage={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
      f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({fid})", flush=True)
print(f"decay: {json.dumps({h: round(v,4) for h, v in m['decay_ic_by_horizon'].items()})}", flush=True)
print(f"ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007, |ICIR|={abs(m['icir']):.4f}/0.084, rho={rho:.3f}/0.5)", flush=True)

if ok:
    regime_notes = ("Validated 2020-01-01..2026-07-15. Joint downside co-exceedance with SPX: "
                    "fraction of days in the trailing 60d where the asset and SPX both fell more "
                    "than 1%. Positive 10d forward IC -- assets with high co-crash frequency with "
                    "SPX (risk-on high-beta-like exposures) tend to keep outperforming, i.e. "
                    "co-exceedance acts as a regime/risk-appetite classifier. Regime ICs: see "
                    "metrics (ic_2020_2022 / ic_2023_2024 / ic_2025_2026). Decay peaks at 10d "
                    "horizon. Decorrelated from library (max rho < 0.5).")
    path, arr = persist_factor(
        factor_id='coexceed_down_60',
        factor_name='Joint downside co-exceedance with SPX (60d)',
        expression='mean( (ret < -0.01) & (spx_ret < -0.01) over trailing 60d )',
        description=("For each asset, count the fraction of trading days in the trailing 60-day "
                     "window where both the asset return and the SPX return fell below -1% "
                     "(joint downside co-exceedance). High values mark assets that crash together "
                     "with US equities -- a risk-on / high systemic-beta signature. Positive "
                     "cross-sectional IC: these co-crash-prone assets tend to outperform over the "
                     "next 10 days in the synthetic benchmark (regime/risk-appetite classifier)."),
        dependencies=['close', 'SPX.close'],
        parameters={'window': 60, 'threshold': 0.01},
        expected_direction=1,
        panel=panel,
        metrics=m,
        tags=['tail-dependence', 'co-exceedance', 'systemic-beta', 'regime', 'cross-asset'],
        grid=grid, prices=prices, version='1.0.0',
        regime_notes=regime_notes,
        extra={'signal_provenance': {
            'source': 'recomputed from alphacrafter.sim.utils daily OHLC series',
            'panel_shape': f"{panel.shape[0]}x{panel.shape[1]}",
            'panel_range': f"{panel.index.min().date()}..{panel.index.max().date()}",
            'validation_window': f"{VAL_START.date()}..{VAL_END.date()}",
            'ic_method': 'daily cross-sectional Spearman rank IC vs 10d forward return',
            'note': 'expression deterministic and reproducible from OHLC series only'}})
    print(f"PERSISTED -> {path}", flush=True)
    # read back and verify
    d = json.load(open(path))
    assert d['factor_id'] == 'coexceed_down_60', 'id mismatch'
    assert d['validation']['status'] == 'EFFECTIVE', 'status mismatch'
    assert d['validation']['metrics']['ic'] == m['ic'], 'ic mismatch'
    assert d['validation']['metrics']['icir'] == m['icir'], 'icir mismatch'
    assert d['validation']['metrics'].get('max_abs_library_correlation') == rho
    art = np.load('factors/' + d['signal_artifact'], allow_pickle=False)
    assert art.shape == (len(grid), len(WATCHLIST)), 'artifact shape mismatch'
    print(f"VERIFY OK: id={d['factor_id']} status={d['validation']['status']} "
          f"ic={d['validation']['metrics']['ic']:.4f} icir={d['validation']['metrics']['icir']:.4f} "
          f"rho={d['validation']['metrics']['max_abs_library_correlation']:.3f} "
          f"artifact={d['signal_artifact']} shape={art.shape}", flush=True)
else:
    print("coexceed_down_60 FAILED re-validation -> NOT persisted", flush=True)

# ---------------- 2) debug updown_volume_20 ----------------
print("\n================ updown_volume_20 debug ================", flush=True)
for s in WATCHLIST:
    df = prices[s]
    print(f"{s:12s} rows={len(df):5d} has_volume={'volume' in df.columns} "
          f"volume_nan={df['volume'].isna().sum() if 'volume' in df.columns else 'NA'} "
          f"volume_zero={int((df['volume'] == 0).sum()) if 'volume' in df.columns else 'NA'} "
          f"volume_dtype={df['volume'].dtype if 'volume' in df.columns else 'NA'}", flush=True)
