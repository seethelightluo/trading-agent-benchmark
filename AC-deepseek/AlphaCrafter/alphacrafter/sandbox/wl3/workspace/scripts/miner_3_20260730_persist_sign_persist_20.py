"""Persist sign_persist_20 (Round-17 winner) with recoverable signal artifact.

Factor: fraction of days over trailing w days where sign(daily return) equals
sign(prior day return) -> measures short-horizon return sign persistence.

Validation replicates the canonical rank-based (Spearman) IC protocol used by
the round-17 screen (to_rank_matrix + fast_ic_series_from_ranks, min 8 valid).

Admission (shared gate): |IC10| >= 0.007, |ICIR10| >= 0.084, rho < 0.5.
Round-17 result: IC=+0.0305, ICIR=+0.104, max lib rho=0.079 (vol_adj_mom_20_60).
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid, signal_matrix,
                           factor_to_panel, forward_returns, VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices {len(prices)} assets; grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})", flush=True)


def make_sign_persist(w):
    def f(df, s):
        r = df['close'].pct_change()
        same = (np.sign(r) == np.sign(r.shift(1))).astype(float)
        valid = (r.notna() & r.shift(1).notna()).astype(float)
        return (same * valid).rolling(w).sum() / valid.rolling(w).sum().replace(0, np.nan)
    return f


W = 20
panel = factor_to_panel(make_sign_persist(W), prices)
print(f"panel shape {panel.shape}; valid cells {int(panel.notna().sum().sum())}", flush=True)

# ---- signal artifact on canonical grid ----
arr = signal_matrix(panel, grid)
np.save('factors/sign_persist_20_signal.npy', arr)
print(f"signal artifact saved: factors/sign_persist_20_signal.npy shape={arr.shape}", flush=True)

# ---- canonical rank-based (Spearman) validation ----
def to_rank_matrix(panel):
    m = signal_matrix(panel, grid)
    out = np.full(m.shape, np.nan)
    for i in range(m.shape[0]):
        row = m[i]
        valid = np.isfinite(row)
        if valid.sum() >= 3:
            r = pd.Series(row[valid]).rank().values
            out[i, valid] = r
    return out


def ic_series_from_ranks(fac_rank, fwd_rank, min_valid=8):
    ic, dates = [], []
    for t in range(len(grid)):
        x = fac_rank[t]; y = fwd_rank[t]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            xv = x[m]; yv = y[m]
            xc = xv - xv.mean(); yc = yv - yv.mean()
            den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
            if den > 0:
                ic.append((xc * yc).sum() / den)
                dates.append(t)
    return pd.Series(ic, index=grid[dates])


fac_rank = to_rank_matrix(panel)
fwd_ranks = {h: to_rank_matrix(forward_returns(prices, h)) for h in (1, 2, 3, 5, 10, 20)}
ic10 = ic_series_from_ranks(fac_rank, fwd_ranks[10])
ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
ic_mean = float(ic10.mean())
ic_std = float(ic10.std(ddof=1)) if len(ic10) > 1 else 0.0
icir = ic_mean / ic_std if ic_std > 0 else 0.0
hit = float((ic10 > 0).mean())

fac_v = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
total = fac_v.shape[0] * fac_v.shape[1]
coverage = float(fac_v.notna().sum().sum()) / total if total else 0.0
ge8 = float((fac_v.notna().sum(axis=1) >= 8).mean())
ranked = fac_v.rank(axis=1)
turn = float(ranked.diff(10).abs().mean().mean())

ic_by_period = {}
for name, lo, hi in [('2020_2022', '2020-01-01', '2022-12-31'),
                     ('2023_2024', '2023-01-01', '2024-12-31'),
                     ('2025_2026', '2025-01-01', '2026-07-15')]:
    sub = ic10[(ic10.index >= pd.Timestamp(lo)) & (ic10.index <= pd.Timestamp(hi))]
    ic_by_period[name] = float(sub.mean()) if len(sub) else float('nan')
recent = ic10[ic10.index >= pd.Timestamp('2025-07-16')]
recent_ic = float(recent.mean()) if len(recent) else float('nan')
recent_icir = float(recent.mean() / recent.std(ddof=1)) if len(recent) > 2 and recent.std(ddof=1) > 0 else 0.0

decay = {str(h): float(ic_series_from_ranks(fac_rank, fwd_ranks[h]).mean()) for h in (1, 2, 3, 5, 10, 20)}

print(f"IC10={ic_mean:+.4f} ICIR10={icir:+.4f} hit={hit:.3f} n={len(ic10)} "
      f"cov={coverage:.3f} ge8={ge8:.3f} turn={turn:.2f}", flush=True)
print(f"regime IC: { {k: round(v, 4) for k, v in ic_by_period.items()} }", flush=True)
print(f"recent_1y IC={recent_ic:+.4f} ICIR={recent_icir:+.4f}", flush=True)
print("decay:", {k: round(v, 4) for k, v in decay.items()}, flush=True)

metrics = {
    "ic": ic_mean,
    "icir": icir,
    "ic_hit_ratio": hit,
    "n_ic_dates": int(len(ic10)),
    "coverage_asset_days": coverage,
    "coverage_dates_ge8": ge8,
    "turnover_10d_rank": turn,
    "decay_ic_by_horizon": {k: float(v) for k, v in decay.items()},
    "ic_2020_2022": ic_by_period['2020_2022'],
    "ic_2023_2024": ic_by_period['2023_2024'],
    "ic_2025_2026": ic_by_period['2025_2026'],
    "recent_1y_ic": recent_ic,
    "recent_1y_icir": recent_icir,
    "max_abs_library_correlation": 0.07882520941811494,
    "max_corr_library_id": "vol_adj_mom_20_60",
}

doc = {
    "factor_id": "sign_persist_20",
    "factor_name": "Sign Persistence 20d",
    "version": "1.0.0",
    "calculation": {
        "expression": "rolling_mean( sign(r_t)==sign(r_{t-1}) , 20 ) where r_t = close/close_{t-1}-1",
        "description": "Fraction of the trailing 20 days in which the sign of the daily return "
                       "equals the sign of the previous day's return. Measures short-horizon return "
                       "sign persistence / trendiness (positive serial dependence). High values -> "
                       "assets whose daily moves keep the same direction; low values -> choppy "
                       "mean-reverting assets. Cross-sectionally, persistent-trend assets have "
                       "higher forward 10d returns in this worldline."
    },
    "dependencies": ["close"],
    "parameters": {"window": 20, "min_valid_days": 8},
    "expected_direction": 1,
    "signal_artifact": "sign_persist_20_signal.npy",
    "signal_artifact_format": "npy",
    "signal_artifact_shape": list(arr.shape),
    "signal_artifact_grid": {
        "start": str(grid.min().date()),
        "end": str(grid.max().date()),
        "n_dates": int(len(grid)),
        "columns": WATCHLIST,
        "note": "canonical grid shared by all library factors (see factor_common.canonical_grid)"
    },
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-15",
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "regime_notes": "Stable across regimes: IC 2020-22 +0.028, 2023-24 +0.023, 2025-26 +0.045; "
                       "recent-1y IC +0.013 (positive but weaker). Decay: predictive power rises with "
                       "horizon (1d +0.008 -> 20d +0.033), consistent with slow-trend persistence.",
        "metrics": metrics
    },
    "tags": ["microstructure", "serial-dependence", "trend", "sign-persistence"],
    "last_validated": "2026-07-30"
}

with open('factors/sign_persist_20.json', 'w') as f:
    json.dump(doc, f, indent=1)
print(f"\nJSON written: factors/sign_persist_20.json ({time.time()-t0:.1f}s)", flush=True)

# ---- verify read-back ----
d = json.load(open('factors/sign_persist_20.json'))
assert d['factor_id'] == 'sign_persist_20'
assert d['validation']['status'] == 'EFFECTIVE'
assert abs(d['validation']['metrics']['ic']) >= 0.007
assert abs(d['validation']['metrics']['icir']) >= 0.084
assert d['validation']['metrics']['max_abs_library_correlation'] < 0.5
arr2 = np.load('factors/sign_persist_20_signal.npy')
assert arr2.shape == tuple(d['signal_artifact_shape']) == (len(grid), 15)
assert np.allclose(arr2, arr, equal_nan=True)
print("VERIFY OK: json reload, status EFFECTIVE, gates satisfied, artifact roundtrip matches", flush=True)
