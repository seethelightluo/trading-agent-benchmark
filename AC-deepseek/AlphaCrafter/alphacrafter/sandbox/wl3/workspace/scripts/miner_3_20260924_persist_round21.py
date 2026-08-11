"""Persist round-20b passing factors (2026-09-24 cycle).

Candidates that passed the gate in round 20b but were never persisted:
  1. leadlag_gap_spx_60 : corr(overnight_gap, prior-day SPX ret) over 60d  |IC|~0.059 |ICIR|~0.206
  2. down_vol_ratio_20  : downside semi-dev / upside semi-dev (20d)        |IC|~0.032 |ICIR|~0.097

Re-validated on the canonical warm-up window (2020-01-01..2026-07-15) with
gate-style Spearman max_abs_library_correlation, then persisted to
factors/<factor_id>.json + <factor_id>_signal.npy.
"""
import sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid, factor_to_panel,
                           validate_factor, signal_matrix, forward_returns,
                           rank_ic_series, VAL_START, VAL_END, persist_factor)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)}", flush=True)

# ---------- full effective library rank panels (Spearman rho) ----------
lib_panels = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        fid = d['factor_id']
        art = d.get('signal_artifact')
        if not art:
            continue
        p = 'factors/' + art
        arr = np.load(p, allow_pickle=False)
        if arr.shape == (len(grid), len(WATCHLIST)):
            lib_panels[fid] = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
    except Exception as e:
        print(f"  artifact ERR {f}: {e}", flush=True)
print(f"library panels: {len(lib_panels)} -> {sorted(lib_panels.keys())}", flush=True)


def spearman_rho_vs_library(panel):
    """Per-date Spearman rho vs each library factor; report max |rho| (gate-style)."""
    pm = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    per_id = {}
    for fid, lp in lib_panels.items():
        lm = lp.values
        corrs = []
        for t in range(len(grid)):
            x = pm[t]; y = lm[t]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xr = pd.Series(x[m]).rank().values
                yr = pd.Series(y[m]).rank().values
                xc = xr - xr.mean(); yc = yr - yr.mean()
                den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
                if den > 0:
                    corrs.append((xc * yc).sum() / den)
        if corrs:
            r = float(np.mean(corrs))
            per_id[fid] = r
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id, per_id


# ---------- factor constructions ----------
spx_ret = prices['SPX']['close'].pct_change()


def f_leadlag_gap_spx(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    z = pd.concat([gap.rename('g'), spx_ret.shift(1).rename('u')], axis=1).dropna()
    return z['g'].rolling(60).corr(z['u'])


def f_down_vol_ratio(df, s):
    r = df['close'].pct_change()
    mu = r.rolling(20).mean()
    dev = r - mu
    dn = (dev.clip(upper=0.0) ** 2).rolling(20).mean() ** 0.5
    up = (dev.clip(lower=0.0) ** 2).rolling(20).mean() ** 0.5
    return dn / up.replace(0, np.nan)


fwd_ret = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}
specs = [
    dict(fid='leadlag_gap_spx_60', name='SPX Lead-Lag Gap Correlation 60d',
         fn=f_leadlag_gap_spx, direction=-1,
         expr='corr(open_t/close_{t-1} - 1, ret_SPX_{t-1}) over 60d',
         desc='Rolling 60d correlation between each asset\'s overnight gap and the prior-day SPX return. '
              'Captures cross-market information diffusion: assets whose gaps are most responsive to US '
              'equity news underperform (negative IC, direction=-1).',
         deps=['open', 'close', 'SPX close'], params={'window': 60, 'lead_asset': 'SPX'},
         tags=['cross-market', 'lead-lag', 'information-diffusion']),
    dict(fid='down_vol_ratio_20', name='Downside/Upside Semi-Deviation Ratio 20d',
         fn=f_down_vol_ratio, direction=-1,
         expr='sqrt(mean(min(r-mean,0)^2)) / sqrt(mean(max(r-mean,0)^2)) over 20d',
         desc='Ratio of downside semi-deviation to upside semi-deviation of daily returns over 20d. '
              'Assets with disproportionately large downside volatility are penalized (negative IC, direction=-1); '
              'a volatility-asymmetry risk factor.',
         deps=['close'], params={'window': 20},
         tags=['volatility', 'asymmetry', 'risk'])
]

for spec in specs:
    fid = spec['fid']
    try:
        panel = factor_to_panel(spec['fn'], prices)
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid}: validation None -> SKIP", flush=True)
            continue
        rho, rho_id, per_id = spearman_rho_vs_library(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rho_id
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
        print(f"\n{fid}: IC10={m['ic']:.4f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
              f"coverage={m['coverage_asset_days']:.3f} turn={m['turnover_10d_rank']:.2f} "
              f"spearman_rho={rho:.3f}({rho_id})", flush=True)
        print(f"  decay: { {h: round(v,4) for h, v in m['decay_ic_by_horizon'].items()} }", flush=True)
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            print(f"{fid}: gate FAIL -> NOT persisted", flush=True)
            continue
        path, arr = persist_factor(
            factor_id=fid, factor_name=spec['name'],
            expression=spec['expr'], description=spec['desc'],
            dependencies=spec['deps'], parameters=spec['params'],
            expected_direction=spec['direction'], panel=panel, metrics=m,
            tags=spec['tags'], grid=grid, prices=prices, version='1.0.0',
            status='EFFECTIVE',
            regime_notes='2020-01..2026-07 warm-up; cross-asset regimes (COVID, 2022 tightening, 2023-25 risk-on, crypto cycles). '
                         f'Re-validated 2026-09-24 on canonical grid; max library rho={rho:.3f} vs {rho_id}.')
        print(f"  PERSISTED -> {path} shape={arr.shape}", flush=True)
    except Exception as e:
        print(f"{fid}: EXCEPTION {e}", flush=True)

print("\nDone persistence round-21.")
