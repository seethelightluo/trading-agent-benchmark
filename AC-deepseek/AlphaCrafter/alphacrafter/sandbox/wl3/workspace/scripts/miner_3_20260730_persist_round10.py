"""Persist round-10 passing candidates (gap_dir_20, intraday_ret_skew_20, win_rate_40).

Recomputes panels with factor_common, runs the shared validation battery
(validate_factor), audits max library correlation against ALL currently
effective library signal artifacts, and persists JSON + .npy artifact.
"""
import sys, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, canonical_grid,
                           signal_matrix, factor_to_panel, validate_factor,
                           persist_factor)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"prices {len(prices)} assets; canonical grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})", flush=True)

# ---------- library artifacts ----------
lib = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            art = d.get('signal_artifact')
            if art and Path('factors', art).exists():
                lib[d['factor_id']] = np.load(Path('factors', art))
    except Exception as e:
        print("lib skip", f, e)
print(f"library artifacts: {len(lib)} -> {sorted(lib)}", flush=True)

MIN_V = 8


def rank_rows(M):
    T, n = M.shape
    R = np.full_like(M, np.nan)
    for t in range(T):
        v = M[t]
        m = np.isfinite(v)
        if m.sum() >= MIN_V:
            idx = np.where(m)[0]
            R[t, idx] = v[idx].argsort().argsort().astype(float)
    return R


def row_spearman(RA, RB):
    m = np.isfinite(RA) & np.isfinite(RB)
    A = np.where(m, RA, np.nan)
    B = np.where(m, RB, np.nan)
    cnt = m.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        Ac = A - np.nanmean(A, axis=1, keepdims=True)
        Bc = B - np.nanmean(B, axis=1, keepdims=True)
        num = np.nansum(Ac * Bc, axis=1)
        den = np.sqrt(np.nansum(Ac * Ac, axis=1) * np.nansum(Bc * Bc, axis=1))
        rho = num / den
    rho[~((cnt >= MIN_V) & (den > 0))] = np.nan
    return rho


def max_lib_corr(mat):
    Rc = rank_rows(mat)
    best, best_id = 0.0, None
    for fid, la in lib.items():
        if la.shape[0] < mat.shape[0]:
            Rc_use = Rc[-la.shape[0]:]
        else:
            Rc_use = Rc
        Rl = rank_rows(la)
        rho = row_spearman(Rc_use, Rl)
        r = float(np.nanmean(rho)) if np.isfinite(rho).any() else 0.0
        if abs(r) > best:
            best, best_id = abs(r), fid
    return best, best_id


# ---------- candidate definitions (identical to round10 screen) ----------
def gap_dir_20(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    atr = pd.concat([(df['high'] - df['low']),
                     (df['high'] - df['close'].shift(1)).abs(),
                     (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr = atr.rolling(14).mean().replace(0, np.nan)
    return (gap / atr).rolling(20).sum()


def intraday_ret_skew_20(df, s):
    intr = df['close'] / df['open'] - 1.0
    return intr.rolling(20).skew()


def win_rate_40(df, s):
    pos = (df['close'].pct_change() > 0).astype(float)
    return pos.rolling(40).mean()


candidates = {
    'gap_dir_20': dict(
        fn=gap_dir_20,
        name='Gap direction vs ATR (20d cumulative)',
        expr="sum_{t-19..t} (open_t/close_{t-1} - 1) / ATR14_t",
        desc=("Cumulative 20-day open-gap sign scaled by trailing 14-day ATR. "
              "Assets that consistently gap in a direction (relative to their "
              "recent range) show short-to-intermediate-term continuation. "
              "Cross-sectional positive IC at 10d horizon; low correlation to "
              "existing library (max rho ~0.23 vs vol_adj_mom_20_60)."),
        deps=['open', 'close', 'high', 'low'],
        params={'window': 20, 'atr_window': 14},
        tags=['gap', 'trend', 'price-action', 'cross-asset'],
        direction=1,
        regime="Validated 2020-01-01..2026-07-15 across COVID crash, 2022 tightening bear, "
               "2023-25 risk-on, crypto/commodity cycles. Positive IC at all horizons 1-20d "
               "with monotone decay growth toward 20d; hit ratio 0.54."),
    'intraday_ret_skew_20': dict(
        fn=intraday_ret_skew_20,
        name='20d skew of intraday return (close/open)',
        expr="skew_20d(close/open - 1)",
        desc=("Skewness of the daily close-vs-open (intraday) return over 20 days. "
              "Positive skew (fat right tail of intraday moves) predicts higher "
              "forward 10d returns cross-sectionally — an asymmetry/trend-quality signal "
              "distinct from raw momentum. Max library rho ~0.36 vs skew_term_20_60 "
              "(term-structure skew) — shares the skew concept but different construction."),
        deps=['close', 'open'],
        params={'window': 20},
        tags=['skew', 'intraday', 'asymmetry', 'cross-asset'],
        direction=1,
        regime="Validated 2020-01-01..2026-07-15; positive IC at all horizons with hit "
               "ratio 0.565, highest ICIR among round-10 candidates (0.128)."),
    'win_rate_40': dict(
        fn=win_rate_40,
        name='40d win rate (fraction of up days)',
        expr="mean_{t-39..t}(close_t/close_{t-1} - 1 > 0)",
        desc=("Fraction of positive daily returns over the trailing 40 sessions. "
              "A robust trend-quality measure: high win rate (many small up days, "
              "few down days) predicts continuation. Positive IC 0.036 at 10d. "
              "Max library rho ~0.45 vs vol_adj_mom_20_60 — below the 0.5 gate but "
              "adds a count/quality dimension beyond magnitude momentum."),
        deps=['close'],
        params={'window': 40},
        tags=['trend', 'win-rate', 'quality', 'cross-asset'],
        direction=1,
        regime="Validated 2020-01-01..2026-07-15; hit ratio 0.563, decay monotone up to 20d."),
}

for fid, cfg in candidates.items():
    panel = factor_to_panel(cfg['fn'], prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: INSUFFICIENT -> skip", flush=True)
        continue
    mat = signal_matrix(panel, grid)
    rho, lib_id = max_lib_corr(mat)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = lib_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    print(f"\n=== {fid} === panel {panel.shape} | IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
          f"ge8={m['coverage_dates_ge8']:.3f} rho={rho:.3f}({lib_id}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print(f"{fid}: FAIL -> not persisted", flush=True)
        continue
    path, arr = persist_factor(
        factor_id=fid,
        factor_name=cfg['name'],
        expression=cfg['expr'],
        description=cfg['desc'],
        dependencies=cfg['deps'],
        parameters=cfg['params'],
        expected_direction=cfg['direction'],
        panel=panel,
        metrics=m,
        tags=cfg['tags'],
        grid=grid,
        prices=prices,
        version='1.0.0',
        status='EFFECTIVE',
        regime_notes=cfg['regime'],
        extra={'signal_provenance': {
            'source': 'recomputed from alphacrafter.sim.utils daily OHLC series',
            'panel_shape': f"{panel.shape[0]}x{panel.shape[1]}",
            'panel_range': f"{panel.index.min().date()}..{panel.index.max().date()}",
            'validation_window': '2020-01-01..2026-07-15',
            'ic_method': 'daily cross-sectional Spearman rank IC vs 10d forward return',
            'note': 'expression deterministic and reproducible from OHLC series only'}},
    )
    print(f"{fid}: PERSISTED -> {path} artifact {arr.shape}", flush=True)

print("\n--- verify round-trip ---", flush=True)
for fid in candidates:
    p = Path('factors') / f'{fid}.json'
    if not p.exists():
        print(f"{fid}: MISSING", flush=True)
        continue
    d = json.loads(p.read_text(encoding='utf-8'))
    art = Path('factors') / d['signal_artifact']
    ok = (d['factor_id'] == fid and d['validation']['status'] == 'EFFECTIVE'
          and art.exists()
          and abs(d['validation']['metrics']['ic']) >= 0.007
          and abs(d['validation']['metrics']['icir']) >= 0.084
          and d['validation']['metrics']['max_abs_library_correlation'] < 0.5)
    print(f"{fid}: id={d['factor_id']} status={d['validation']['status']} "
          f"ic={d['validation']['metrics']['ic']:+.4f} icir={d['validation']['metrics']['icir']:+.4f} "
          f"rho={d['validation']['metrics']['max_abs_library_correlation']:.3f} "
          f"artifact={d['signal_artifact']}({np.load(art).shape}) -> {'VERIFIED' if ok else 'CHECK'}", flush=True)
print("done", flush=True)
