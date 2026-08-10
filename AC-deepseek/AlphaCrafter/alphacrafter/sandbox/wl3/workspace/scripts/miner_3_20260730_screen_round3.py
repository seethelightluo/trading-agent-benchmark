"""miner_3 2026-07-30 screening round 3: duration/tail/regime/beta-asymmetry family.

Winning families so far: beta-to-risk-anchor (spx_beta_60 IC .080), conditional
FX/VIX beta, vol-adjusted momentum, hilo/breakout, RSI/bollinger oscillator.
Screen NEW, distinct members not yet covered by any persisted factor:
  1. dd_duration_120 : log(1 + days since last 120d closing high). Depth was
     tested (dd_60d); DURATION is a different dimension (time-in-drawdown).
  2. vol_z_20_120    : z-score of 20d realized vol vs its trailing 120d
     distribution -> vol REGIME (rising/falling), distinct from vol level
     ratio (vol_term_20_60) and vol-of-vol (2nd moment of vol).
  3. kurt_60d        : rolling 60d excess kurtosis of returns -> tail fatness.
  4. downside_beta_60_spx : beta vs SPX computed ONLY on SPX down days
     (asymmetric/crash exposure) - anchor is same as spx_beta_60 but the
     conditioning should make it distinct.
  5. basket_beta_60  : beta vs equal-weight cross-asset basket (includes
     crypto/commodities) -> global risk exposure, different anchor from SPX.
  6. range_level_20  : mean daily (high-low)/close over 20d -> range-based vol
     level (Parkinson-style), distinct from close-close vol-of-vol.

Correlation audit uses extended library: 4 recomputed library factors + ALL
artifact-bearing effective factors currently persisted.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel,
                           validate_factor, max_library_correlation,
                           canonical_grid, VAL_START, VAL_END,
                           WATCHLIST, build_library_panels,
                           load_artifact_matrix, Path)

prices = load_prices(days=2100)
spx = prices.get('SPX')
print(f"loaded {len(prices)} assets")


def load_all_artifact_panels():
    """Load every persisted factor that carries a recoverable signal artifact."""
    out = {}
    grid = canonical_grid(prices)
    for jp in sorted(Path('factors').glob('*.json')):
        if jp.name == 'factor_ensemble.json':
            continue
        art = load_artifact_matrix(str(jp))
        if art is None:
            continue
        fid = json.loads(jp.read_text(encoding='utf-8')).get('factor_id')
        # align artifact rows to canonical grid dates (grid length must match)
        if art.shape[0] != len(grid):
            print(f'  skip {fid}: artifact rows {art.shape[0]} != grid {len(grid)}')
            continue
        out[fid] = pd.DataFrame(art, index=grid, columns=WATCHLIST)
    return out


def days_since_high_120(df, s):
    c = df['close']
    h = c.rolling(120, min_periods=60).max()
    is_high = (c == h).fillna(False)
    idx_high = np.flatnonzero(is_high.values)
    positions = np.arange(len(c))
    last = np.searchsorted(idx_high, positions) - 1
    dur = np.where(last >= 0, positions - idx_high[np.maximum(last, 0)], np.nan)
    return pd.Series(np.log1p(dur), index=c.index)


def vol_z_20_120(df, s):
    r = df['close'].pct_change()
    v20 = r.rolling(20).std()
    mu = v20.rolling(120, min_periods=60).mean()
    sd = v20.rolling(120, min_periods=60).std()
    return ((v20 - mu) / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def kurt_60d(df, s):
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=30).kurt()


def downside_beta_spx(df, s):
    if spx is None:
        return None
    r = df['close'].pct_change()
    rm = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
    def f(w):
        w = np.asarray(w, dtype=float)
        m = w[:, 1] < 0
        if m.sum() < 12:
            return np.nan
        r_, ms_ = w[:, 0][m], w[:, 1][m]
        v = ms_.var(ddof=1)
        if v <= 0 or not np.isfinite(v):
            return np.nan
        return np.cov(r_, ms_, ddof=1)[0, 1] / v
    return z.rolling(60, min_periods=30).apply(f, raw=True)


def basket_beta_60(df, s, basket=None):
    if basket is None:
        return None
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), basket.rename('b')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['b']) / z['b'].rolling(60).var().replace(0, np.nan)
    return b.reindex(z.index)


def range_level_20(df, s):
    rng = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
    return rng.rolling(20).mean()


# equal-weight cross-asset basket daily return
ret_panel = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()})
basket = ret_panel.mean(axis=1)
print(f'basket span {basket.index.min().date()}..{basket.index.max().date()}')

library_panels = build_library_panels(prices)
library_panels.update(load_all_artifact_panels())
print('extended library size:', len(library_panels), '->', sorted(library_panels.keys()))

candidates = {
    'dd_duration_120': days_since_high_120,
    'vol_z_20_120': vol_z_20_120,
    'kurt_60d': kurt_60d,
    'downside_beta_60_spx': downside_beta_spx,
    'basket_beta_60': lambda df, s: basket_beta_60(df, s, basket),
    'range_level_20': range_level_20,
}

results = {}
for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data -> None'); continue
    rho, best = max_library_correlation(panel, library_panels)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = best
    results[fid] = (m, panel)
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"Factor {fid}: panel {panel.shape} range {panel.index.min().date()}..{panel.index.max().date()}")
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=1, default=str))
    print("decay:", json.dumps(m['decay_ic_by_horizon'], default=str))
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {abs(m['ic'])>=0.007} | |ICIR|={abs(m['icir']):.4f}>=0.084 {abs(m['icir'])>=0.084} -> {'PASS' if ok else 'FAIL'}  max_corr={rho:.3f} vs {best}")
    print()
