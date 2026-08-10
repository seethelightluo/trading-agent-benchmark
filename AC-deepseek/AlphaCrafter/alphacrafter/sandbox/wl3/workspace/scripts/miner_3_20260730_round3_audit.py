"""miner_3 2026-07-30 round3 full audit: correlation vs extended library for
dd_duration_120, basket_beta_60, downside_beta_60_spx (fixed rolling beta)."""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           max_library_correlation, canonical_grid,
                           WATCHLIST, build_library_panels,
                           load_artifact_matrix, Path)

prices = load_prices(days=2100)
spx = prices.get('SPX')
ret_panel = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()})
basket = ret_panel.mean(axis=1)


def dd_duration_120(df, s):
    c = df['close']; h = c.rolling(120, min_periods=60).max()
    is_high = (c == h).fillna(False)
    idx_high = np.flatnonzero(is_high.values)
    positions = np.arange(len(c))
    last = np.searchsorted(idx_high, positions) - 1
    dur = np.where(last >= 0, positions - idx_high[np.maximum(last, 0)], np.nan)
    return pd.Series(np.log1p(dur), index=c.index)


def basket_beta_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), basket.rename('b')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['b']) / z['b'].rolling(60).var().replace(0, np.nan)
    return b.reindex(z.index)


def downside_beta_spx(df, s):
    if spx is None:
        return None
    r = df['close'].pct_change(); rm = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
    a = z.values
    n, win = len(a), 60
    vals = np.full(n, np.nan)
    if n >= win:
        sw = np.lib.stride_tricks.sliding_window_view(a, (win, 2))[:, :, 0, :]
        for i in range(len(sw)):
            w = sw[i]
            m = w[:, 1] < 0
            if m.sum() < 12:
                continue
            r_, ms_ = w[:, 0][m], w[:, 1][m]
            v = ms_.var(ddof=1)
            if v <= 0 or not np.isfinite(v):
                continue
            vals[i + win - 1] = np.cov(r_, ms_, ddof=1)[0, 1] / v
    return pd.Series(vals, index=z.index)


def load_all_artifact_panels():
    out = {}
    grid = canonical_grid(prices)
    for jp in sorted(Path('factors').glob('*.json')):
        if jp.name == 'factor_ensemble.json':
            continue
        art = load_artifact_matrix(str(jp))
        if art is None or art.shape[0] != len(grid):
            continue
        fid = json.loads(jp.read_text(encoding='utf-8')).get('factor_id')
        out[fid] = pd.DataFrame(art, index=grid, columns=WATCHLIST)
    return out


library_panels = build_library_panels(prices)
library_panels.update(load_all_artifact_panels())
print('extended library:', sorted(library_panels.keys()))

for fid, fn in [('dd_duration_120', dd_duration_120),
                ('basket_beta_60', basket_beta_60),
                ('downside_beta_60_spx', downside_beta_spx)]:
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(fid, 'insufficient data'); continue
    rho, best = max_library_correlation(panel, library_panels)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = best
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"\n== {fid}: panel {panel.shape}")
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=1, default=str))
    print('decay:', json.dumps(m['decay_ic_by_horizon'], default=str))
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f} |ICIR|={abs(m['icir']):.4f} -> {'PASS' if ok else 'FAIL'}  max_corr={rho:.3f} vs {best}")
