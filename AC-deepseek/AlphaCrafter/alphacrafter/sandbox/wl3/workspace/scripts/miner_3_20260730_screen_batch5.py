"""miner_3 2026-07-30 screening round 4 (batch5): NEW orthogonal families.

Library (11 effective): spx/hs300 beta, dxy/eurusd/vix cond beta, vol_adj_mom,
max_ret, vol_of_vol, dd_duration_resid, skew_term, hilo_pos. Already tried:
btc/wti/us10y/basket beta, kurt level, vol_z, range_level, autocorr, var_ratio,
volume_z, vol_price_corr, bw_zscore, RSI/bollinger, breakout.

New candidates (distinct economic drivers):
  1. amihud_illiq_20   : log(1+mean(|ret|/volume,20)) - liquidity/impact (volume-based
                         but NOT a volume z-score; cross-sectionally scale-free).
  2. downside_vol_ratio_60 : std(ret<0)/std(ret>0) over 60d - vol asymmetry
                         (skew of risk), distinct from vol-of-vol LEVEL.
  3. gold_beta_cond_60x20  : beta(asset,XAU,60) * (XAU 20d move) - safe-haven
                         anchor, distinct from DXY/VIX/EURUSD conditioning.
  4. upday_ratio_60    : fraction of positive days over 60d - trend participation.
  5. idio_vol_ratio_60 : 1 - R2 from 60d regression on equal-weight basket
                         (idiosyncratic share), distinct from vol level.
  6. copper_beta_60    : rolling 60d beta to COPPER - cyclical commodity anchor.
  7. gap_drift_20      : mean(open/prevclose - 1) over 20d - overnight/gap drift.
  8. dd_depth_120      : max drawdown depth over 120d (vs duration in library).

Correlation audit vs extended library = ALL artifact-bearing factor JSONs.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           max_library_correlation, canonical_grid,
                           VAL_START, VAL_END, WATCHLIST,
                           load_artifact_matrix, Path)

prices = load_prices(days=2100)
print(f"loaded {len(prices)} assets")
grid = canonical_grid(prices)
print(f"canonical grid: {grid.min().date()}..{grid.max().date()} n={len(grid)}")

xau = prices.get('XAU')
copper = prices.get('COPPER')

ret_panel = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()})
basket = ret_panel.mean(axis=1)


def load_all_artifact_panels():
    out = {}
    for jp in sorted(Path('factors').glob('*.json')):
        if jp.name == 'factor_ensemble.json':
            continue
        payload = json.loads(jp.read_text(encoding='utf-8'))
        if payload.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        art = load_artifact_matrix(str(jp))
        if art is None or art.shape[0] != len(grid):
            continue
        out[payload['factor_id']] = pd.DataFrame(art, index=grid, columns=WATCHLIST)
    return out


def amihud_illiq_20(df, s):
    r = df['close'].pct_change().abs()
    vol = df['volume'].replace(0, np.nan).astype(float)
    ill = (r / vol).replace([np.inf, -np.inf], np.nan)
    return np.log1p(ill.rolling(20, min_periods=10).mean())


def downside_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    dn = r.where(r < 0)
    up = r.where(r > 0)
    d = dn.rolling(60, min_periods=25).std()
    u = up.rolling(60, min_periods=25).std()
    return (d / u.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def gold_beta_cond_60x20(df, s):
    if xau is None:
        return None
    r = df['close'].pct_change()
    rg = xau['close'].pct_change()
    z = pd.concat([r.rename('r'), rg.rename('g')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['g']) / z['g'].rolling(60).var().replace(0, np.nan)
    g_move = xau['close'] / xau['close'].shift(20) - 1.0
    return (b * g_move).reindex(z.index)


def upday_ratio_60(df, s):
    r = df['close'].pct_change()
    pos = (r > 0).astype(float)
    return pos.rolling(60, min_periods=30).mean()


def idio_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), basket.rename('b')], axis=1).dropna()
    var_b = z['b'].rolling(60, min_periods=30).var()
    cov = z['r'].rolling(60, min_periods=30).cov(z['b'])
    var_r = z['r'].rolling(60, min_periods=30).var()
    r2 = (cov ** 2 / (var_b * var_r)).replace([np.inf, -np.inf], np.nan)
    return (1.0 - r2).reindex(z.index)


def copper_beta_60(df, s):
    if copper is None:
        return None
    r = df['close'].pct_change()
    rc = copper['close'].pct_change()
    z = pd.concat([r.rename('r'), rc.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['c']) / z['c'].rolling(60).var().replace(0, np.nan)
    return b.reindex(z.index)


def gap_drift_20(df, s):
    prev_close = df['close'].shift(1)
    gap = df['open'] / prev_close.replace(0, np.nan) - 1.0
    return gap.rolling(20, min_periods=10).mean()


def dd_depth_120(df, s):
    c = df['close']
    roll_max = c.rolling(120, min_periods=60).max()
    dd = c / roll_max.replace(0, np.nan) - 1.0
    return dd.rolling(120, min_periods=60).min()


library_panels = load_all_artifact_panels()
print('extended effective library:', sorted(library_panels.keys()), f"({len(library_panels)} factors)")

candidates = {
    'amihud_illiq_20': amihud_illiq_20,
    'downside_vol_ratio_60': downside_vol_ratio_60,
    'gold_beta_cond_60x20': gold_beta_cond_60x20,
    'upday_ratio_60': upday_ratio_60,
    'idio_vol_ratio_60': idio_vol_ratio_60,
    'copper_beta_60': copper_beta_60,
    'gap_drift_20': gap_drift_20,
    'dd_depth_120': dd_depth_120,
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

print("SUMMARY_TABLE")
for fid, (m, _) in results.items():
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and m['max_abs_library_correlation'] < 0.5
    print(f"{fid:28s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} rho={m['max_abs_library_correlation']:.3f} vs {str(m['max_corr_library_id']):30s} cov={m['coverage_asset_days']:.2f} -> {'ADMIT' if ok else 'skip'}")
