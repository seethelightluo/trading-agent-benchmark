"""miner_3 2026-07-30 screening round 5 (batch6): finish round-4 novel ideas + fresh families.

Remaining round-4 ideas (crash-fixed):
  semi_vol_ratio_60, parkinson_ratio_20, trend_r2_60, basket_corr_60, min_ret_20d,
  updown_vol_asym_20, cn10y_beta_60, us10y_beta_60, gap_fill_ratio_20, ret_skew_raw_60

Fresh orthogonal ideas (distinct economic drivers, absent from library):
  range_squeeze_20_60  : mean(high-low,20)/mean(high-low,60) - volatility squeeze
  gap_down_freq_20     : freq(open < prev low, 20d) - downside gap microstructure
  spread_beta_uscn_60  : beta(asset, US10Y-CN10Y spread change, 60) - cross-market
                         rate-differential sensitivity (EM/DM growth signal)
  up_capture_60        : mean(up-day ret)/mean(|down-day ret|) - asymmetric capture

US10Y/CN10Y are tradable watchlist members -> use prices[] directly (round-4 script
used load_index which returned None for these).

Library for correlation audit: 4 legacy recomputed + all 12 EFFECTIVE artifact factors.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           max_library_correlation, canonical_grid,
                           WATCHLIST, load_artifact_matrix, Path)

prices = load_prices(days=3000)
grid = canonical_grid(prices)
print(f"loaded {len(prices)} assets; canonical grid {grid.min().date()}..{grid.max().date()} n={len(grid)}")

ret_panel = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()})
basket = ret_panel.mean(axis=1)
us10y = prices.get('US10Y')
cn10y = prices.get('CN10Y')
print('US10Y rows:', 0 if us10y is None else len(us10y),
      'CN10Y rows:', 0 if cn10y is None else len(cn10y))


def load_effective_artifact_panels():
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


# --- candidate definitions ---
def semi_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    dv = r.where(r < 0).rolling(60, min_periods=25).std()
    uv = r.where(r > 0).rolling(60, min_periods=25).std()
    return (dv / uv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def parkinson_ratio_20(df, s):
    c = df['close'].replace(0, np.nan)
    park = np.sqrt((np.log(df['high'] / df['low'].replace(0, np.nan)) ** 2).rolling(20, min_periods=10).mean() / (4 * np.log(2)))
    cc = c.pct_change().rolling(20, min_periods=10).std()
    return (park / cc.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def trend_r2_60(df, s):
    c = df['close']
    out = pd.Series(np.nan, index=c.index)
    vals = c.values
    n = len(vals)
    x = np.arange(60, dtype=float)
    sx, sx2 = x.sum(), (x ** 2).sum()
    denom = n * sx2 - sx * sx
    for i in range(60 - 1, n):
        w = vals[i - 59:i + 1]
        if not np.all(np.isfinite(w)):
            continue
        sy, sy2, sxy = w.sum(), (w ** 2).sum(), float((x * w).sum())
        if (n * sy2 - sy * sy) <= 0 or denom <= 0:
            continue
        r = (n * sxy - sx * sy) / np.sqrt(denom * (n * sy2 - sy * sy))
        out.iloc[i] = r * r
    return out


def basket_corr_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), basket.rename('b')], axis=1).dropna()
    return z['r'].rolling(60, min_periods=30).corr(z['b'])


def min_ret_20d(df, s):
    return df['close'].pct_change().rolling(20, min_periods=10).min()


def updown_vol_asym_20(df, s):
    r = df['close'].pct_change()
    dv = r.where(r < 0).rolling(20, min_periods=8).std()
    uv = r.where(r > 0).rolling(20, min_periods=8).std()
    tot = r.rolling(20, min_periods=8).std().replace(0, np.nan)
    return ((uv - dv) / tot).replace([np.inf, -np.inf], np.nan)


def beta_to(df, s, anchor_df):
    if anchor_df is None:
        return None
    r = df['close'].pct_change()
    a = anchor_df['close'].pct_change()
    z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['a']) / z['a'].rolling(60, min_periods=30).var().replace(0, np.nan)
    return (b.replace([np.inf, -np.inf], np.nan)).reindex(z.index)


def cn10y_beta_60(df, s):
    return beta_to(df, s, cn10y)


def us10y_beta_60(df, s):
    return beta_to(df, s, us10y)


def gap_fill_ratio_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return ((df['close'] - df['open']) / rng).rolling(20, min_periods=10).mean()


def ret_skew_raw_60(df, s):
    return df['close'].pct_change().rolling(60, min_periods=30).skew()


def range_squeeze_20_60(df, s):
    rng = df['high'] - df['low']
    m20 = rng.rolling(20, min_periods=10).mean()
    m60 = rng.rolling(60, min_periods=30).mean()
    return (m20 / m60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def gap_down_freq_20(df, s):
    prev_low = df['low'].shift(1)
    gap_down = (df['open'] < prev_low).astype(float)
    return gap_down.rolling(20, min_periods=10).mean()


def spread_beta_uscn_60(df, s):
    if us10y is None or cn10y is None:
        return None
    spread = us10y['close'] - cn10y['close']
    ds = spread.pct_change()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), ds.rename('s')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['s']) / z['s'].rolling(60, min_periods=30).var().replace(0, np.nan)
    return (b.replace([np.inf, -np.inf], np.nan)).reindex(z.index)


def up_capture_60(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0).rolling(60, min_periods=25).mean()
    dn = r.where(r < 0).rolling(60, min_periods=25).mean().abs()
    return (up / dn.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


candidates = {
    'semi_vol_ratio_60': semi_vol_ratio_60,
    'parkinson_ratio_20': parkinson_ratio_20,
    'trend_r2_60': trend_r2_60,
    'basket_corr_60': basket_corr_60,
    'min_ret_20d': min_ret_20d,
    'updown_vol_asym_20': updown_vol_asym_20,
    'cn10y_beta_60': cn10y_beta_60,
    'us10y_beta_60': us10y_beta_60,
    'gap_fill_ratio_20': gap_fill_ratio_20,
    'ret_skew_raw_60': ret_skew_raw_60,
    'range_squeeze_20_60': range_squeeze_20_60,
    'gap_down_freq_20': gap_down_freq_20,
    'spread_beta_uscn_60': spread_beta_uscn_60,
    'up_capture_60': up_capture_60,
}

library_panels = load_effective_artifact_panels()
print(f"effective artifact library: {sorted(library_panels.keys())} ({len(library_panels)} factors)")

results = {}
for fid, fn in candidates.items():
    try:
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
    except Exception as e:
        print(f"{fid}: ERROR {type(e).__name__}: {e}")
        print()

print("SUMMARY_TABLE")
for fid, (m, _) in sorted(results.items()):
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and m['max_abs_library_correlation'] < 0.5
    print(f"{fid:24s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} rho={m['max_abs_library_correlation']:.3f} vs {str(m['max_corr_library_id']):24s} cov={m['coverage_asset_days']:.2f} -> {'ADMIT' if ok else 'skip'}")
