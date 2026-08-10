"""miner_3 2026-07-30 screening round 6: vectorized harness.

Round-6 fresh orthogonal ideas (distinct economic drivers, absent from library):
  amihud_illiq_20   : Amihud price-impact |ret|/volume (liquidity) - 20d mean
  vol_dispersion_20_60: |vol20 - vol60|/vol60 - volatility regime acceleration
  drawdown_depth_60 : close/rolling_max(close,60) - 1 - drawdown depth
  kurtosis_60       : excess kurtosis of 60d returns - tail-shape risk
  volume_ratio_5_60 : mean(vol,5)/mean(vol,60) - short-horizon volume surge
  upper_shadow_20   : mean((high-max(open,close))/range) - overhead supply
  lower_shadow_20   : mean((min(open,close)-low)/range) - support wick
  overnight_gap_20  : mean(open/prev_close - 1) - gap direction
  rsi_14            : classic RSI momentum/mean-reversion oscillator
  zero_ret_freq_20  : fraction of days with |ret| < 0.1% (staleness/liquidity)
  us10y_beta_20     : short-horizon sensitivity to US10Y yield changes
  real_range_20     : mean((high-low)/close) - level-normalized range
  ret_autocorr_10   : 10d return autocorrelation (continuation/reversal)
  hurst_vol_ratio_20_5: ln(vol20/vol5)/ln(4) - long-memory (Hurst) estimate

Also quickly re-confirms round-5 candidates with the fast harness.
Admission gates: |IC|>=0.007 and |ICIR|>=0.084 at 10d horizon on the shared
15-asset universe; pairwise library rho < 0.5 against 12 effective artifact
factors (max_library_correlation).
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, canonical_grid, WATCHLIST, VAL_START,
                           VAL_END, load_artifact_matrix, Path)

HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_VALID = 8
IC_TH = 0.007
ICIR_TH = 0.084
RHO_TH = 0.5

t0 = time.time()
prices = load_prices(days=3000)
grid = canonical_grid(prices)
print(f"loaded {len(prices)} assets; canonical grid {grid.min().date()}..{grid.max().date()} n={len(grid)} "
      f"({time.time()-t0:.1f}s)", flush=True)

# ---------------- forward return rank matrices (precomputed once) ------------
def fwd_ret_rank(h):
    cols = {}
    for s, df in prices.items():
        cols[s] = df['close'].shift(-h) / df['close'] - 1.0
    p = pd.DataFrame(cols).reindex(grid)
    return p.rank(axis=1)

fwd_rank = {h: fwd_ret_rank(h) for h in HORIZONS}
print("forward return rank panels ready", flush=True)


def rowwise_pearson(a, b, min_valid=MIN_VALID):
    """Vectorized row-wise Pearson between two (n,15) float arrays with NaN."""
    mask = np.isfinite(a) & np.isfinite(b)
    n = mask.sum(axis=1)
    ok = n >= min_valid
    if not ok.any():
        return np.full(len(a), np.nan)
    a = np.where(mask, a, 0.0)
    b = np.where(mask, b, 0.0)
    a_mean = a.sum(1) / np.maximum(n, 1)
    b_mean = b.sum(1) / np.maximum(n, 1)
    ac = np.where(mask, a - a_mean[:, None], 0.0)
    bc = np.where(mask, b - b_mean[:, None], 0.0)
    num = (ac * bc).sum(1)
    den = np.sqrt((ac ** 2).sum(1) * (bc ** 2).sum(1))
    r = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    r[~ok] = np.nan
    return r


# ---------------- library panels (12 effective artifact factors) -------------
def load_effective_library_rank():
    out = {}
    for jp in sorted(Path('factors').glob('*.json')):
        if jp.name == 'factor_ensemble.json':
            continue
        payload = json.loads(jp.read_text(encoding='utf-8'))
        if payload.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        art = load_artifact_matrix(str(jp))
        if art is None or art.shape[0] != len(grid) or art.shape[1] != 15:
            continue
        df = pd.DataFrame(art, index=grid, columns=WATCHLIST)
        out[payload['factor_id']] = df.rank(axis=1)
    return out

lib_rank = load_effective_library_rank()
print(f"effective artifact library for rho audit: {sorted(lib_rank.keys())} "
      f"({len(lib_rank)} factors)", flush=True)


def max_lib_rho(rank_panel):
    best, best_id = 0.0, None
    a = rank_panel.values
    for fid, lrp in lib_rank.items():
        r = rowwise_pearson(a, lrp.values)
        r = r[np.isfinite(r)]
        if len(r) == 0:
            continue
        rho = float(np.mean(r))
        if abs(rho) > best:
            best, best_id = abs(rho), fid
    return best, best_id


# ---------------- candidate definitions --------------------------------------
def amihud_illiq_20(df, s):
    r = df['close'].pct_change().abs()
    v = df['volume'].replace(0, np.nan)
    return (r / v).rolling(20, min_periods=10).mean()


def vol_dispersion_20_60(df, s):
    v20 = df['close'].pct_change().rolling(20, min_periods=10).std()
    v60 = df['close'].pct_change().rolling(60, min_periods=30).std()
    return ((v20 - v60).abs() / v60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def drawdown_depth_60(df, s):
    c = df['close']
    return c / c.rolling(60, min_periods=30).max() - 1.0


def kurtosis_60(df, s):
    return df['close'].pct_change().rolling(60, min_periods=30).kurt()


def volume_ratio_5_60(df, s):
    v = df['volume']
    m5 = v.rolling(5, min_periods=3).mean()
    m60 = v.rolling(60, min_periods=30).mean()
    return (m5 / m60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def upper_shadow_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    shadow = (df['high'] - df[['open', 'close']].max(axis=1)) / rng
    return shadow.rolling(20, min_periods=10).mean()


def lower_shadow_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    shadow = (df[['open', 'close']].min(axis=1) - df['low']) / rng
    return shadow.rolling(20, min_periods=10).mean()


def overnight_gap_20(df, s):
    g = df['open'] / df['close'].shift(1) - 1.0
    return g.rolling(20, min_periods=10).mean()


def rsi_14(df, s):
    r = df['close'].pct_change()
    up = r.clip(lower=0).rolling(14, min_periods=7).mean()
    dn = (-r).clip(lower=0).rolling(14, min_periods=7).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).replace([np.inf, -np.inf], np.nan)


def zero_ret_freq_20(df, s):
    r = df['close'].pct_change()
    return (r.abs() < 0.001).astype(float).rolling(20, min_periods=10).mean()


def us10y_beta_20(df, s, anchor=None):
    if anchor is None:
        return None
    r = df['close'].pct_change()
    a = anchor['close'].pct_change()
    z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
    b = z['r'].rolling(20, min_periods=10).cov(z['a']) / z['a'].rolling(20, min_periods=10).var().replace(0, np.nan)
    return b.replace([np.inf, -np.inf], np.nan).reindex(z.index)


def real_range_20(df, s):
    rng = df['high'] - df['low']
    c = df['close'].replace(0, np.nan)
    return (rng / c).rolling(20, min_periods=10).mean()


def ret_autocorr_10(df, s):
    r = df['close'].pct_change()
    return r.rolling(10, min_periods=6).apply(lambda w: w.autocorr() if len(w) > 2 else np.nan, raw=False)


def hurst_vol_ratio_20_5(df, s):
    v5 = df['close'].pct_change().rolling(5, min_periods=3).std()
    v20 = df['close'].pct_change().rolling(20, min_periods=10).std()
    return np.log(v20 / v5.replace(0, np.nan)) / np.log(4.0)


# round-5 fast confirmation set
def semi_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    dv = r.where(r < 0).rolling(60, min_periods=25).std()
    uv = r.where(r > 0).rolling(60, min_periods=25).std()
    return (dv / uv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def parkinson_ratio_20(df, s):
    c = df['close'].replace(0, np.nan)
    park = np.sqrt((np.log(df['high'] / df['low'].replace(0, np.nan)) ** 2)
                   .rolling(20, min_periods=10).mean() / (4 * np.log(2)))
    cc = c.pct_change().rolling(20, min_periods=10).std()
    return (park / cc.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def basket_corr_60(df, s):
    r = df['close'].pct_change()
    b = pd.Series({d: r.loc[:d].tail(60).mean() for d in r.index[59::5]}, dtype=float)
    return None  # placeholder, replaced below


def min_ret_20d(df, s):
    return df['close'].pct_change().rolling(20, min_periods=10).min()


def updown_vol_asym_20(df, s):
    r = df['close'].pct_change()
    dv = r.where(r < 0).rolling(20, min_periods=8).std()
    uv = r.where(r > 0).rolling(20, min_periods=8).std()
    tot = r.rolling(20, min_periods=8).std().replace(0, np.nan)
    return ((uv - dv) / tot).replace([np.inf, -np.inf], np.nan)


def gap_fill_ratio_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return ((df['close'] - df['open']) / rng).rolling(20, min_periods=10).mean()


def ret_skew_raw_60(df, s):
    return df['close'].pct_change().rolling(60, min_periods=30).skew()


def range_squeeze_20_60(df, s):
    rng = df['high'] - df['low']
    return (rng.rolling(20, min_periods=10).mean() /
            rng.rolling(60, min_periods=30).mean().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def gap_down_freq_20(df, s):
    g = (df['open'] < df['low'].shift(1)).astype(float)
    return g.rolling(20, min_periods=10).mean()


def up_capture_60(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0).rolling(60, min_periods=25).mean()
    dn = r.where(r < 0).rolling(60, min_periods=25).mean().abs()
    return (up / dn.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


us10y = prices.get('US10Y')


def basket_corr_60_real(df, s):
    r = df['close'].pct_change()
    bb = pd.concat([r.rename('r'),
                    pd.Series({d: r.loc[:d].tail(60).mean() for d in r.index}, dtype=float).rename('b')], axis=1)
    return bb['r'].rolling(60, min_periods=30).corr(bb['b'])


candidates = {
    # round-6 fresh
    'amihud_illiq_20': amihud_illiq_20,
    'vol_dispersion_20_60': vol_dispersion_20_60,
    'drawdown_depth_60': drawdown_depth_60,
    'kurtosis_60': kurtosis_60,
    'volume_ratio_5_60': volume_ratio_5_60,
    'upper_shadow_20': upper_shadow_20,
    'lower_shadow_20': lower_shadow_20,
    'overnight_gap_20': overnight_gap_20,
    'rsi_14': rsi_14,
    'zero_ret_freq_20': zero_ret_freq_20,
    'us10y_beta_20': lambda df, s: us10y_beta_20(df, s, us10y),
    'real_range_20': real_range_20,
    'ret_autocorr_10': ret_autocorr_10,
    'hurst_vol_ratio_20_5': hurst_vol_ratio_20_5,
    # round-5 fast confirmation
    'r5_semi_vol_ratio_60': semi_vol_ratio_60,
    'r5_parkinson_ratio_20': parkinson_ratio_20,
    'r5_basket_corr_60': basket_corr_60_real,
    'r5_min_ret_20d': min_ret_20d,
    'r5_updown_vol_asym_20': updown_vol_asym_20,
    'r5_gap_fill_ratio_20': gap_fill_ratio_20,
    'r5_ret_skew_raw_60': ret_skew_raw_60,
    'r5_range_squeeze_20_60': range_squeeze_20_60,
    'r5_gap_down_freq_20': gap_down_freq_20,
    'r5_up_capture_60': up_capture_60,
}


def factor_to_rank_panel(fn):
    cols = {}
    for s, df in prices.items():
        try:
            ser = fn(df, s)
            if ser is not None and len(ser) > 0:
                cols[s] = ser.astype(float)
        except Exception:
            pass
    panel = pd.DataFrame(cols)
    panel = panel[~panel.index.duplicated(keep='last')].sort_index()
    return panel.reindex(grid).rank(axis=1), panel


def validate_fast(rank_panel, raw_panel):
    ic_series = {}
    for h in HORIZONS:
        ic_series[h] = rowwise_pearson(rank_panel.values, fwd_rank[h].values)
    ic10 = ic_series[10]
    ic10 = ic10[(grid >= VAL_START) & (grid <= VAL_END)]
    ic10 = ic10[np.isfinite(ic10)]
    if len(ic10) < 100:
        return None
    ic_mean = float(ic10.mean())
    ic_std = float(ic10.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())
    fac = raw_panel[(raw_panel.index >= VAL_START) & (raw_panel.index <= VAL_END)]
    total_cells = fac.shape[0] * fac.shape[1]
    valid_cells = int(fac.notna().sum().sum())
    coverage = valid_cells / total_cells if total_cells else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= MIN_VALID).mean())
    ranked = fac.rank(axis=1)
    turnover = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    decay = {}
    for h in HORIZONS:
        s = ic_series[h]
        s = s[np.isfinite(s)]
        decay[str(h)] = float(s.mean()) if len(s) else float('nan')
    return {'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit,
            'n_ic_dates': int(len(ic10)), 'coverage_asset_days': coverage,
            'coverage_dates_ge8': ge8, 'turnover_10d_rank': turnover,
            'decay_ic_by_horizon': decay}


results = {}
for fid, fn in candidates.items():
    t1 = time.time()
    try:
        rank_panel, raw_panel = factor_to_rank_panel(fn)
        m = validate_fast(rank_panel, raw_panel)
        if m is None:
            print(f'{fid}: insufficient data -> None ({time.time()-t1:.1f}s)', flush=True)
            continue
        ok_ic = abs(m['ic']) >= IC_TH and abs(m['icir']) >= ICIR_TH
        rho, best = (max_lib_rho(rank_panel) if ok_ic else (float('nan'), None))
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = best
        results[fid] = (m, rank_panel, raw_panel)
        admit = ok_ic and rho < RHO_TH
        print(f'{fid}: ic={m["ic"]:+.4f} icir={m["icir"]:+.4f} hit={m["ic_hit_ratio"]:.3f} '
              f'n={m["n_ic_dates"]} cov={m["coverage_asset_days"]:.3f} ge8={m["coverage_dates_ge8"]:.3f} '
              f'turn={m["turnover_10d_rank"]:.2f} rho={rho:.3f} vs {best} '
              f'({"ADMIT" if admit else "skip"}) [{time.time()-t1:.1f}s]', flush=True)
    except Exception as e:
        print(f'{fid}: ERROR {type(e).__name__}: {e}', flush=True)

print('\nDECAY_TABLE')
for fid, (m, _, _) in results.items():
    print(f'{fid:24s} ' + ' '.join(f'h{h}:{m["decay_ic_by_horizon"][str(h)]:+.4f}' for h in HORIZONS))

print('\nSUMMARY_TABLE')
for fid, (m, _, _) in sorted(results.items()):
    admit = abs(m['ic']) >= IC_TH and abs(m['icir']) >= ICIR_TH and m['max_abs_library_correlation'] < RHO_TH
    print(f'{fid:24s} ic={m["ic"]:+.4f} icir={m["icir"]:+.4f} '
          f'rho={m["max_abs_library_correlation"]:.3f} vs {str(m["max_corr_library_id"]):22s} '
          f'-> {"ADMIT" if admit else "skip"}')
print(f'\nTOTAL {time.time()-t0:.1f}s')
