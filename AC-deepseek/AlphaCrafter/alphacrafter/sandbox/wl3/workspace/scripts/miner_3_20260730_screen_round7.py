"""miner_3 2026-07-30 screening round 7: orthogonal beta/region + return-shape ideas.

Library now 13 effective factors. This round targets distinct economic drivers
not yet in the library:
  gold_beta_60     : XAU beta (safe-haven sensitivity)
  hsi_beta_60      : Hang Seng beta (HK/China equity channel)
  n225_beta_60     : Nikkei 225 beta (Japan equity channel)
  sx5e_beta_60     : Euro Stoxx 50 beta (EU equity channel)
  ndx_beta_60      : Nasdaq-100 beta (US growth/tech channel)
  sox_beta_60      : SOX beta (semiconductor cycle channel)
  close_loc_20     : mean((close-low)/(high-low),20) intraday closing location
  amplitude_20_60  : (20d high - 20d low)/mean(close,20) amplitude ratio
  dd_speed_20      : (close - 20d max)/20 drawdown speed
  streak_down_20   : max consecutive down days in 20d
  up_move_20       : mean(up-day ret)/vol20 (up-day intensity)
  trend_accel_20_60: normalized 20d return - normalized 60d return
  beta_resid_mom_20_60: SPX-beta residual cumulative return (20d)
  gap_mag_20       : mean(|open-prev close|/(high-low),20) gap magnitude
  overnight_ratio_20: std(overnight ret)/std(intraday ret) 20d
  down_close_loc_20: mean((close-low)/(high-low) | down days, 20)
  volume_beta_60   : beta(volume change, market volume change) 60d
  cornish_10       : Cornish-Fisher VaR skew-kurtosis adjustment 10d
  drawup_speed_20  : (close - 20d min)/20 recovery speed

NOTE: oil_beta_60 (WTI) and btc_beta_60 were previously quarantined
(btc rho 0.747 vs spx_beta_60; wti sign-inconsistent) -> skipped here by design.

Admission: |IC|>=0.007 and |ICIR|>=0.084 at h=10; library rho < 0.5.
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
print(f"loaded {len(prices)} assets; grid {grid.min().date()}..{grid.max().date()} n={len(grid)} "
      f"({time.time()-t0:.1f}s)", flush=True)


def fwd_ret_rank(h):
    cols = {}
    for s, df in prices.items():
        cols[s] = df['close'].shift(-h) / df['close'] - 1.0
    return pd.DataFrame(cols).reindex(grid).rank(axis=1)


fwd_rank = {h: fwd_ret_rank(h) for h in HORIZONS}
print("forward rank panels ready", flush=True)


def rowwise_pearson(a, b, min_valid=MIN_VALID):
    mask = np.isfinite(a) & np.isfinite(b)
    n = mask.sum(axis=1)
    ok = n >= min_valid
    if not ok.any():
        return np.full(len(a), np.nan)
    a = np.where(mask, a, 0.0)
    b = np.where(mask, b, 0.0)
    am = a.sum(1) / np.maximum(n, 1)
    bm = b.sum(1) / np.maximum(n, 1)
    ac = np.where(mask, a - am[:, None], 0.0)
    bc = np.where(mask, b - bm[:, None], 0.0)
    num = (ac * bc).sum(1)
    den = np.sqrt((ac ** 2).sum(1) * (bc ** 2).sum(1))
    r = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    r[~ok] = np.nan
    return r


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


# ---------- candidate definitions ----------
def beta_factor(anchor_name):
    anchor = prices.get(anchor_name)

    def fn(df, s):
        if anchor is None:
            return None
        r = df['close'].pct_change()
        a = anchor['close'].pct_change()
        z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
        b = z['r'].rolling(60, min_periods=30).cov(z['a']) / \
            z['a'].rolling(60, min_periods=30).var().replace(0, np.nan)
        return b.replace([np.inf, -np.inf], np.nan).reindex(z.index)
    return fn


def close_loc_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    loc = ((df['close'] - df['low']) / rng).replace([np.inf, -np.inf], np.nan)
    return loc.rolling(20, min_periods=10).mean()


def amplitude_20_60(df, s):
    h20 = df['high'].rolling(20, min_periods=10).max()
    l20 = df['low'].rolling(20, min_periods=10).min()
    c20 = df['close'].rolling(20, min_periods=10).mean().replace(0, np.nan)
    return ((h20 - l20) / c20).replace([np.inf, -np.inf], np.nan)


def dd_speed_20(df, s):
    c = df['close']
    return ((c - c.rolling(20, min_periods=10).max()) / 20.0)


def streak_down_20(df, s):
    r = df['close'].pct_change()
    neg = (r < 0).astype(int)
    grp = (neg == 0).cumsum()
    streak = neg.groupby(grp).cumsum()
    return streak.rolling(20, min_periods=10).max()


def up_move_20(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0).rolling(20, min_periods=10).mean()
    vol = r.rolling(20, min_periods=10).std().replace(0, np.nan)
    return (up / vol).replace([np.inf, -np.inf], np.nan)


def trend_accel_20_60(df, s):
    r = df['close'].pct_change()
    v20 = r.rolling(20, min_periods=10).std().replace(0, np.nan)
    v60 = r.rolling(60, min_periods=30).std().replace(0, np.nan)
    r20 = r.rolling(20, min_periods=10).mean() * np.sqrt(20)
    r60 = r.rolling(60, min_periods=30).mean() * np.sqrt(60)
    return ((r20 / v20) - (r60 / v60)).replace([np.inf, -np.inf], np.nan)


def beta_resid_mom_20_60(df, s, spx=None):
    if spx is None:
        return None
    r = df['close'].pct_change()
    mkt = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), mkt.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['m']) / \
        z['m'].rolling(60, min_periods=30).var().replace(0, np.nan)
    alpha = z['r'] - b * z['m']
    return alpha.rolling(20, min_periods=10).sum()


def gap_mag_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    gap = (df['open'] - df['close'].shift(1)).abs()
    return (gap / rng).rolling(20, min_periods=10).mean()


def overnight_ratio_20(df, s):
    on = df['open'] / df['close'].shift(1) - 1.0
    intr = df['close'] / df['open'] - 1.0
    o = on.rolling(20, min_periods=10).std()
    i = intr.rolling(20, min_periods=10).std().replace(0, np.nan)
    return (o / i).replace([np.inf, -np.inf], np.nan)


def down_close_loc_20(df, s):
    r = df['close'].pct_change()
    rng = (df['high'] - df['low']).replace(0, np.nan)
    loc = ((df['close'] - df['low']) / rng).replace([np.inf, -np.inf], np.nan)
    down_loc = loc.where(r < 0)
    return down_loc.rolling(20, min_periods=8).mean()


def volume_beta_60(df, s):
    v = df['volume'].pct_change()
    mktv = pd.concat([dd['volume'].pct_change().rename(sym) for sym, dd in prices.items()], axis=1)
    m = mktv.mean(axis=1)
    z = pd.concat([v.rename('v'), m.rename('m')], axis=1).dropna()
    b = z['v'].rolling(60, min_periods=30).cov(z['m']) / \
        z['m'].rolling(60, min_periods=30).var().replace(0, np.nan)
    return b.replace([np.inf, -np.inf], np.nan).reindex(z.index)


def cornish_10(df, s):
    r = df['close'].pct_change().rolling(10, min_periods=8)
    mu = r.mean()
    sd = r.std().replace(0, np.nan)
    sk = r.skew()
    ku = r.kurt()
    # Cornish-Fisher 5% VaR modifier: z = 1.645 + (1.645^2-1)sk/6 + (1.645^3-3*1.645)ku/24 - (2*1.645^3-5*1.645)sk^2/36
    z = 1.645 + (1.645 ** 2 - 1) * sk / 6.0 + (1.645 ** 3 - 3 * 1.645) * ku / 24.0 \
        - (2 * 1.645 ** 3 - 5 * 1.645) * sk ** 2 / 36.0
    return (mu - z * sd).replace([np.inf, -np.inf], np.nan)


def drawup_speed_20(df, s):
    c = df['close']
    return ((c - c.rolling(20, min_periods=10).min()) / 20.0)


spx = prices.get('SPX')

candidates = {
    'gold_beta_60': beta_factor('XAU'),
    'hsi_beta_60': beta_factor('HSI'),
    'n225_beta_60': beta_factor('N225'),
    'sx5e_beta_60': beta_factor('SX5E'),
    'ndx_beta_60': beta_factor('NDX'),
    'sox_beta_60': beta_factor('SOX'),
    'close_loc_20': close_loc_20,
    'amplitude_20_60': amplitude_20_60,
    'dd_speed_20': dd_speed_20,
    'streak_down_20': streak_down_20,
    'up_move_20': up_move_20,
    'trend_accel_20_60': trend_accel_20_60,
    'beta_resid_mom_20_60': lambda df, s: beta_resid_mom_20_60(df, s, spx),
    'gap_mag_20': gap_mag_20,
    'overnight_ratio_20': overnight_ratio_20,
    'down_close_loc_20': down_close_loc_20,
    'volume_beta_60': volume_beta_60,
    'cornish_10': cornish_10,
    'drawup_speed_20': drawup_speed_20,
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
