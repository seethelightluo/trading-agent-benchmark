"""miner_1 2026-07-30: canonical-grid rebuild + deterministic pairwise gate simulation.

Rebuilds every candidate factor panel on ONE canonical grid (shared by the gate),
recomputes IC/ICIR (h=10, window 2020-01-01..2026-07-15) + recent-1y drift stats,
computes the full pairwise mean-daily-Spearman correlation matrix among
gate-passing factors, applies the worldline_pairwise_signal_quality_v1 greedy
admission (sort by |ICIR| desc, quarantine rho>=0.5 duplicates), then persists
admitted factors WITH canonical-grid signal artifacts and fresh rho provenance.
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, persist_factor,
                           canonical_grid, WATCHLIST, VAL_START, VAL_END)

t0 = time.time()
prices = load_prices(days=2600)
dxy = load_index('DXY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
vix = load_index('VIX', prices=prices)
grid = canonical_grid(prices)
print(f'[load] {len(prices)} assets; canonical grid {len(grid)} dates '
      f'{grid.min().date()}..{grid.max().date()}', flush=True)

# ---------------- factor definitions (expressions mirrored from persisted JSONs) ----------------
def f_bollinger_z(df, s):
    c = df['close']; return (c - c.rolling(20).mean()) / c.rolling(20).std()
def f_brkout(df, s):
    prior_high = df['high'].rolling(60).max().shift(1)
    return (df['close'] > prior_high).astype(float).rolling(20).mean()
def _beta_anchor(anchor_close):
    def f(df, s):
        a = anchor_close.reindex(df.index)
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['a']) / z['a'].rolling(60).var()
        return b.reindex(df.index)
    return f
def _beta_cond(anchor_close, sign=1.0):
    def f(df, s):
        a = anchor_close.reindex(df.index)
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), a.rename('a')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['a']) / z['a'].rolling(60).var()
        mom = (anchor_close / anchor_close.shift(20) - 1.0).reindex(df.index)
        return (sign * b * mom).reindex(df.index)
    return f
def f_hilo(df, s, w):
    hi = df['high'].rolling(w).max(); lo = df['low'].rolling(w).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)
def f_max_ret(df, s):
    return df['close'].pct_change().rolling(20).max()
def f_mom(df, s, lb, sk):
    return df['close'].shift(sk) / df['close'].shift(sk + lb) - 1.0
def f_rsi(df, s):
    c = df['close']; d = c.diff()
    up = d.clip(lower=0.0); dn = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1 / 14, adjust=False).mean()
    rd = dn.ewm(alpha=1 / 14, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return 100 - 100 / (1 + rs)
def f_skew_term(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).skew() - r.rolling(60).skew()
def f_upper_wick(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    return ((df['high'] - np.maximum(df['open'], df['close'])) / rng).rolling(10).mean()
def f_vixbeta(df, s):
    return _beta_cond(vix['close'], sign=-1.0)(df, s)
def f_vol_adj_mom(df, s):
    m = df['close'].shift(5) / df['close'].shift(25) - 1.0
    v = df['close'].pct_change().rolling(60).std()
    return m / v.replace(0, np.nan)
def f_vol_of_vol(df, s):
    return df['close'].pct_change().rolling(20).std().rolling(60).std()
def f_vol_price_corr(df, s):
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=30).corr(df['volume'])
def f_streak(df, s):
    r = df['close'].pct_change()
    up = (r > 0).astype(float)
    grp = (up != up.shift()).cumsum()
    cnt = up.groupby(grp).cumcount() + 1
    return cnt.where(up == 1, -cnt)
def f_vol_shock(df, s):
    v = df['volume']
    return v / v.rolling(20).mean() - 1.0
def f_amihud(df, s):
    r = df['close'].pct_change().abs()
    v = df['volume'].replace(0, np.nan)
    return (r / v).rolling(60).mean()

DEPRECATED = {'upper_wick_10', 'vix_beta_cond_60x20', 'vol_of_vol20x60'}
QUARANTINED = {'bollinger_z_20d', 'btc_beta_60', 'high_low_range_pos_20'}


def f_bw_zscore(df, s):
    c = df['close']; ma = c.rolling(20).mean(); sd = c.rolling(20).std()
    bw = 2 * sd / ma
    mu = bw.rolling(60).mean(); s60 = bw.rolling(60).std()
    return (bw - mu) / s60.replace(0, np.nan)

FACTOR_DEFS = {
    'brkout_60_20':         (f_brkout, {'lookback': 60, 'window': 20}, 'rolling_mean((close > shift(1, rolling_max(high,60))).astype(float), 20)'),
    'bw_zscore_20_60':      (f_bw_zscore, {'window': 20, 'norm_window': 60}, 'zscore60(2*STD20(close)/SMA20(close))'),
    'dxy_beta_cond_60x20':  (_beta_cond(dxy['close']), {'beta_window': 60, 'dxy_lookback': 20}, 'beta(ret,DXY_ret,60)*(DXY/DXY.shift(20)-1)'),
    'eurusd_beta_cond_60x20': (_beta_cond(eurusd['close']), {'beta_window': 60, 'move_window': 20}, 'BETA(ret,EURUSD_ret,60)*(EURUSD/EURUSD.shift(20)-1)'),
    'hilo_pos_20d':         (lambda df, s: f_hilo(df, s, 20), {'window': 20}, 'range-position 20d'),
    'hilo_pos_60':          (lambda df, s: f_hilo(df, s, 60), {'window': 60}, 'range-position 60d'),
    'hilo_pos_120':         (lambda df, s: f_hilo(df, s, 120), {'window': 120}, 'range-position 120d'),
    'hs300_beta_60':        (_beta_anchor(prices['000300.SH']['close']), {'anchor': '000300.SH', 'window': 60}, 'BETA(ret,000300.SH_ret,60)'),
    'max_ret_20d':          (f_max_ret, {'window': 20}, 'max20(pct_change(close))'),
    'mom_10d_skip5':        (lambda df, s: f_mom(df, s, 10, 5), {'lookback': 10, 'skip': 5}, 'close.shift(5)/close.shift(15)-1'),
    'mom_120d_skip5':       (lambda df, s: f_mom(df, s, 120, 5), {'lookback': 120, 'skip': 5}, 'close.shift(5)/close.shift(125)-1'),
    'rsi_14d':              (f_rsi, {'lookback': 14}, 'Wilder RSI-14'),
    'skew_term_20_60':      (f_skew_term, {'short': 20, 'long': 60}, 'skew20(ret)-skew60(ret)'),
    'spx_beta_60':          (_beta_anchor(prices['SPX']['close']), {'beta_window': 60}, 'BETA(ret,SPX_ret,60)'),
    'vol_adj_mom_20_60':    (f_vol_adj_mom, {'lookback': 20, 'skip': 5, 'vol_window': 60}, '(mom20_skip5)/STD60(ret)'),
    'vol_price_corr_60':    (f_vol_price_corr, {'window': 60, 'min_periods': 30}, 'corr(pct_change(close),volume,60)'),
    'wti_beta_60':          (_beta_anchor(prices['WTI']['close']), {'anchor': 'WTI', 'window': 60}, 'BETA(ret,WTI_ret,60)'),
    # reference-only
    'bollinger_z_20d':      (f_bollinger_z, {'window': 20}, 'zscore20(close)'),
    'btc_beta_60':          (_beta_anchor(prices['BTC']['close']), {'beta_window': 60}, 'BETA(ret,BTC_ret,60)'),
    'high_low_range_pos_20': (lambda df, s: f_hilo(df, s, 20), {'window': 20}, 'range-pos 20d'),
    'upper_wick_10':        (f_upper_wick, {'window': 10}, 'mean10(upper-wick ratio)'),
    'vix_beta_cond_60x20':  (f_vixbeta, {'beta_window': 60, 'move_window': 20}, '-beta(ret,VIX_ret,60)*(VIX/VIX.shift(20)-1)'),
    'vol_of_vol20x60':      (f_vol_of_vol, {'short_win': 20, 'long_win': 60}, 'STD60(STD20(ret))'),
    # new candidates
    'streak_signed':        (f_streak, {'cap': None}, 'signed consecutive up/down streak'),
    'vol_shock_20':         (f_vol_shock, {'window': 20}, 'volume/SMA20(volume)-1'),
    'amihud_60':            (f_amihud, {'window': 60}, 'mean(|ret|/volume,60)'),
}

# expected_direction from existing JSONs (fallback +1)
DIR = {}
for fid in FACTOR_DEFS:
    try:
        d = json.load(open(f'factors/{fid}.json'))
        DIR[fid] = d.get('expected_direction', 1)
    except Exception:
        DIR[fid] = 1
DIR['streak_signed'] = 1
DIR['vol_shock_20'] = 1
DIR['amihud_60'] = -1

# ---------------- panels + validation ----------------
fwd = {h: pd.DataFrame({s: df['close'].shift(-h) / df['close'] - 1.0
                        for s, df in prices.items()}).sort_index()
       for h in (1, 2, 3, 5, 10, 20)}

def rank_ic_series_fast(panel, fr, min_valid=8):
    df = pd.concat({'x': panel, 'y': fr}, axis=1).sort_index()
    x = df['x'].rank(axis=1); y = df['y'].rank(axis=1)
    valid = df['x'].notna() & df['y'].notna()
    n = valid.sum(axis=1)
    x = x.where(valid); y = y.where(valid)
    mx = x.mean(axis=1); my = y.mean(axis=1)
    cov = ((x.sub(mx, axis=0)) * (y.sub(my, axis=0))).sum(axis=1) / (n - 1)
    vx = ((x.sub(mx, axis=0)) ** 2).sum(axis=1) / (n - 1)
    vy = ((y.sub(my, axis=0)) ** 2).sum(axis=1) / (n - 1)
    ic = cov / np.sqrt(vx * vy)
    return ic.where((n >= min_valid) & (vx > 0) & (vy > 0)).dropna()

def validate(panel, rec_start=pd.Timestamp('2025-07-15')):
    ic_s = {h: rank_ic_series_fast(panel, fwd[h]) for h in fwd}
    ic10 = ic_s[10]
    full = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    rec = ic10[(ic10.index >= rec_start) & (ic10.index <= VAL_END)]
    def stat(s):
        m = float(s.mean()); sd = float(s.std(ddof=1))
        return m, (m / sd if sd > 0 else 0.0), int(len(s))
    ic_m, icir_m, n_m = stat(full)
    ic_r, icir_r, n_r = stat(rec) if len(rec) > 30 else (float('nan'), float('nan'), 0)
    hit = float((full > 0).mean()) if ic_m >= 0 else float((full < 0).mean())
    fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    cov = float(fac.notna().sum().sum()) / (fac.shape[0] * fac.shape[1])
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    decay = {str(h): float(ic_s[h].mean()) for h in fwd}
    return dict(ic=ic_m, icir=icir_m, ic_hit=hit, n=n_m, cov=cov, ge8=ge8, turn=turn,
                decay=decay, ic_rec=ic_r, icir_rec=icir_r, n_rec=n_r)

panels, stats = {}, {}
for fid, (fn, params, expr) in FACTOR_DEFS.items():
    panel = pd.DataFrame({s: fn(prices[s], s) for s in WATCHLIST})
    panel = panel[~panel.index.duplicated(keep='last')].sort_index()
    panels[fid] = panel
    stats[fid] = validate(panel)
    m = stats[fid]
    print(f'{fid:24s} IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} hit={m["ic_hit"]:.3f} '
          f'n={m["n"]} cov={m["cov"]:.2f} ge8={m["ge8"]:.2f} turn={m["turn"]:.2f} '
          f'| rec1y IC={m["ic_rec"]:+.4f} ICIR={m["icir_rec"]:+.4f} n={m["n_rec"]}', flush=True)

# ---------------- pairwise rho matrix on canonical grid ----------------
def pairwise_rho(ids):
    R = {fid: panels[fid].rank(axis=1) for fid in ids}
    M = pd.DataFrame(index=ids, columns=ids, dtype=float)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            xa, xb = R[a], R[b]
            valid = xa.notna() & xb.notna()
            n = valid.sum(axis=1)
            x1 = xa.where(valid); x2 = xb.where(valid)
            m1 = x1.mean(axis=1); m2 = x2.mean(axis=1)
            cov = ((x1.sub(m1, axis=0)) * (x2.sub(m2, axis=0))).sum(axis=1) / (n - 1)
            v1 = ((x1.sub(m1, axis=0)) ** 2).sum(axis=1) / (n - 1)
            v2 = ((x2.sub(m2, axis=0)) ** 2).sum(axis=1) / (n - 1)
            c = (cov / np.sqrt(v1 * v2)).where((n >= 8) & (v1 > 0) & (v2 > 0))
            rho = float(c.mean()) if len(c) else float('nan')
            M.loc[a, b] = M.loc[b, a] = rho
    return M

ids_all = list(FACTOR_DEFS.keys())
M = pairwise_rho(ids_all)
print('\n===== pairwise mean daily cross-sectional Spearman rho (canonical grid) =====', flush=True)
pd.set_option('display.width', 260)
pd.set_option('display.max_columns', 40)
print(M.round(2).to_string())
print('\n--- pairs with |rho| >= 0.5 (redundancy conflicts) ---')
for a in ids_all:
    for b in ids_all:
        if a < b and abs(M.loc[a, b]) >= 0.5:
            print(f'  {a:24s} <-> {b:24s} rho={M.loc[a,b]:+.3f}')

# ---------------- greedy admission (worldline_pairwise_signal_quality_v1) ----------------
cands = [fid for fid in FACTOR_DEFS if fid not in DEPRECATED and fid not in QUARANTINED]
cands = [fid for fid in cands if abs(stats[fid]['ic']) >= 0.007 and abs(stats[fid]['icir']) >= 0.084]
cands.sort(key=lambda f: (-abs(stats[f]['icir']), -abs(stats[f]['ic'])))
admitted, quarantined, failed = [], [], []
for fid in cands:
    rho_max = max((abs(M.loc[fid, x]) for x in admitted), default=0.0)
    if rho_max < 0.5:
        admitted.append(fid)
    else:
        quarantined.append((fid, rho_max))
for fid in FACTOR_DEFS:
    if fid in DEPRECATED or fid in QUARANTINED:
        continue
    if abs(stats[fid]['ic']) < 0.007 or abs(stats[fid]['icir']) < 0.084:
        failed.append(fid)

print('\n===== GATE DECISION =====')
print('ADMITTED (pass IC/ICIR + rho<0.5):')
for fid in admitted:
    rho_lib = max((abs(M.loc[fid, x]) for x in admitted if x != fid), default=0.0)
    print(f'  {fid:24s} IC={stats[fid]["ic"]:+.4f} ICIR={stats[fid]["icir"]:+.4f} rho_vs_lib={rho_lib:.3f} rec1y_ICIR={stats[fid]["icir_rec"]:+.4f}')
print('QUARANTINE (rho>=0.5 vs higher-|ICIR| admitted):')
for fid, rho in quarantined:
    print(f'  {fid:24s} rho={rho:.3f}')
print('FAILED IC/ICIR gate:')
for fid in failed:
    print(f'  {fid:24s} IC={stats[fid]["ic"]:+.4f} ICIR={stats[fid]["icir"]:+.4f}')
print('Reference (deprecated/quarantined by others):', sorted(set(DEPRECATED) | set(QUARANTINED)))

# ---------------- persist admitted with canonical-grid artifacts ----------------
print('\n===== PERSIST =====')
NAMES = {'brkout_60_20': 'Breakout Fraction 60x20', 'bw_zscore_20_60': 'Bollinger Bandwidth Z 20x60',
         'dxy_beta_cond_60x20': 'DXY-beta Conditional 60x20', 'eurusd_beta_cond_60x20': 'EURUSD-beta Conditional 60x20',
         'hilo_pos_20d': 'HiLo Range Position 20d', 'hilo_pos_60': 'HiLo Range Position 60d',
         'hilo_pos_120': 'HiLo Range Position 120d', 'hs300_beta_60': 'HS300-beta 60d',
         'max_ret_20d': 'Max Daily Return 20d', 'mom_10d_skip5': 'Momentum 10d skip5',
         'mom_120d_skip5': 'Momentum 120d skip5', 'rsi_14d': 'RSI 14d',
         'skew_term_20_60': 'Skew Term Structure 20-60', 'spx_beta_60': 'SPX-beta 60d',
         'vol_adj_mom_20_60': 'Vol-Adjusted Momentum 20x60', 'vol_price_corr_60': 'Return-Volume Corr 60d',
         'wti_beta_60': 'WTI-beta 60d', 'streak_signed': 'Signed Streak Length',
         'vol_shock_20': 'Volume Shock 20d', 'amihud_60': 'Amihud Illiquidity 60d'}
TAGS = {'brkout_60_20': ['breakout', 'trend'], 'bw_zscore_20_60': ['volatility', 'regime'],
        'dxy_beta_cond_60x20': ['macro', 'fx_beta'], 'eurusd_beta_cond_60x20': ['macro', 'fx_beta'],
        'hilo_pos_20d': ['range-position', 'trend'], 'hilo_pos_60': ['range-position', 'trend'],
        'hilo_pos_120': ['range-position', 'trend'], 'hs300_beta_60': ['beta', 'china'],
        'max_ret_20d': ['momentum', 'tail'], 'mom_10d_skip5': ['momentum'],
        'mom_120d_skip5': ['momentum', 'trend'], 'rsi_14d': ['mean_reversion', 'technical'],
        'skew_term_20_60': ['skewness', 'tail-risk'], 'spx_beta_60': ['beta', 'risk-on'],
        'vol_adj_mom_20_60': ['momentum', 'vol-adjusted'], 'vol_price_corr_60': ['volume', 'liquidity'],
        'wti_beta_60': ['beta', 'commodity'], 'streak_signed': ['microstructure', 'momentum'],
        'vol_shock_20': ['volume', 'liquidity'], 'amihud_60': ['liquidity', 'microstructure']}
DEPS = {'brkout_60_20': ['close', 'high'], 'bw_zscore_20_60': ['close'], 'dxy_beta_cond_60x20': ['close', 'DXY'],
        'eurusd_beta_cond_60x20': ['close', 'EURUSD'], 'hilo_pos_20d': ['close', 'high', 'low'],
        'hilo_pos_60': ['close', 'high', 'low'], 'hilo_pos_120': ['close', 'high', 'low'],
        'hs300_beta_60': ['close', '000300.SH'], 'max_ret_20d': ['close'], 'mom_10d_skip5': ['close'],
        'mom_120d_skip5': ['close'], 'rsi_14d': ['close'], 'skew_term_20_60': ['close'],
        'spx_beta_60': ['close', 'SPX'], 'vol_adj_mom_20_60': ['close'], 'vol_price_corr_60': ['close', 'volume'],
        'wti_beta_60': ['close', 'WTI'], 'streak_signed': ['close'], 'vol_shock_20': ['volume'],
        'amihud_60': ['close', 'volume']}

persisted = []
for fid in admitted:
    m = stats[fid]
    rho_lib = max((abs(M.loc[fid, x]) for x in admitted if x != fid), default=0.0)
    rho_id = max((x for x in admitted if x != fid),
                 key=lambda x: abs(M.loc[fid, x]), default=None)
    metrics = {
        'ic': m['ic'], 'icir': m['icir'], 'ic_hit_ratio': m['ic_hit'],
        'n_ic_dates': m['n'], 'coverage_asset_days': m['cov'],
        'coverage_dates_ge8': m['ge8'], 'turnover_10d_rank': m['turn'],
        'decay_ic_by_horizon': m['decay'],
        'recent_1y_ic': m['ic_rec'], 'recent_1y_icir': m['icir_rec'],
        'max_abs_library_correlation': rho_lib,
        'max_corr_library_id': rho_id,
    }
    regime = ('2020-01..2026-07 warm-up; cross-asset regimes (COVID, 2022 tightening, '
              '2023-25 risk-on, crypto cycles). Canonical-grid rebuild 2026-07-30.')
    path, arr = persist_factor(
        factor_id=fid, factor_name=NAMES.get(fid, fid), expression=FACTOR_DEFS[fid][2],
        description=f'Rebuilt on shared canonical grid {len(grid)} dates; gate: |IC|>=0.007 |ICIR|>=0.084 rho<0.5.',
        dependencies=DEPS.get(fid, ['close']), parameters=FACTOR_DEFS[fid][1],
        expected_direction=DIR[fid], panel=panels[fid], metrics=metrics, tags=TAGS.get(fid, []),
        grid=grid, prices=prices, version='2.0.0', status='EFFECTIVE', regime_notes=regime)
    print(f'  wrote {fid}: IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} rho={rho_lib:.3f} art={arr.shape}')
    persisted.append(fid)

# ---------------- read-back verification ----------------
print('\n===== READ-BACK VERIFICATION =====')
ok_all = True
for fid in persisted:
    d = json.load(open(f'factors/{fid}.json'))
    art = np.load(f'factors/{fid}_signal.npy')
    mm = d['validation']['metrics']
    chk = (d['factor_id'] == fid and d['validation']['status'] == 'EFFECTIVE'
           and abs(mm['ic']) >= 0.007 and abs(mm['icir']) >= 0.084
           and art.shape == (len(grid), 15))
    ok_all &= chk
    print(f'  {"OK " if chk else "BAD"} {fid}: status={d["validation"]["status"]} '
          f'IC={mm["ic"]:.4f} ICIR={mm["icir"]:.4f} rho={mm["max_abs_library_correlation"]:.3f} '
          f'art={art.shape} lv={d["validation"]["last_validated"]}')
print('\nADMITTED:', persisted)
print('QUARANTINE_RECOMMEND:', [f for f, _ in quarantined])
print(f'[total] {time.time()-t0:.1f}s')
