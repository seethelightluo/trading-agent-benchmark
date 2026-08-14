"""miner_1 2035-10-11: screen novel factor candidates on the 15-asset universe.

Validation window matches library: 2020-01-01..2026-07-15 (research warm-up).
Gates: |IC|>=0.007, |ICIR|>=0.084 at h=10. Correlation vs 22-factor library
computed from persisted .npy signal artifacts on the canonical grid.
"""
import json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
VAL_START = pd.Timestamp('2020-01-01')
VAL_END = pd.Timestamp('2026-07-15')

def load_prices(days=6000):
    out = {}
    for s in WATCHLIST:
        try:
            df = get_stock_daily_data(symbol=s, days=days)
            if df is not None and len(df) >= 30:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                for c in ['open', 'close', 'high', 'low', 'volume']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                out[s] = df
        except Exception as e:
            print("load fail", s, e)
    return out

def load_index(symbol, prices, days=6000):
    try:
        df = get_index_daily_data(symbol=symbol, days=days)
        if df is not None and len(df) >= 30:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            return df
    except Exception:
        pass
    path = Path('../persistent/index_data') / f'{symbol}.csv'
    df = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    visible = max(dd.index.max() for dd in prices.values())
    return df[df.index <= visible]

def canonical_grid(prices):
    idx = set()
    for s, df in prices.items():
        idx.update(df.index)
    grid = pd.DatetimeIndex(sorted(idx))
    return grid[(grid >= VAL_START) & (grid <= VAL_END)]

def factor_to_panel(factor_fn, prices):
    cols = {}
    for s, df in prices.items():
        try:
            ser = factor_fn(df, s)
            if ser is not None and len(ser) > 0:
                cols[s] = ser.astype(float)
        except Exception:
            pass
    panel = pd.DataFrame(cols)
    return panel[~panel.index.duplicated(keep='last')].sort_index()

def forward_returns(prices, horizon):
    cols = {}
    for s, df in prices.items():
        if 'close' in df:
            cols[s] = df['close'].shift(-horizon) / df['close'] - 1.0
    return pd.DataFrame(cols).sort_index()

def rank_ic_series(factor_panel, fwd_ret, min_valid=8):
    common = factor_panel.index.intersection(fwd_ret.index)
    ic = {}
    for d in common:
        x = factor_panel.loc[d]; y = fwd_ret.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            ic[d] = x[m].rank().corr(y[m].rank())
    return pd.Series(ic).sort_index()

def validate(factor_panel, prices, min_valid=8):
    fwd10 = forward_returns(prices, 10)
    ic10 = rank_ic_series(factor_panel, fwd10, min_valid)
    ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    if len(ic10) < 100:
        return None
    ic_mean = float(ic10.mean())
    ic_std = float(ic10.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())
    fac = factor_panel[(factor_panel.index >= VAL_START) & (factor_panel.index <= VAL_END)]
    total_cells = fac.shape[0] * fac.shape[1]
    coverage = float(fac.notna().sum().sum()) / total_cells if total_cells else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= min_valid).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    # decay
    decay = {}
    for h in (1, 2, 3, 5, 10, 20):
        fh = forward_returns(prices, h)
        ics = rank_ic_series(factor_panel, fh, min_valid)
        ics = ics[(ics.index >= VAL_START) & (ics.index <= VAL_END)]
        decay[str(h)] = float(ics.mean()) if len(ics) else float('nan')
    return {'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit, 'n_ic_dates': int(len(ic10)),
            'coverage_asset_days': coverage, 'coverage_dates_ge8': ge8,
            'turnover_10d_rank': turn, 'decay_ic_by_horizon': decay}

def load_library_matrices(grid):
    """Load all EFFECTIVE factor .npy artifacts aligned to canonical grid."""
    mats, ids = {}, []
    for p in sorted(glob.glob('factors/*.json')):
        if '.bak' in p:
            continue
        try:
            d = json.load(open(p))
            if d.get('validation', {}).get('status') != 'EFFECTIVE':
                continue
            art = d.get('signal_artifact')
            if not art:
                continue
            arr = np.load(Path('factors') / art)
            if arr.shape != (len(grid), 15):
                # try to align: artifact may be on a slightly different grid
                continue
            mats[d['factor_id']] = arr
        except Exception:
            continue
    return mats

def max_lib_corr(panel, grid, lib_mats, min_valid=8):
    arr = panel.reindex(grid)[WATCHLIST].values.astype(float)
    best, best_id = 0.0, None
    for fid, lm in lib_mats.items():
        corrs = []
        for i in range(len(grid)):
            x = arr[i]; y = lm[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= min_valid:
                xs = pd.Series(x[m]).rank(); ys = pd.Series(y[m]).rank()
                c = xs.corr(ys)
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

# ---------------- candidate factor definitions ----------------
def f_autocorr_20(df, s):
    r = df['close'].pct_change()
    ac = r.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 3 and np.std(x) > 0 else np.nan, raw=False)
    return ac

def f_kurtosis_20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).kurt()

def f_downside_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    def f(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 10:
            return np.nan
        sd = np.std(x[x < 0]) if (x < 0).any() else 0.0
        tot = np.std(x)
        return sd / tot if tot > 0 else np.nan
    return r.rolling(20).apply(f, raw=True)

def f_updown_vol_asym_20(df, s):
    r = df['close'].pct_change()
    def f(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        up = x[x > 0]; dn = x[x < 0]
        if len(up) < 3 or len(dn) < 3:
            return np.nan
        su = np.std(up); sd = np.std(dn)
        return su / sd if sd > 0 else np.nan
    return r.rolling(20).apply(f, raw=True)

def f_btc_beta_60(df, s, btc_ret):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), btc_ret.rename('b')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['b']) / z['b'].rolling(60).var()
    return b.reindex(z.index)

def f_usdjpy_beta_cond_60x20(df, s, jpy):
    r = df['close'].pct_change(); jr = jpy['close'].pct_change()
    z = pd.concat([r.rename('r'), jr.rename('j')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['j']) / z['j'].rolling(60).var()
    cond = (jpy['close'] / jpy['close'].shift(20) - 1.0)
    return (-b * cond).reindex(z.index)

def f_park_vol_ratio_20_60(df, s):
    hl = np.log(df['high'] / df['low'])
    park = np.sqrt(hl.pow(2).rolling(20).mean() / (4 * np.log(2)))
    park60 = np.sqrt(hl.pow(2).rolling(60).mean() / (4 * np.log(2)))
    return (park / park60).reindex(df.index)

def f_gk_vol_ratio_20_60(df, s):
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    logc = np.log(c / o); loghl = np.log(h / l); loghc = np.log(h / c); loglo = np.log(l / o)
    gk = np.sqrt(0.5 * loghl.pow(2) - (2 * np.log(2) - 1) * logc.pow(2))
    gk20 = gk.rolling(20).mean(); gk60 = gk.rolling(60).mean()
    return (gk20 / gk60).reindex(df.index)

def f_vol_beta_spx_60(df, s, spx_ret):
    r = df['close'].pct_change()
    v = r.rolling(20).std(); vs = spx_ret.rolling(20).std()
    z = pd.concat([v.rename('v'), vs.rename('vs')], axis=1).dropna()
    dv = z['v'].diff(); dvs = z['vs'].diff()
    z2 = pd.concat([dv.rename('dv'), dvs.rename('dvs')], axis=1).dropna()
    b = z2['dv'].rolling(60).cov(z2['dvs']) / z2['dvs'].rolling(60).var()
    return b.reindex(z2.index)

def f_mom_252_pos(df, s):
    hi = df['close'].rolling(252).max(); lo = df['close'].rolling(252).min()
    rng = (hi - lo)
    return ((df['close'] - lo) / rng).reindex(df.index)

def f_maxdd_depth_60(df, s):
    c = df['close']
    roll_max = c.rolling(60, min_periods=30).max()
    dd = c / roll_max - 1.0
    return dd

def f_intraday_noise_20(df, s):
    """mean (high-low)/close vs close-to-close vol: noise-to-trend ratio"""
    hl = (df['high'] - df['low']) / df['close']
    r = df['close'].pct_change()
    noise = hl.rolling(20).mean()
    vol = r.rolling(20).std()
    return (noise / vol).reindex(df.index)

prices = load_prices(6000)
grid = canonical_grid(prices)
print(f"prices loaded: {len(prices)} assets, grid {len(grid)} dates "
      f"({grid.min().date()}..{grid.max().date()})")
lib_mats = load_library_matrices(grid)
print(f"library matrices loaded: {len(lib_mats)} factors")

btc = prices['BTC']['close'].pct_change()
spx = prices['SPX']['close'].pct_change()
jpy = load_index('USDJPY', prices)

candidates = {
    'autocorr_ret_20': f_autocorr_20,
    'kurtosis_20': f_kurtosis_20,
    'downside_vol_ratio_20': f_downside_vol_ratio_20,
    'updown_vol_asym_20': f_updown_vol_asym_20,
    'btc_beta_60': lambda df, s: f_btc_beta_60(df, s, btc),
    'usdjpy_beta_cond_60x20': lambda df, s: f_usdjpy_beta_cond_60x20(df, s, jpy),
    'park_vol_ratio_20_60': f_park_vol_ratio_20_60,
    'gk_vol_ratio_20_60': f_gk_vol_ratio_20_60,
    'vol_beta_spx_60': lambda df, s: f_vol_beta_spx_60(df, s, spx),
    'mom_252_pos': f_mom_252_pos,
    'maxdd_depth_60': f_maxdd_depth_60,
    'intraday_noise_20': f_intraday_noise_20,
}

results = {}
for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    m = validate(panel, prices)
    if m is None:
        print(f"{fid}: INSUFFICIENT DATA (panel {panel.shape})")
        continue
    rho, rho_id = max_lib_corr(panel, grid, lib_mats)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    results[fid] = m
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"\n=== {fid} === panel {panel.shape} coverage={m['coverage_asset_days']:.2f} "
          f"dates_ge8={m['coverage_dates_ge8']:.2f}")
    print(f"IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} turn={m['turnover_10d_rank']:.2f} maxrho={rho:.3f}({rho_id})")
    print("decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()})
    print(f"ADMISSION: {'PASS' if ok else 'FAIL'}")

Path('scripts/miner_1_20351011_screen_results.json').write_text(
    json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'decay_ic_by_horizon'}
                for k, v in results.items()}, indent=2, default=str))
print("\nDONE")
