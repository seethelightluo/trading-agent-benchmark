"""miner_1 2035-10-11: screen novel factor candidates (vectorized, fast).
Validation window: 2020-01-01..2026-07-15. Gates |IC|>=0.007, |ICIR|>=0.084 @h10.
Phase 1: compute panels + IC/ICIR only. Library correlation done for PASS only.
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
        except Exception:
            pass
    return out

def load_index(symbol, prices, days=6000):
    try:
        df = get_index_daily_data(symbol=symbol, days=days)
        if df is not None and len(df) >= 30:
            df = df.copy(); df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date'); df['close'] = pd.to_numeric(df['close'], errors='coerce')
            return df
    except Exception:
        pass
    path = Path('../persistent/index_data') / f'{symbol}.csv'
    df = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    visible = max(dd.index.max() for dd in prices.values())
    return df[df.index <= visible]

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
    ic_mean = float(ic10.mean()); ic_std = float(ic10.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())
    fac = factor_panel[(factor_panel.index >= VAL_START) & (factor_panel.index <= VAL_END)]
    total_cells = fac.shape[0] * fac.shape[1]
    coverage = float(fac.notna().sum().sum()) / total_cells if total_cells else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= min_valid).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    decay = {}
    for h in (1, 2, 3, 5, 10, 20):
        fh = forward_returns(prices, h)
        ics = rank_ic_series(factor_panel, fh, min_valid)
        ics = ics[(ics.index >= VAL_START) & (ics.index <= VAL_END)]
        decay[str(h)] = float(ics.mean()) if len(ics) else float('nan')
    return {'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit, 'n_ic_dates': int(len(ic10)),
            'coverage_asset_days': coverage, 'coverage_dates_ge8': ge8,
            'turnover_10d_rank': turn, 'decay_ic_by_horizon': decay}

# ---------------- vectorized candidate factors ----------------
def f_autocorr_20(df, s):
    r = df['close'].pct_change()
    return (r.rolling(20).cov(r.shift(1)) / r.rolling(20).var()).reindex(df.index)

def f_kurtosis_20(df, s):
    return df['close'].pct_change().rolling(20).kurt()

def _downside_ratio(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    neg = x[x < 0]
    dd = np.sqrt(np.mean(neg ** 2)) if len(neg) else 0.0
    tot = np.std(x)
    return dd / tot if tot > 0 else np.nan

def f_downside_vol_ratio_20(df, s):
    r = df['close'].pct_change().values
    w = 20
    if len(r) < w + 1:
        return pd.Series(index=df.index, dtype=float)
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(r, w)
    vals = np.apply_along_axis(_downside_ratio, 1, sw)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.iloc[w - 1:] = vals
    return out

def _updown_asym(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    up = x[x > 0]; dn = x[x < 0]
    if len(up) < 3 or len(dn) < 3:
        return np.nan
    su = np.std(up); sd = np.std(dn)
    return su / sd if sd > 0 else np.nan

def f_updown_vol_asym_20(df, s):
    r = df['close'].pct_change().values
    w = 20
    if len(r) < w + 1:
        return pd.Series(index=df.index, dtype=float)
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(r, w)
    vals = np.apply_along_axis(_updown_asym, 1, sw)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.iloc[w - 1:] = vals
    return out

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
    return (c / roll_max - 1.0).reindex(df.index)

def f_intraday_noise_20(df, s):
    hl = (df['high'] - df['low']) / df['close']
    r = df['close'].pct_change()
    noise = hl.rolling(20).mean(); vol = r.rolling(20).std()
    return (noise / vol).reindex(df.index)

def f_ret_skew_60(df, s):
    """60d return skewness (longer horizon than existing 20d skew)"""
    return df['close'].pct_change().rolling(60).skew()

def f_winrate_ratio_20(df, s):
    """win rate over 20d scaled by avg win/loss magnitude (profit factor proxy)"""
    r = df['close'].pct_change()
    def pf(x):
        x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
        up = x[x > 0].sum(); dn = -x[x < 0].sum()
        return up / dn if dn > 0 else np.nan
    return r.rolling(20).apply(pf, raw=True)

prices = load_prices(6000)
print(f"prices: {len(prices)} assets")
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
    'ret_skew_60': f_ret_skew_60,
    'winrate_ratio_20': f_winrate_ratio_20,
}

results = {}
for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    m = validate(panel, prices)
    if m is None:
        print(f"{fid}: INSUFFICIENT (panel {panel.shape})")
        continue
    results[fid] = m
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"{fid}: IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.2f} ge8={m['coverage_dates_ge8']:.2f} "
          f"turn={m['turnover_10d_rank']:.2f} -> {'PASS' if ok else 'FAIL'}")
    print("   decay:", {k: round(v, 4) for k, v in m['decay_ic_by_horizon'].items()})

Path('scripts/miner_1_20351011_screen_phase1.json').write_text(json.dumps(results, indent=2, default=str))
print("DONE")
