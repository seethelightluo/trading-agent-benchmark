"""Verify recomputed library factor panels are faithful to persisted definitions.

For each EFFECTIVE library factor with a JSON definition, recompute the signal
panel on the current canonical grid and compare:
  1. coverage_asset_days / coverage_dates_ge8 vs persisted validation.metrics
  2. artifact shape metadata (n_dates) vs current canonical grid
  3. If the stored artifact grid were date-aligned we'd compare directly; the
     stored artifacts are on OLD grids (more rows) so here we verify that our
     recomputation reproduces the persisted coverage statistics, which is the
     key faithfulness check for the rho gate proxy.
(2027-01-28)
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
T, N = len(grid), len(WATCHLIST)
print(f"canonical grid: {T} dates ({grid.min().date()}..{grid.max().date()}) | assets {len(prices)}", flush=True)

dxy = load_index('DXY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
vix = load_index('VIX', prices=prices)

def beta_to(x_ret, win, cond=None):
    def f(df, s):
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), x_ret.rename('x')], axis=1).dropna()
        if cond is not None:
            z = z[cond.reindex(z.index).astype(bool)]
        if len(z) < win + 5:
            return None
        b = z['r'].rolling(win).cov(z['x']) / z['x'].rolling(win).var().replace(0, np.nan)
        return b
    return f

def f_cn10y_beta(df, s):
    return beta_to(prices['CN10Y']['close'].diff(), 60)(df, s)

def f_comm_basket_beta(df, s):
    x = pd.concat([prices[a]['close'].pct_change() for a in ('XAU', 'COPPER', 'WTI')], axis=1).mean(axis=1)
    return beta_to(x, 60)(df, s)

def f_copper_gold_beta(df, s):
    x = prices['COPPER']['close'].pct_change() - prices['XAU']['close'].pct_change()
    return beta_to(x, 20)(df, s)

def f_hilo_pos_60(df, s):
    hi = df['high'].rolling(60).max(); lo = df['low'].rolling(60).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)

def f_hilo_vol_ratio_20(df, s):
    rng = df['close'].rolling(20).max() - df['close'].rolling(20).min()
    return rng / df['close'] / df['close'].pct_change().rolling(20).std().replace(0, np.nan)

def f_hs300_beta(df, s):
    return beta_to(prices['000300.SH']['close'].pct_change(), 60)(df, s)

def f_intraday_ret_skew_20(df, s):
    return (df['close'] / df['open'] - 1.0).rolling(20).skew()

def f_mom_accel_60_120(df, s):
    c = df['close']
    return (c.shift(5) / c.shift(65) - 1.0) - (c.shift(5) / c.shift(125) - 1.0)

def f_range_skew_20(df, s):
    return ((df['high'] - df['low']) / df['close']).rolling(20).skew()

def f_sign_persist_20(df, s):
    r = df['close'].pct_change()
    same = (np.sign(r) == np.sign(r.shift(1))).astype(float)
    return same.rolling(20).mean()

def f_spx_beta(df, s):
    return beta_to(prices['SPX']['close'].pct_change(), 60)(df, s)

def f_streak_60(df, s):
    r = df['close'].pct_change()
    us = pd.Series(0.0, index=r.index); ds = pd.Series(0.0, index=r.index)
    for i in range(1, len(r)):
        us.iloc[i] = (us.iloc[i-1] + 1) if r.iloc[i] > 0 else 0.0
        ds.iloc[i] = (ds.iloc[i-1] + 1) if r.iloc[i] < 0 else 0.0
    return (us - ds).rolling(60).max() / 60.0

def f_vix_beta_cond(df, s):
    if vix is None:
        return None
    r = df['close'].pct_change(); x = vix['close'].pct_change()
    cond = vix['close'] / vix['close'].shift(20) - 1.0
    z = pd.concat([r.rename('r'), x.rename('x'), cond.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['x']) / z['x'].rolling(60).var().replace(0, np.nan)
    return (-b * z['c']).reindex(z.index)

def f_vol_adj_mom_20_60(df, s):
    c = df['close']
    mom20 = c.shift(5) / c.shift(25) - 1.0
    return mom20 / c.pct_change().rolling(60).std().replace(0, np.nan)

def f_vol_of_vol(df, s):
    return df['close'].pct_change().rolling(20).std().rolling(60).std()

def f_vol_regime_switch(df, s):
    rvol = df['close'].pct_change().rolling(20).std()
    state = (rvol > rvol.rolling(60).median()).astype(float)
    return state.diff().abs().rolling(60).mean()

def f_dxy_beta_cond(df, s):
    if dxy is None:
        return None
    r = df['close'].pct_change(); x = dxy['close'].pct_change()
    cond = dxy['close'] / dxy['close'].shift(20) - 1.0
    z = pd.concat([r.rename('r'), x.rename('x'), cond.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['x']) / z['x'].rolling(60).var().replace(0, np.nan)
    return (b * z['c']).reindex(z.index)

def f_eurusd_beta_cond(df, s):
    if eurusd is None:
        return None
    r = df['close'].pct_change(); x = eurusd['close'].pct_change()
    cond = eurusd['close'] / eurusd['close'].shift(20) - 1.0
    z = pd.concat([r.rename('r'), x.rename('x'), cond.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['x']) / z['x'].rolling(60).var().replace(0, np.nan)
    return (b * z['c']).reindex(z.index)

def dd_panel():
    cols = {}
    for s, df in prices.items():
        c = df['close']
        hh = c.rolling(120).max()
        is_high = (c >= hh)
        idx = np.arange(len(c))
        last_high = pd.Series(np.where(is_high.values, idx, np.nan), index=c.index).ffill()
        cols[s] = pd.Series(np.log1p(idx - last_high.values), index=c.index)
    return pd.DataFrame(cols).sort_index()

def mom120_panel():
    cols = {}
    for s, df in prices.items():
        c = df['close']
        cols[s] = c.shift(5) / c.shift(125) - 1.0
    return pd.DataFrame(cols).sort_index()

LIB_FNS = {
    'cn10y_beta_60': f_cn10y_beta,
    'comm_basket_beta_60': f_comm_basket_beta,
    'copper_gold_beta_20': f_copper_gold_beta,
    'down_beta_60': lambda df, s: beta_to(prices['SPX']['close'].pct_change(), 60,
                                          cond=prices['SPX']['close'].pct_change() < 0)(df, s),
    'dxy_beta_cond_60x20': f_dxy_beta_cond,
    'eurusd_beta_cond_60x20': f_eurusd_beta_cond,
    'hilo_pos_60': f_hilo_pos_60,
    'hilo_vol_ratio_20': f_hilo_vol_ratio_20,
    'hs300_beta_60': f_hs300_beta,
    'intraday_ret_skew_20': f_intraday_ret_skew_20,
    'mom_accel_60_120': f_mom_accel_60_120,
    'range_skew_20': f_range_skew_20,
    'sign_persist_20': f_sign_persist_20,
    'spx_beta_60': f_spx_beta,
    'streak_60': f_streak_60,
    'vix_beta_cond_60x20': f_vix_beta_cond,
    'vol_adj_mom_20_60': f_vol_adj_mom_20_60,
    'vol_of_vol20x60': f_vol_of_vol,
    'vol_regime_switch_20x60': f_vol_regime_switch,
}

def factor_to_panel(fn, prices):
    cols = {}
    for s, df in prices.items():
        try:
            ser = fn(df, s)
            if ser is not None and len(ser) > 0:
                cols[s] = ser.astype(float)
        except Exception:
            pass
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols)
    return panel[~panel.index.duplicated(keep='last')].sort_index()

lib_panels = {}
for fid, fn in LIB_FNS.items():
    p = factor_to_panel(fn, prices)
    if p is not None and len(p):
        lib_panels[fid] = p

# dd_duration_120_resid (cross-sectional ortho)
ddp, m120 = dd_panel(), mom120_panel()
common = ddp.index.intersection(m120.index)
ddp, m120 = ddp.loc[common], m120.loc[common]
z = m120.sub(m120.mean(axis=1), axis=0).div(m120.std(axis=1).replace(0, np.nan), axis=0)
b = (ddp * z).sum(axis=1) / (z * z).sum(axis=1).replace(0, np.nan)
dd_resid = ddp.sub(z.mul(b, axis=0))
dd_resid = dd_resid[~dd_resid.index.duplicated(keep='last')].sort_index()
lib_panels['dd_duration_120_resid'] = dd_resid
print(f"recomputed {len(lib_panels)} library panels in {time.time()-t0:.1f}s", flush=True)

# load persisted metadata
persisted = {}
import glob, os
for f in sorted(glob.glob('factors/*.json')):
    if f.endswith('.bak') or 'ensemble' in f:
        continue
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') == 'EFFECTIVE':
            persisted[d['factor_id']] = d
    except Exception:
        pass

print(f"\n{'factor':26s} {'rec_cov':>7s} {'per_cov':>7s} {'rec_ge8':>7s} {'per_ge8':>7s} {'n_dates':>7s} {'art_n':>6s}  verdict", flush=True)
ok_all = True
for fid in sorted(lib_panels):
    p = lib_panels[fid]
    fac = p[(p.index >= VAL_START) & (p.index <= VAL_END)]
    rec_cov = float(fac.notna().sum().sum()) / (fac.shape[0] * fac.shape[1]) if fac.shape[0] else float('nan')
    rec_ge8 = float((fac.notna().sum(axis=1) >= 8).mean()) if len(fac) else float('nan')
    pd_ = persisted.get(fid, {})
    pm = pd_.get('validation', {}).get('metrics', {})
    per_cov = pm.get('coverage_asset_days')
    per_ge8 = pm.get('coverage_dates_ge8')
    art_n = (pd_.get('signal_artifact_shape') or [None])[0]
    verdict = 'OK'
    if per_cov is not None and abs(rec_cov - per_cov) > 0.15:
        verdict = 'COV-MISMATCH'; ok_all = False
    print(f"{fid:26s} {rec_cov:7.3f} {str(per_cov):>7s} {rec_ge8:7.3f} {str(per_ge8):>7s} {fac.shape[0]:7d} {str(art_n):>6s}  {verdict}", flush=True)

# save recomputed library rank matrices for the batch-B screen
out = {}
for fid, p in lib_panels.items():
    out[fid] = signal_matrix(p, grid)
np.savez_compressed('scripts/miner_1_20270128_lib_signals.npz', **{k: v for k, v in out.items()})
print(f"\nsaved recomputed library signals -> scripts/miner_1_20270128_lib_signals.npz | ALL_OK={ok_all} | {time.time()-t0:.1f}s", flush=True)
