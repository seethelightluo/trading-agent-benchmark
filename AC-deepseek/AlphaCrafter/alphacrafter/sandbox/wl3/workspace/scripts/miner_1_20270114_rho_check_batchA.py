"""Recompute effective library factor signals on current data and compute
exact date-aligned max_abs_library_correlation for new batch-A candidates.
(2027-01-14)"""
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
print(f"grid {T} dates | assets {len(prices)}", flush=True)

dxy = load_index('DXY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
vix = load_index('VIX', prices=prices)

def beta_to(x_ret, win, cond=None):
    """Return fn(df,s) -> rolling beta of asset ret to x_ret (optionally on
    filtered rows via cond mask applied before rolling)."""
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
    cn = prices['CN10Y']['close'].diff()
    return beta_to(cn, 60)(df, s)

def f_comm_basket_beta(df, s):
    x = pd.concat([prices[a]['close'].pct_change() for a in ('XAU', 'COPPER', 'WTI')], axis=1).mean(axis=1)
    return beta_to(x, 60)(df, s)

def f_copper_gold_beta(df, s):
    x = prices['COPPER']['close'].pct_change() - prices['XAU']['close'].pct_change()
    return beta_to(x, 20)(df, s)

def f_dd_duration_120_resid(df, s):
    c = df['close']
    hh = c.rolling(120).max()
    is_high = (c >= hh)
    idx = np.arange(len(c))
    last_high = pd.Series(np.where(is_high.values, idx, np.nan), index=c.index).ffill()
    dd = np.log1p(idx - last_high.values)
    mom120 = c.shift(5) / c.shift(125) - 1.0
    z = mom120
    # per-date cross-sectional orthogonalization
    panel_dd = pd.Series(dd, index=c.index)
    return panel_dd  # ortho done cross-sectionally below via helper

def f_hilo_pos_60(df, s):
    hi = df['high'].rolling(60).max(); lo = df['low'].rolling(60).min()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)

def f_hilo_vol_ratio_20(df, s):
    rng = df['close'].rolling(20).max() - df['close'].rolling(20).min()
    return rng / df['close'] / df['close'].pct_change().rolling(20).std().replace(0, np.nan)

def f_hs300_beta(df, s):
    x = prices['000300.SH']['close'].pct_change()
    return beta_to(x, 60)(df, s)

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
    x = prices['SPX']['close'].pct_change()
    return beta_to(x, 60)(df, s)

def f_streak_60(df, s):
    r = df['close'].pct_change()
    pos = (r > 0).astype(float); neg = (r < 0).astype(float)
    up = pos * (pos.shift(1).fillna(0) * 0 + 1)  # placeholder
    us = pd.Series(0.0, index=r.index); ds = pd.Series(0.0, index=r.index)
    for i in range(1, len(r)):
        us.iloc[i] = (us.iloc[i-1] + 1) if r.iloc[i] > 0 else 0.0
        ds.iloc[i] = (ds.iloc[i-1] + 1) if r.iloc[i] < 0 else 0.0
    net = (us - ds).rolling(60).max() / 60.0
    return net

def f_vix_beta_cond(df, s):
    if vix is None:
        return None
    r = df['close'].pct_change()
    x = vix['close'].pct_change()
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
    flips = state.diff().abs()
    return flips.rolling(60).mean()

def f_dxy_beta_cond(df, s):
    if dxy is None:
        return None
    r = df['close'].pct_change()
    x = dxy['close'].pct_change()
    cond = dxy['close'] / dxy['close'].shift(20) - 1.0
    z = pd.concat([r.rename('r'), x.rename('x'), cond.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['x']) / z['x'].rolling(60).var().replace(0, np.nan)
    return (b * z['c']).reindex(z.index)

def f_eurusd_beta_cond(df, s):
    if eurusd is None:
        return None
    r = df['close'].pct_change()
    x = eurusd['close'].pct_change()
    cond = eurusd['close'] / eurusd['close'].shift(20) - 1.0
    z = pd.concat([r.rename('r'), x.rename('x'), cond.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['x']) / z['x'].rolling(60).var().replace(0, np.nan)
    return (b * z['c']).reindex(z.index)

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
    # dd_duration_120_resid handled separately (cross-sectional ortho)
}

def factor_to_panel(fn, prices):
    cols = {}
    for s, df in prices.items():
        try:
            ser = fn(df, s)
            if ser is not None and len(ser) > 0:
                cols[s] = ser.astype(float)
        except Exception as e:
            pass
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols)
    return panel[~panel.index.duplicated(keep='last')].sort_index()

def rank_matrix(df):
    return df.rank(axis=1).values.astype(float)

def spearman_from_ranks(xr, yr):
    valid = np.isfinite(xr) & np.isfinite(yr)
    nv = valid.sum(axis=1)
    ok = nv >= 8
    out = np.full(len(nv), np.nan)
    xc = np.where(valid, xr, np.nan); yc = np.where(valid, yr, np.nan)
    mx = np.nanmean(xc, axis=1, keepdims=True); my = np.nanmean(yc, axis=1, keepdims=True)
    xc = np.where(valid, xr - mx, 0.0); yc = np.where(valid, yr - my, 0.0)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    out[ok] = num[ok] / den[ok]
    return out

# ---- dd_duration_120_resid with cross-sectional ortho ----
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

ddp, m120 = dd_panel(), mom120_panel()
common = ddp.index.intersection(m120.index)
ddp, m120 = ddp.loc[common], m120.loc[common]
z = m120.sub(m120.mean(axis=1), axis=0).div(m120.std(axis=1).replace(0, np.nan), axis=0)
b = (ddp * z).sum(axis=1) / (z * z).sum(axis=1).replace(0, np.nan)
dd_resid = ddp.sub(z.mul(b, axis=0))
dd_resid = dd_resid[~dd_resid.index.duplicated(keep='last')].sort_index()

print(f"library panels built in {time.time()-t0:.1f}s", flush=True)

lib_panels = {}
for fid, fn in LIB_FNS.items():
    p = factor_to_panel(fn, prices)
    if p is not None and len(p):
        lib_panels[fid] = p
lib_panels['dd_duration_120_resid'] = dd_resid
print(f"library panels ready: {len(lib_panels)} -> {sorted(lib_panels.keys())}", flush=True)

# ---- candidate panels (from batch A) ----
usdjpy = load_index('USDJPY', prices=prices)

def f_usdjpy_beta_cond(df, s):
    if usdjpy is None:
        return None
    r = df['close'].pct_change()
    x = usdjpy['close'].pct_change()
    cond = usdjpy['close'] / usdjpy['close'].shift(20) - 1.0
    zz = pd.concat([r.rename('r'), x.rename('x'), cond.rename('c')], axis=1).dropna()
    b = zz['r'].rolling(60).cov(zz['x']) / zz['x'].rolling(60).var().replace(0, np.nan)
    return (b * zz['c']).reindex(zz.index)

def f_boll_bandwidth_20(df, s):
    c = df['close']
    return 2.0 * c.rolling(20).std() / c.rolling(20).mean().replace(0, np.nan)

def f_downside_beta_20(df, s):
    spx = prices['SPX']['close'].pct_change()
    r = df['close'].pct_change()
    zz = pd.concat([r.rename('r'), spx.rename('m')], axis=1).dropna()
    dn = zz[zz['m'] < 0]
    if len(dn) < 30:
        return None
    return dn['r'].rolling(20).cov(dn['m']) / dn['m'].rolling(20).var().replace(0, np.nan)

def f_spx_hsi_ratio_beta(df, s):
    xr = (prices['SPX']['close'] / prices['HSI']['close']).pct_change()
    r = df['close'].pct_change()
    zz = pd.concat([r.rename('r'), xr.rename('x')], axis=1).dropna()
    return zz['r'].rolling(60).cov(zz['x']) / zz['x'].rolling(60).var().replace(0, np.nan)

def f_updown_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    us = (r.clip(lower=0) ** 2).rolling(20).mean().apply(np.sqrt)
    ds = (r.clip(upper=0) ** 2).rolling(20).mean().apply(np.sqrt)
    return ds / us.replace(0, np.nan)

def f_zscore_20(df, s):
    c = df['close']
    return (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)

def f_btc_eth_ratio_beta(df, s):
    xr = (prices['BTC']['close'] / prices['ETH']['close']).pct_change()
    r = df['close'].pct_change()
    zz = pd.concat([r.rename('r'), xr.rename('x')], axis=1).dropna()
    return zz['r'].rolling(60).cov(zz['x']) / zz['x'].rolling(60).var().replace(0, np.nan)

CAND_FNS = {
    'boll_bandwidth_20': f_boll_bandwidth_20,
    'downside_beta_20': f_downside_beta_20,
    'spx_hsi_ratio_beta_60': f_spx_hsi_ratio_beta,
    'updown_vol_ratio_20': f_updown_vol_ratio_20,
    'zscore_20': f_zscore_20,
    'usdjpy_beta_cond_60x20': f_usdjpy_beta_cond,
    'btc_eth_ratio_beta_60': f_btc_eth_ratio_beta,
}

lib_rank = {fid: rank_matrix(pd.DataFrame(signal_matrix(p, grid), index=grid, columns=WATCHLIST))
            for fid, p in lib_panels.items()}

results = {}
for cid, fn in CAND_FNS.items():
    panel = factor_to_panel(fn, prices)
    pm = pd.DataFrame(signal_matrix(panel, grid), index=grid, columns=WATCHLIST)
    pr = rank_matrix(pm)
    best, best_id, per = 0.0, None, {}
    for fid, lr in lib_rank.items():
        r = spearman_from_ranks(pr, lr)
        rr = r[np.isfinite(r)]
        if len(rr) > 0:
            per[fid] = float(np.nanmean(rr))
            if abs(per[fid]) > best:
                best, best_id = abs(per[fid]), fid
    results[cid] = {'max_abs_library_correlation': best, 'max_corr_library_id': best_id,
                    'top': {k: round(v, 3) for k, v in sorted(per.items(), key=lambda kv: -abs(kv[1]))[:5]}}
    print(f"\n{cid}: rho={best:.3f} ({best_id})", flush=True)
    for k, v in results[cid]['top'].items():
        print(f"    {k}: {v}", flush=True)

with open('scripts/miner_1_20270114_rho_check_batchA.json', 'w') as fh:
    json.dump(results, fh, indent=1)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
