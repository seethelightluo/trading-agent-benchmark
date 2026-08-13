"""miner_1 2031-12-11: full-library drift re-validation (all 23 effective factors).

Recomputes every factor signal on full available history and evaluates:
  WARM  : 2020-01-01..2026-07-15 (admission reference)
  OOS   : 2026-07-16..<latest visible date> (online period)
  RECENT: last ~365 days (decision-relevant regime)

Admission gate (shared): |IC|>=0.007 and |ICIR|>=0.084 (h=10).
Frozen assets (HSI,SX5E,BTC,US10Y,CN10Y) excluded from OOS cross-sections
to match online trading universe composition.
"""
import sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           forward_returns, rank_ic_series, VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=3200)
max_date = max(dd.index.max() for dd in prices.values())
print(f"prices: {len(prices)} assets, last date {max_date.date()} ({time.time()-t0:.1f}s)", flush=True)

FROZEN = {'HSI', 'SX5E', 'BTC', 'US10Y', 'CN10Y'}
LIVE = [s for s in WATCHLIST if s not in FROZEN]
print(f"live assets ({len(LIVE)}): {LIVE}", flush=True)

# ---------- market references ----------
spx_r = prices['SPX']['close'].pct_change()
hs300_r = prices['000300.SH']['close'].pct_change()
cn10y_d = prices['CN10Y']['close'].diff()
dxy = load_index('DXY', prices=prices); dxy_r = dxy['close'].pct_change() if dxy is not None else None
eur = load_index('EURUSD', prices=prices); eur_r = eur['close'].pct_change() if eur is not None else None
vix = load_index('VIX', prices=prices); vix_r = vix['close'].pct_change() if vix is not None else None
usd_jpy = load_index('USDJPY', prices=prices); jpy_r = usd_jpy['close'].pct_change() if usd_jpy is not None else None
usd_cny = load_index('USDCNY', prices=prices); cny_r = usd_cny['close'].pct_change() if usd_cny is not None else None
comm_r = pd.concat([prices[s]['close'].pct_change().rename(s) for s in ['XAU', 'COPPER', 'WTI']], axis=1).mean(axis=1)
cg_spread = prices['COPPER']['close'].pct_change() - prices['XAU']['close'].pct_change()


def rb(r, m, w, cond=None, min_obs=0.5):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    if cond is not None:
        z = z[cond.reindex(z.index).astype(bool)]
    if len(z) < 30:
        return pd.Series(np.nan, index=r.index)
    b = z['r'].rolling(w, min_periods=int(w * min_obs)).cov(z['m']) / \
        z['m'].rolling(w, min_periods=int(w * min_obs)).var().replace(0, np.nan)
    return b.reindex(r.index)


def f_cn10y_beta_60(df, s):
    return rb(df['close'].pct_change(), cn10y_d, 60)

def f_comm_basket_beta_60(df, s):
    return rb(df['close'].pct_change(), comm_r, 60)

def f_copper_gold_beta_20(df, s):
    return rb(df['close'].pct_change(), cg_spread, 20)

def f_down_beta_60(df, s):
    return rb(df['close'].pct_change(), spx_r, 60, cond=spx_r < 0)

def f_dxy_beta_cond_60x20(df, s):
    if dxy_r is None:
        return None
    r = df['close'].pct_change()
    return rb(r, dxy_r, 60) * (dxy['close'] / dxy['close'].shift(20) - 1.0)

def f_eurusd_beta_cond_60x20(df, s):
    if eur_r is None:
        return None
    r = df['close'].pct_change()
    return rb(r, eur_r, 60) * (eur['close'] / eur['close'].shift(20) - 1.0)

def f_gap_freq_60(df, s):
    gap = (df['open'] / df['close'].shift(1) - 1.0).abs()
    return (gap > 0.01).astype(float).rolling(60, min_periods=30).mean()

def f_hilo_pos_60(df, s):
    hi = df['high'].rolling(60).max(); lo = df['low'].rolling(60).min()
    return ((df['close'] - lo) / (hi - lo).replace(0, np.nan))

def f_hilo_vol_ratio_20(df, s):
    c = df['close']
    rng = (c.rolling(20).max() - c.rolling(20).min()) / c
    v = c.pct_change().rolling(20).std()
    return (rng / v).replace([np.inf, -np.inf], np.nan)

def f_hs300_beta_60(df, s):
    return rb(df['close'].pct_change(), hs300_r, 60)

def f_intraday_ret_skew_20(df, s):
    return (df['close'] / df['open'] - 1.0).rolling(20, min_periods=12).skew()

def f_mom_accel_60_120(df, s):
    c = df['close']
    return c.shift(5) / c.shift(65) - c.shift(5) / c.shift(125)

def f_range_amplitude_60(df, s):
    c = df['close']
    return ((c.rolling(60).max() - c.rolling(60).min()) / c.rolling(60).mean())

def f_range_skew_20(df, s):
    return ((df['high'] - df['low']) / df['close']).rolling(20, min_periods=12).skew()

def f_sign_persist_20(df, s):
    r = df['close'].pct_change()
    same = (np.sign(r) == np.sign(r.shift(1))).astype(float)
    same[r == 0] = np.nan
    return same.rolling(20, min_periods=8).mean()

def f_spx_beta_60(df, s):
    return rb(df['close'].pct_change(), spx_r, 60)

def f_streak_60(df, s):
    r = df['close'].pct_change()
    up = (r > 0).astype(int); dn = (r < 0).astype(int)
    up_s = up.groupby((up != up.shift()).cumsum()).cumsum()
    dn_s = dn.groupby((dn != dn.shift()).cumsum()).cumsum()
    net = up_s - dn_s
    return net.rolling(60).max() / 60.0

def f_vix_beta_cond_60x20(df, s):
    if vix_r is None:
        return None
    r = df['close'].pct_change()
    return -rb(r, vix_r, 60) * (vix['close'] / vix['close'].shift(20) - 1.0)

def f_vol_adj_mom_20_60(df, s):
    r = df['close'].pct_change()
    mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
    return (mom / r.rolling(60).std()).replace([np.inf, -np.inf], np.nan)

def f_vol_of_vol20x60(df, s):
    return df['close'].pct_change().rolling(20).std().rolling(60).std()

def f_vol_regime_switch_20x60(df, s):
    rvol = df['close'].pct_change().rolling(20).std()
    med = rvol.rolling(60).median()
    state = (rvol > med).astype(float)
    switch = (state != state.shift()).astype(float)
    return switch.rolling(60).mean()

def f_dd_raw(df, s):
    c = df['close']
    run_max = c.rolling(120, min_periods=60).max()
    below = (c < run_max).astype(int)
    grp = (below != below.shift()).cumsum()
    days_since = below.groupby(grp).cumsum()
    return np.log1p(days_since)

def f_dd_duration_120_resid(df, s):
    dd = f_dd_raw(df, s)
    resid = dd - dd.rolling(120).mean()
    return resid


FUNCS = {
    'cn10y_beta_60': f_cn10y_beta_60,
    'comm_basket_beta_60': f_comm_basket_beta_60,
    'copper_gold_beta_20': f_copper_gold_beta_20,
    'dd_duration_120_resid': f_dd_duration_120_resid,
    'down_beta_60': f_down_beta_60,
    'dxy_beta_cond_60x20': f_dxy_beta_cond_60x20,
    'eurusd_beta_cond_60x20': f_eurusd_beta_cond_60x20,
    'gap_freq_60': f_gap_freq_60,
    'hilo_pos_60': f_hilo_pos_60,
    'hilo_vol_ratio_20': f_hilo_vol_ratio_20,
    'hs300_beta_60': f_hs300_beta_60,
    'intraday_ret_skew_20': f_intraday_ret_skew_20,
    'mom_accel_60_120': f_mom_accel_60_120,
    'range_amplitude_60': f_range_amplitude_60,
    'range_skew_20': f_range_skew_20,
    'sign_persist_20': f_sign_persist_20,
    'spx_beta_60': f_spx_beta_60,
    'streak_60': f_streak_60,
    'vix_beta_cond_60x20': f_vix_beta_cond_60x20,
    'vol_adj_mom_20_60': f_vol_adj_mom_20_60,
    'vol_of_vol20x60': f_vol_of_vol20x60,
    'vol_regime_switch_20x60': f_vol_regime_switch_20x60,
}

H = 10
fwd = forward_returns(prices, H)
oos_start = VAL_END + pd.Timedelta(days=1)
recent_start = max_date - pd.Timedelta(days=365)
print(f"OOS window: {oos_start.date()} .. {max_date.date()}", flush=True)
print(f"RECENT window: {recent_start.date()} .. {max_date.date()}", flush=True)

results = {}
for name, fn in FUNCS.items():
    t1 = time.time()
    panel = factor_to_panel(fn, prices)
    # warm IC
    warm_p = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    warm_f = fwd.reindex(warm_p.index)
    warm_ic = rank_ic_series(warm_p, warm_f, min_valid=8)
    # oos IC (live assets only)
    oos_p = panel[(panel.index >= oos_start)]
    oos_f = fwd.reindex(oos_p.index)
    oos_ic = rank_ic_series(oos_p[LIVE], oos_f[LIVE], min_valid=8)
    # recent IC (live assets only)
    rec_p = panel[(panel.index >= recent_start)]
    rec_f = fwd.reindex(rec_p.index)
    rec_ic = rank_ic_series(rec_p[LIVE], rec_f[LIVE], min_valid=8)

    def stats(ic):
        if len(ic) < 2:
            return {'ic': float('nan'), 'icir': float('nan'), 'hit': float('nan'), 'n': int(len(ic))}
        sd = ic.std(ddof=1)
        return {'ic': float(ic.mean()), 'icir': float(ic.mean() / sd) if sd > 0 else float('nan'),
                'hit': float((ic > 0).mean()), 'n': int(ic.notna().sum())}

    results[name] = {'warm': stats(warm_ic), 'oos_live': stats(oos_ic), 'recent_live': stats(rec_ic)}
    r = results[name]
    print(f"{name}: warm_ic={r['warm']['ic']:+.4f}({r['warm']['n']}) "
          f"oos_ic={r['oos_live']['ic']:+.4f}({r['oos_live']['n']}) "
          f"recent_ic={r['recent_live']['ic']:+.4f} icir={r['recent_live']['icir']:+.3f} "
          f"recent_hit={r['recent_live']['hit']:.3f} ({time.time()-t1:.1f}s)", flush=True)

with open('scripts/miner_1_20311211_revalidate_lib.json', 'w') as f:
    json.dump(results, f, indent=1)
print("saved scripts/miner_1_20311211_revalidate_lib.json")
