"""miner_2 2029-05-31: full-library drift re-validation (all 20 factors).

Purpose: factors were last validated 2026-07/11 (warm-up only). This script
recomputes every factor signal on the full available history and evaluates
predictive power on:
  WARM  : 2020-01-01..2026-07-15 (reproduction of persisted IC/ICIR)
  OOS   : 2026-07-16..<max visible date> (online period, live assets only)
  RECENT: last ~12 months (most decision-relevant regime)

Admission gate (shared): |IC|>=0.007 and |ICIR|>=0.084 (h=10).
Frozen assets (HSI,SX5E,BTC,US10Y,CN10Y) are excluded from OOS cross-sections
because their flat close produces degenerate factor/forward values; the online
portfolio keeps them only via the frozenfix floor, not via factor signal.
"""
import sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, factor_to_panel,
                           forward_returns, rank_ic_series, VAL_START, VAL_END)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=3000)
max_date = max(dd.index.max() for dd in prices.values())
min_date = min(dd.index.min() for dd in prices.values())
print(f"prices: {len(prices)} assets, {min_date.date()}..{max_date.date()} "
      f"({(time.time()-t0):.1f}s)", flush=True)

# ---------- frozen audit ----------
print("\n=== frozen/flat audit (last 60 trading days) ===")
for s in WATCHLIST:
    c = prices[s]['close']
    last60 = c.iloc[-60:]
    if last60.nunique() <= 1:
        print(f"  {s:12s} FLAT last {int(last60.nunique())} unique value since {c[c != c.iloc[-1]].index.max().date() if (c != c.iloc[-1]).any() else 'ever'}")
FROZEN = {'HSI', 'SX5E', 'BTC', 'US10Y', 'CN10Y'}
LIVE = [s for s in WATCHLIST if s not in FROZEN]
print(f"live assets ({len(LIVE)}): {LIVE}")

# ---------- market references ----------
spx_r = prices['SPX']['close'].pct_change()
hs300_r = prices['000300.SH']['close'].pct_change()
cn10y_d = prices['CN10Y']['close'].diff()
dxy = load_index('DXY', prices=prices); dxy_r = dxy['close'].pct_change() if dxy is not None else None
eur = load_index('EURUSD', prices=prices); eur_r = eur['close'].pct_change() if eur is not None else None
vix = load_index('VIX', prices=prices); vix_r = vix['close'].pct_change() if vix is not None else None
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

def f_mom120(df, s):
    c = df['close']
    return c.shift(5) / c.shift(125) - 1.0


FUNCS = {
    'cn10y_beta_60': f_cn10y_beta_60,
    'comm_basket_beta_60': f_comm_basket_beta_60,
    'copper_gold_beta_20': f_copper_gold_beta_20,
    'down_beta_60': f_down_beta_60,
    'dxy_beta_cond_60x20': f_dxy_beta_cond_60x20,
    'eurusd_beta_cond_60x20': f_eurusd_beta_cond_60x20,
    'hilo_pos_60': f_hilo_pos_60,
    'hilo_vol_ratio_20': f_hilo_vol_ratio_20,
    'hs300_beta_60': f_hs300_beta_60,
    'intraday_ret_skew_20': f_intraday_ret_skew_20,
    'mom_accel_60_120': f_mom_accel_60_120,
    'range_skew_20': f_range_skew_20,
    'sign_persist_20': f_sign_persist_20,
    'spx_beta_60': f_spx_beta_60,
    'streak_60': f_streak_60,
    'vix_beta_cond_60x20': f_vix_beta_cond_60x20,
    'vol_adj_mom_20_60': f_vol_adj_mom_20_60,
    'vol_of_vol20x60': f_vol_of_vol20x60,
    'vol_regime_switch_20x60': f_vol_regime_switch_20x60,
}

fwd10 = forward_returns(prices, 10)
OOS_START = pd.Timestamp('2026-07-16')
REC_START = pd.Timestamp('2029-05-16') - pd.Timedelta(days=365)  # ~12m window


def ic_stats(ic, lo, hi):
    s = ic[(ic.index >= lo) & (ic.index <= hi)]
    if len(s) < 30:
        return None
    m = float(s.mean()); sd = float(s.std(ddof=1))
    return {'ic': m, 'icir': m / sd if sd > 0 else 0.0,
            'hit': float((s > 0).mean()), 'n': int(len(s))}


results = {}
print("\n=== building factor panels ===")
panels = {}
for fid, fn in FUNCS.items():
    panels[fid] = factor_to_panel(fn, prices)
    print(f"  {fid:26s} {panels[fid].shape}", flush=True)

# special: dd_duration_120_resid (per-date cross-sectional orthogonalization)
dd_raw = factor_to_panel(f_dd_raw, prices)
mom120 = factor_to_panel(f_mom120, prices)
dd_resid = dd_raw.copy() * np.nan
for d in dd_raw.index:
    y = dd_raw.loc[d]; z = mom120.loc[d]
    m = y.notna() & z.notna() & np.isfinite(y) & np.isfinite(z)
    if m.sum() >= 8:
        zv = (z[m] - z[m].mean()) / (z[m].std(ddof=0) + 1e-12)
        yv = y[m]
        b = float((zv * yv).sum() / (zv * zv).sum()) if (zv * zv).sum() > 0 else 0.0
        a = float(yv.mean()) - b * float(zv.mean())
        dd_resid.loc[d, m.index[m]] = yv - (a + b * zv)
panels['dd_duration_120_resid'] = dd_resid
print(f"  dd_duration_120_resid (orthogonalized panel) {dd_resid.shape}", flush=True)

print("\n=== per-factor drift (h=10) ===")
print(f"{'factor':26s} {'WARM_IC':>8s} {'WARM_IR':>8s} | {'OOS_IC':>8s} {'OOS_IR':>8s} {'OOSn':>5s} | {'REC_IC':>8s} {'REC_IR':>8s} {'RECn':>5s} | OOScov")
for fid in list(FUNCS.keys()) + ['dd_duration_120_resid']:
    panel = panels[fid]
    ic = rank_ic_series(panel, fwd10, 8)
    w = ic_stats(ic, VAL_START, VAL_END)
    o = ic_stats(ic, OOS_START, max_date)
    r = ic_stats(ic, REC_START, max_date)
    # live-only OOS (exclude frozen)
    panel_live = panel[LIVE]
    ic_live = rank_ic_series(panel_live, fwd10[LIVE], 8)
    ol = ic_stats(ic_live, OOS_START, max_date)
    sub = panel[(panel.index >= OOS_START) & (panel.index <= max_date)]
    cov = float((sub.notna().sum(axis=1) >= 8).mean())
    results[fid] = {'warm': w, 'oos': o, 'oos_live': ol, 'recent': r, 'oos_cov': cov}
    def fmt(x, d=1):
        return f"{x['ic']:8.4f} {x['icir']:8.4f} {x['n']:5d}" if x else "     nan      nan     0"
    print(f"{fid:26s} {fmt(w,0)[:17]} | {fmt(ol)[:22]} | {fmt(r)[:22]} | {cov:.2f}", flush=True)

print("\n=== direction-consistency check (live-only OOS vs warm-up) ===")
for fid in results:
    w = results[fid]['warm']; ol = results[fid]['oos_live']
    if w and ol:
        flip = (w['ic'] > 0) != (ol['ic'] > 0)
        print(f"{fid:26s} warmIC={w['ic']:+.4f} oosLiveIC={ol['ic']:+.4f} flipped={flip}")

json.dump({k: {kk: (vv if vv is None else {**vv}) for kk, vv in v.items()}
           for k, v in results.items()},
          open('scripts/miner_2_20290531_revalidate_results.json', 'w'), indent=1, default=str)
print(f"\nsaved scripts/miner_2_20290531_revalidate_results.json; total {time.time()-t0:.1f}s")
