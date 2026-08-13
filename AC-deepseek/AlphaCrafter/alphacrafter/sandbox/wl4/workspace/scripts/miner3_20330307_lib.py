"""miner_3 shared research lib for 2033-03-07 cycle.

Loads the 15-asset tradable universe + macro observation signals through the
last completed trading day (2033-03-04). Computes daily cross-sectional rank IC
at h=10 (admission horizon), coverage, turnover, decay, yearly IC and
max-abs correlation vs the full persisted library signal set.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU',
         'COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO = ['VIX','DXY','USDCNY','USDJPY','EURUSD']
VISIBLE_THROUGH = '2033-03-04'   # last completed trading day before 2033-03-07
MIN_INSTR = 8
IC_TH, ICIR_TH = 0.007, 0.084

def load_panels(days=4500):
    panels = {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is None or len(df) < 120:
            continue
        df = df.copy(); df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df[df.index <= pd.Timestamp(VISIBLE_THROUGH)]
        panels[s] = df
    for s in MACRO:
        df = get_index_daily_data(symbol=s, days=days)
        if df is None or len(df) < 120:
            continue
        df = df.copy(); df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df[df.index <= pd.Timestamp(VISIBLE_THROUGH)]
        panels[s] = df
    return panels

def close_panel(panels):
    return pd.concat({a: panels[a]['close'].astype(float) for a in WATCH if a in panels}, axis=1).sort_index()

def fwd_returns(closes, h):
    return closes.shift(-h) / closes - 1.0

def rank_ic_series(F, R, min_valid=MIN_INSTR):
    dates, ics = [], []
    for dt in F.index:
        if dt not in R.index:
            continue
        f = F.loc[dt]; r = R.loc[dt]
        pair = pd.concat([f.rename('f'), r.rename('r')], axis=1).dropna()
        if len(pair) < min_valid or pair['r'].std() < 1e-14 or pair['f'].std() < 1e-14:
            continue
        ic = pair['f'].corr(pair['r'], method='spearman')
        if math.isfinite(ic):
            dates.append(dt); ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name='ic')

def summarize(ics, expected_sign=1):
    if len(ics) == 0:
        return None
    m = ics.mean(); s = ics.std(ddof=1)
    icir = m / s if s > 0 else 0.0
    hit = float((np.sign(ics) == np.sign(m)).mean())
    return {'ic': float(m), 'icir': float(icir), 'ic_hit': float(hit),
            'n': int(len(ics)), 'ic_std': float(s)}

def coverage_turnover(F):
    cov = F.notna().mean(axis=1)
    dates_ge8 = float((F.notna().sum(axis=1) >= MIN_INSTR).mean())
    ranks = F.rank(axis=1)
    turn = ranks.diff(10).abs().mean().mean()
    return {'cov_asset_days': float(F.notna().sum().sum() / (F.shape[0]*F.shape[1])),
            'cov_dates_ge8': round(dates_ge8, 3),
            'turnover_10d_rank': round(float(turn), 3) if math.isfinite(turn) else None}

def yearly_ic(ics):
    out = {}
    for y, g in ics.groupby(ics.index.year):
        if len(g) >= 20:
            out[str(y)] = (round(float(g.mean()), 4), round(float(g.mean()/g.std(ddof=1)), 3) if g.std(ddof=1) > 0 else 0.0, len(g))
    return out

# ---------------- library signal recomputation ----------------
def _beta(asset_ret, fac_ret):
    b = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename('a'), fac_ret.rename('f')], axis=1).dropna()
        b[a] = z['a'].rolling(60).cov(z['f']) / z['f'].rolling(60).var()
    return pd.DataFrame(b, index=asset_ret.index)

def library_signals(panels):
    c = close_panel(panels)
    r = c.pct_change()
    mkt = r.mean(axis=1)
    sig = {}
    sig['vol_adj_mom_accel_20x60'] = (c.shift(1)/c.shift(21)-1 - (c.shift(1)/c.shift(61)-1)) / r.rolling(20).std()
    sig['dn_mkt_beta_60d'] = _beta(r, mkt.clip(upper=0.0))
    cn10y = panels['CN10Y']['close'].astype(float)
    sig['rate_beta_cn10y_60d'] = _beta(r, cn10y.pct_change())
    sig['mom_10d_skip5'] = c.shift(5)/c.shift(15) - 1.0
    sig['mom_120d_skip5'] = c.shift(5)/c.shift(125) - 1.0
    sig['mom60_skip5_voladj'] = (c.shift(5)/c.shift(65)-1.0) / r.rolling(60).std()
    med60 = (c.shift(5)/c.shift(65)-1.0).median(axis=1)
    sig['mom_vs_median_60d'] = (c.shift(5)/c.shift(65)-1.0).sub(med60, axis=0)
    # rsi_14 (Wilder approx with ewm)
    delta = c.diff()
    up = delta.clip(lower=0.0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-delta.clip(upper=0.0)).ewm(alpha=1/14, adjust=False).mean()
    sig['rsi_14'] = 100 - 100/(1 + up/dn.replace(0, np.nan))
    sig['hl_pos_20d'] = (c - c.rolling(20).min()) / (c.rolling(20).max() - c.rolling(20).min()).replace(0, np.nan)
    sig['kurt_60d'] = r.rolling(60).kurt()
    sig['max_dd_60d'] = c.rolling(60).max() / c - 1.0
    pos = r.clip(lower=0.0); neg = (-r.clip(upper=0.0))
    sig['downside_ratio_60d'] = neg.rolling(60).mean() / pos.rolling(60).mean().replace(0, np.nan)
    sig['corr_asset_mkt_20'] = r.rolling(20).corr(mkt)
    sig['corr_asset_mkt_60'] = r.rolling(60).corr(mkt)
    sig['vol_of_vol20x60'] = r.rolling(20).std().rolling(60).std()
    sig['vol_price_corr_20'] = r.rolling(20).corr(c.pct_change(20))
    sig['vol_ratio_20_60'] = r.rolling(20).std() / r.rolling(60).std().replace(0, np.nan)
    v = panels.get('VIX')
    if v is not None:
        vix = v['close'].astype(float)
        sig['vix_beta_cond_60x20'] = -_beta(r, vix.pct_change()) * (vix/vix.shift(20)-1.0).reindex(r.index).ffill()
    for nm, key in [('usdcny_beta_60d','USDCNY'), ('eurusd_beta_60d','EURUSD'), ('us10y_cond_beta_60d','US10Y')]:
        if nm == 'us10y_cond_beta_60d':
            s = panels['US10Y']['close'].astype(float).pct_change()
            sig[nm] = _beta(r, s)
        else:
            p = panels.get(key)
            if p is not None:
                sig[nm] = _beta(r, p['close'].astype(float).pct_change())
    return sig

def max_library_corr(cand, library):
    best, best_key = 0.0, None
    cs = cand.stack().rename('c')
    for name, lib in library.items():
        both = pd.concat([cs, lib.stack().rename('l')], axis=1).dropna()
        if len(both) < 100:
            continue
        rr = float(both['c'].corr(both['l']))
        if abs(rr) > best:
            best, best_key = abs(rr), name
    return round(best, 4), best_key

def full_eval(F, closes, library, name, expected_sign=1, horizons=(1,2,3,5,10,20)):
    R = fwd_returns(closes, 10)
    ics = rank_ic_series(F, R)
    s = summarize(ics, expected_sign)
    print(f'===== {name} (expected_sign={expected_sign:+d}) =====')
    if s is None:
        print('  NO VALID IC OBS'); return None
    print(f"  h=10: IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit']:.3f} n={s['n']} std={s['ic_std']:.3f}")
    dec = {}
    for h in horizons:
        rh = fwd_returns(closes, h)
        ih = rank_ic_series(F, rh)
        dec[str(h)] = round(float(ih.mean()), 4) if len(ih) else None
    print('  decay:', dec)
    cov = coverage_turnover(F)
    print(f"  coverage: asset_days={cov['cov_asset_days']:.3f} dates_ge8={cov['cov_dates_ge8']:.3f} turnover_10d_rank={cov['turnover_10d_rank']}")
    y = yearly_ic(ics)
    print('  yearly(IC,ICIR,n):', y)
    corr, key = max_library_corr(F, library)
    print(f'  max_abs_library_corr={corr} (vs {key})')
    rec = {'factor_id': name, 'ic': s['ic'], 'icir': s['icir'], 'hit': s['ic_hit'],
           'n_dates': s['n'], 'decay': dec, **cov, 'yearly': {k: list(v) for k, v in y.items()},
           'max_abs_library_correlation': corr, 'max_corr_factor': key,
           'expected_sign': expected_sign,
           'pass_gate': abs(s['ic']) >= IC_TH and abs(s['icir']) >= ICIR_TH}
    return rec
