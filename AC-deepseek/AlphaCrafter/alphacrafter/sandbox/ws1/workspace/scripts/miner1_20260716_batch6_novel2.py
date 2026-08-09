"""miner_1 batch-6: explore NEW novel factor families beyond batch-5.

Library (12 EFFECTIVE): amihud_liquidity_20d, btc_spill_cond_60x20,
consec_up_ratio_20, dxy_cond_60x20, eff_ratio_60d, max_ratio_20,
mom_10d_skip5, mom_120d_skip5, ret_autocorr_20d, usdjpy_beta_cond_60x20,
vix_beta_cond_60x20, vol_of_vol20x60.

Admission gate at h=10: |IC| >= 0.007, |ICIR| >= 0.084,
max_abs_library_correlation < 0.50.
"""
import time
import numpy as np
import pandas as pd
from miner1_20260716_lib import (build_panel, factor_values, forward_returns,
                                 WATCH, DATA_START, WARMUP_END)
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

H = 10
MIN_VALID = 8
t0 = time.time()

panel = build_panel()
closes, volumes, grid = panel['closes'], panel['volumes'], panel['grid']
print(f'panel built in {time.time()-t0:.0f}s grid_dates={len(grid)} assets={len(closes)}')

# ---------- OHLC panel ----------
def _fetch_ohlc(sym):
    try:
        df = get_stock_daily_data(symbol=sym, days=4000)
    except Exception:
        df = get_index_daily_data(symbol=sym, days=4000)
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= DATA_START) & (df['date'] <= WARMUP_END)].sort_values('date')
    return df.set_index('date')

ohlc = {}
for sym in WATCH:
    d = _fetch_ohlc(sym)
    if d is not None and {'open', 'high', 'low', 'close'}.issubset(d.columns):
        ohlc[sym] = d[['open', 'high', 'low', 'close']].astype(float)
print(f'ohlc assets: {len(ohlc)}')

# ---------- candidate factor constructors ----------
def rsi14(sym, close, volume):
    r = close.diff()
    up = r.clip(lower=0.0).rolling(14).mean()
    dn = (-r.clip(upper=0.0)).rolling(14).mean()
    rs = up / dn
    return (100.0 - 100.0 / (1.0 + rs)).replace([np.inf, -np.inf], np.nan)

def hl_pos(win=20):
    def fn(sym, close, volume):
        hi = close.rolling(win).max()
        lo = close.rolling(win).min()
        return ((close - lo) / (hi - lo)).replace([np.inf, -np.inf], np.nan)
    return fn

def drawdown(win=60):
    def fn(sym, close, volume):
        return (close / close.rolling(win).max() - 1.0).replace([np.inf, -np.inf], np.nan)
    return fn

def vol_trend(short=20, long=60):
    def fn(sym, close, volume):
        if volume is None:
            return None
        vs = volume.rolling(short).mean()
        vl = volume.rolling(long).mean()
        return (vs / vl - 1.0).replace([np.inf, -np.inf], np.nan)
    return fn

def win_rate(win=20):
    def fn(sym, close, volume):
        return (close.pct_change() > 0).astype(float).rolling(win).mean()
    return fn

def skew(win=20):
    def fn(sym, close, volume):
        return close.pct_change().rolling(win, min_periods=15).skew().replace([np.inf, -np.inf], np.nan)
    return fn

def eq_beta(win=60):
    def fn(sym, close, volume):
        spx = closes.get('SPX')
        if spx is None or sym == 'SPX':
            return None
        g = grid
        r_a = close.pct_change().reindex(g)
        r_m = spx.pct_change().reindex(g)
        beta = r_a.rolling(win, min_periods=30).cov(r_m) / r_m.rolling(win, min_periods=30).var()
        return beta.replace([np.inf, -np.inf], np.nan)
    return fn

def gk_ratio(win=20):
    def fn(sym, close, volume):
        d = ohlc.get(sym)
        if d is None or len(d) < win + 5:
            return None
        o, h, l, c = d['open'], d['high'], d['low'], d['close']
        gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
        gk_v = np.sqrt(gk.rolling(win).mean())
        cc_v = close.pct_change().rolling(win).std()
        return (gk_v / cc_v).replace([np.inf, -np.inf], np.nan)
    return fn

def volconf(win=20):
    def fn(sym, close, volume):
        if volume is None:
            return None
        r = close.pct_change()
        return r.rolling(win, min_periods=15).corr(volume).replace([np.inf, -np.inf], np.nan)
    return fn

def shadow_ratio(upper=True, win=20):
    def fn(sym, close, volume):
        d = ohlc.get(sym)
        if d is None or len(d) < win + 5:
            return None
        o, h, l = d['open'], d['high'], d['low']
        rng = (h - l).replace(0, np.nan)
        if upper:
            s = (h - np.maximum(o, close)) / rng
        else:
            s = (np.minimum(o, close) - l) / rng
        return s.rolling(win).mean().replace([np.inf, -np.inf], np.nan)
    return fn

def macro_beta_cond(macro_key, sign=1.0, win=60, mom=20):
    def fn(sym, close, volume):
        macro = panel['macro'].get(macro_key)
        if macro is None:
            macro = panel['closes'].get(macro_key)
        if macro is None:
            return None
        g = panel['grid']
        r_a = close.pct_change().reindex(g)
        r_m = macro.pct_change().reindex(g)
        beta = r_a.rolling(win, min_periods=30).cov(r_m) / r_m.rolling(win, min_periods=30).var()
        mm = macro.reindex(g) / macro.shift(mom).reindex(g) - 1.0
        return (sign * beta * mm).replace([np.inf, -np.inf], np.nan)
    return fn

def macd(short=20, long=60):
    def fn(sym, close, volume):
        s = close.rolling(short).mean()
        l = close.rolling(long).mean()
        return (s / l - 1.0).replace([np.inf, -np.inf], np.nan)
    return fn

def gap_freq(win=20, thr=0.01):
    def fn(sym, close, volume):
        d = ohlc.get(sym)
        if d is None or len(d) < win + 5:
            return None
        gap = (d['open'] / d['close'].shift(1) - 1.0).abs()
        return (gap > thr).astype(float).rolling(win).mean()
    return fn

CANDIDATES = [
    ('rsi14', rsi14),
    ('hl_pos_20', hl_pos(20)),
    ('drawdown_60', drawdown(60)),
    ('vol_trend_20x60', vol_trend(20, 60)),
    ('win_rate_20', win_rate(20)),
    ('skew_20', skew(20)),
    ('eq_beta_60', eq_beta(60)),
    ('gk_ratio_20', gk_ratio(20)),
    ('volconf_20', volconf(20)),
    ('high_shadow_20', shadow_ratio(upper=True, win=20)),
    ('low_shadow_20', shadow_ratio(upper=False, win=20)),
    ('copper_spill_cond_60x20', macro_beta_cond('COPPER', sign=1.0)),
    ('ndx_spill_cond_60x20', macro_beta_cond('NDX', sign=1.0)),
    ('macd_20x60', macd(20, 60)),
    ('gap_freq_20', gap_freq(20, 0.01)),
]

# ---------- fast vectorized daily rank IC ----------
def fast_daily_ic(fac, ret, min_valid=MIN_VALID):
    idx = fac.index.intersection(ret.index)
    fac = fac.reindex(columns=ret.columns)
    F = fac.loc[idx].values
    R = ret.loc[idx].values
    out = []
    for i in range(len(idx)):
        f, r = F[i], R[i]
        m = np.isfinite(f) & np.isfinite(r)
        if m.sum() < min_valid:
            continue
        fv, rv = f[m], r[m]
        if np.unique(fv).size < 2 or np.unique(rv).size < 2:
            continue
        fr = pd.Series(fv).rank().values
        rr = pd.Series(rv).rank().values
        fr = fr - fr.mean(); rr = rr - rr.mean()
        denom = np.sqrt((fr * fr).sum() * (rr * rr).sum())
        if denom == 0 or not np.isfinite(denom):
            continue
        ic = float((fr * rr).sum() / denom)
        out.append((idx[i], ic, int(m.sum())))
    return pd.DataFrame(out, columns=['date', 'ic', 'n']).set_index('date') if out else pd.DataFrame(columns=['ic', 'n'])

def summarize_fast(label, ics, h, turnover=None, coverage=None):
    if len(ics) == 0:
        print(f'[{label}] NO VALID IC DATES')
        return None
    ic = ics['ic']
    mean_ic = float(ic.mean())
    std_ic = float(ic.std(ddof=1)) if len(ic) > 1 else float('nan')
    icir = mean_ic / std_ic if std_ic and np.isfinite(std_ic) and std_ic > 0 else float('nan')
    hit = float((ic > 0).mean())
    print(f'[{label}] h={h} dates={len(ic)} med_n={int(ics["n"].median())} '
          f'IC={mean_ic:+.4f} ICIR={icir:+.3f} hit={hit:.2f} '
          f'turn={turnover:.3f} cov={coverage:.3f}')
    return {'label': label, 'horizon': h, 'dates': len(ic), 'ic': mean_ic,
            'icir': icir, 'hit': hit, 'std': std_ic, 'turnover': turnover, 'coverage': coverage}

def max_lib_corr(fac, libs):
    best = (0.0, None)
    for l, lf in libs.items():
        common = fac.index.intersection(lf.index)
        F = fac.loc[common].values
        L = lf.loc[common].values
        cs = []
        for i in range(len(common)):
            f, g = F[i], L[i]
            m = np.isfinite(f) & np.isfinite(g)
            if m.sum() < MIN_VALID:
                continue
            fv = pd.Series(f[m]).rank().values
            gv = pd.Series(g[m]).rank().values
            fv = fv - fv.mean(); gv = gv - gv.mean()
            denom = np.sqrt((fv * fv).sum() * (gv * gv).sum())
            if denom == 0 or not np.isfinite(denom):
                continue
            cs.append(float((fv * gv).sum() / denom))
        v = float(np.mean(cs)) if cs else np.nan
        if np.isfinite(v) and abs(v) > abs(best[0]):
            best = (v, l)
    return best

# ---------- library factor frames (all 12) ----------
def _amihud(sym, close, volume):
    if volume is None:
        return None
    return -((close.pct_change().abs() / volume).rolling(20).mean())

def _consec_up(sym, close, volume):
    r = (close.pct_change() > 0).astype(float)
    def run_len(x):
        x = x.values
        best_up = best_dn = cur_u = cur_d = 0
        for v in x:
            if v == 1:
                cur_u += 1; cur_d = 0
                best_up = max(best_up, cur_u)
            else:
                cur_d += 1; cur_u = 0
                best_dn = max(best_dn, cur_d)
        s = best_up + best_dn
        return best_up / s if s > 0 else np.nan
    return r.rolling(20).apply(run_len, raw=False)

def _autocorr(sym, close, volume):
    r = close.pct_change()
    return -r.rolling(20, min_periods=15).apply(
        lambda x: x.autocorr(lag=1) if np.isfinite(x.autocorr(lag=1)) else np.nan, raw=False)

LIB_FNS = {
    'amihud_liquidity_20d': _amihud,
    'btc_spill_cond_60x20': macro_beta_cond('BTC', sign=1.0),
    'consec_up_ratio_20': _consec_up,
    'dxy_cond_60x20': macro_beta_cond('DXY', sign=1.0),
    'eff_ratio_60d': lambda s, c, v: (c - c.shift(60)).abs() / c.diff().abs().rolling(60).sum(),
    'max_ratio_20': lambda s, c, v: c.pct_change().rolling(20).max() / c.pct_change().rolling(20).min().abs(),
    'mom_10d_skip5': lambda s, c, v: c.shift(5) / c.shift(15) - 1.0,
    'mom_120d_skip5': lambda s, c, v: c.shift(5) / c.shift(125) - 1.0,
    'ret_autocorr_20d': _autocorr,
    'usdjpy_beta_cond_60x20': macro_beta_cond('USDJPY', sign=1.0),
    'vix_beta_cond_60x20': macro_beta_cond('VIX', sign=-1.0),
    'vol_of_vol20x60': lambda s, c, v: c.pct_change().rolling(20).std().rolling(60).std(),
}
lib_frames = {l: factor_values(closes, volumes, grid, fn) for l, fn in LIB_FNS.items()}
print(f'library frames built in {time.time()-t0:.0f}s')

# ---------- stage 1: h=10 gate ----------
print(f'\n=== STAGE 1: H={H} GATE (|IC|>=0.007, |ICIR|>=0.084) ===')
ret10 = forward_returns(closes, grid, H)
frames, metrics = {}, {}
for label, fn in CANDIDATES:
    fac = factor_values(closes, volumes, grid, fn)
    frames[label] = fac
    cov = float(fac.notna().mean().mean())
    f10 = fac.iloc[::10]
    turn = float(f10.rank(axis=1).diff().abs().mean().mean()) if len(f10) > 2 else np.nan
    ics = fast_daily_ic(fac, ret10)
    m = summarize_fast(label, ics, H, turnover=turn, coverage=cov)
    metrics[label] = m
print(f'stage 1 done in {time.time()-t0:.0f}s')

# ---------- stage 2: passers -> decay + library corr ----------
passers = [l for l, m in metrics.items() if m is not None
           and abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084]
print(f'\n=== STAGE 2: IC/ICIR passers = {passers} ===')
ret_decay = {h: forward_returns(closes, grid, h) for h in (1, 2, 3, 5, 10, 20)}
for label in passers:
    fac = frames[label]
    parts = []
    for h in (1, 2, 3, 5, 10, 20):
        ics = fast_daily_ic(fac, ret_decay[h])
        parts.append(f'h{h}:{ics["ic"].mean():+.4f}' if len(ics) else f'h{h}:nan')
    print(f'  {label}: ' + ' '.join(parts))
    v, l = max_lib_corr(fac, lib_frames)
    metrics[label]['max_lib_corr'] = abs(v)
    metrics[label]['max_lib_corr_vs'] = l
    print(f'  {label}: max_abs_lib_corr={abs(v):.4f} (vs {l})')

print('\n=== PASS/FAIL SUMMARY (corr < 0.50) ===')
for label, m in metrics.items():
    if m is None:
        continue
    ic_ok = abs(m['ic']) >= 0.007
    icir_ok = abs(m['icir']) >= 0.084
    corr_ok = m.get('max_lib_corr', 0.0) < 0.50
    status = 'PASS' if (ic_ok and icir_ok and corr_ok) else 'FAIL'
    print(f"  {status} {label}: IC={m['ic']:+.4f} ICIR={m['icir']:+.3f} "
          f"corr={m.get('max_lib_corr', float('nan')):.3f}")
print(f'total time {time.time()-t0:.0f}s')
