"""miner_1 batch-5 (optimized): explore NOVEL factor families not yet in the library.

Library (4 admitted): mom_10d_skip5, mom_120d_skip5, vix_beta_cond_60x20, vol_of_vol20x60.
Admission gate at h=10: |IC| >= 0.007 and |ICIR| >= 0.084, plus
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

# ---------- OHLC panel for range/gap candidates ----------
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


def with_panel(fn):
    def wrapped(sym, close, volume):
        return fn(sym, close, volume, panel=panel)
    return wrapped


def macro_beta_cond(macro_key, sign=1.0, win=60, mom=20, use_diff=False):
    def fn(sym, close, volume, panel=None):
        macro = panel['macro'].get(macro_key)
        if macro is None:
            macro = panel['closes'].get(macro_key)
        if macro is None:
            return None
        g = panel['grid']
        r_a = close.pct_change().reindex(g)
        if use_diff:
            r_m = macro.reindex(g).diff()
            mm = macro.reindex(g) - macro.shift(mom).reindex(g)
        else:
            r_m = macro.pct_change().reindex(g)
            mm = macro.reindex(g) / macro.shift(mom).reindex(g) - 1.0
        cov = r_a.rolling(win, min_periods=30).cov(r_m)
        var = r_m.rolling(win, min_periods=30).var()
        beta = cov / var
        return (sign * beta * mm).replace([np.inf, -np.inf], np.nan)
    return fn


def park_ratio(short=20):
    def fn(sym, close, volume):
        d = ohlc.get(sym)
        if d is None or len(d) < short + 5:
            return None
        hi, lo = d['high'], d['low']
        r = np.log(hi / lo) ** 2
        park = np.sqrt(r.rolling(short).mean() / (4.0 * np.log(2.0)))
        cvol = close.pct_change().rolling(short).std()
        return (park / cvol).replace([np.inf, -np.inf], np.nan)
    return fn


def overnight_share(win=60):
    def fn(sym, close, volume):
        d = ohlc.get(sym)
        if d is None or len(d) < win + 5:
            return None
        on = d['open'] / d['close'].shift(1) - 1.0
        cc = close.pct_change()
        v_on = on.rolling(win).var()
        v_cc = cc.rolling(win).var()
        return (v_on / v_cc).replace([np.inf, -np.inf], np.nan)
    return fn


def downside_vol_ratio(win=60):
    def fn(sym, close, volume):
        r = close.pct_change()
        dd = r.clip(upper=0.0).rolling(win).std()
        v = r.rolling(win).std()
        return (dd / v).replace([np.inf, -np.inf], np.nan)
    return fn


def skew_neg(win=60):
    def fn(sym, close, volume):
        return (-close.pct_change().rolling(win).skew()).replace([np.inf, -np.inf], np.nan)
    return fn


def max_ratio(win=20):
    def fn(sym, close, volume):
        r = close.pct_change().rolling(win)
        up = r.max()
        dn = r.min()
        return (up / dn.abs()).replace([np.inf, -np.inf], np.nan)
    return fn


def vol_ratio_10x60():
    def fn(sym, close, volume):
        r = close.pct_change()
        v10 = r.rolling(10).std()
        v60 = r.rolling(60).std()
        return (1.0 - v10 / v60).replace([np.inf, -np.inf], np.nan)  # high = calm
    return fn


def consec_up_ratio(win=20):
    def fn(sym, close, volume):
        r = (close.pct_change() > 0).astype(float)
        def run_len(x):
            x = x.values
            best_up = best_dn = cur_u = cur_d = 0
            for v in x:
                if v == 1:
                    cur_u += 1
                    cur_d = 0
                    best_up = max(best_up, cur_u)
                elif v == 0:
                    cur_d += 1
                    cur_u = 0
                    best_dn = max(best_dn, cur_d)
            s = best_up + best_dn
            return best_up / s if s > 0 else np.nan
        return r.rolling(win).apply(run_len, raw=False)
    return fn


CANDIDATES = [
    ('us10y_beta_cond_60x20', with_panel(macro_beta_cond('US10Y', sign=-1.0))),
    ('us10y_diff_beta_cond_60x20', with_panel(macro_beta_cond('US10Y', sign=-1.0, use_diff=True))),
    ('usdjpy_beta_cond_60x20', with_panel(macro_beta_cond('USDJPY', sign=1.0))),
    ('dxy_beta_cond_40x10', with_panel(macro_beta_cond('DXY', sign=-1.0, win=40, mom=10))),
    ('btc_spill_cond_60x20', with_panel(macro_beta_cond('BTC', sign=1.0))),
    ('wti_spill_cond_60x20', with_panel(macro_beta_cond('WTI', sign=1.0))),
    ('xau_spill_cond_60x20', with_panel(macro_beta_cond('XAU', sign=1.0))),
    ('park_ratio_20', park_ratio(20)),
    ('overnight_share_60', overnight_share(60)),
    ('downside_vol_ratio_60', downside_vol_ratio(60)),
    ('skew_neg_60', skew_neg(60)),
    ('max_ratio_20', max_ratio(20)),
    ('vol_ratio_10x60', vol_ratio_10x60()),
    ('consec_up_ratio_20', consec_up_ratio(20)),
]

# ---------- fast vectorized daily rank IC ----------
def fast_daily_ic(fac, ret, min_valid=MIN_VALID):
    idx = fac.index.intersection(ret.index)
    F = fac.loc[idx].values
    R = ret.loc[idx].values
    out = []
    for i in range(len(idx)):
        f, r = F[i], R[i]
        if len(f) == 0 or len(r) == 0:
            continue
        m = np.isfinite(f) & np.isfinite(r)
        if m.sum() < min_valid:
            continue
        fv, rv = f[m], r[m]
        if np.unique(fv).size < 2 or np.unique(rv).size < 2:
            continue
        fr = pd.Series(fv).rank().values
        rr = pd.Series(rv).rank().values
        fr = fr - fr.mean()
        rr = rr - rr.mean()
        denom = np.sqrt((fr * fr).sum() * (rr * rr).sum())
        if denom == 0 or not np.isfinite(denom):
            continue
        ic = float((fr * rr).sum() / denom)
        out.append((idx[i], ic, int(m.sum())))
    if not out:
        return pd.DataFrame(columns=['ic', 'n'])
    df = pd.DataFrame(out, columns=['date', 'ic', 'n']).set_index('date')
    return df


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


# ---------- library factor frames ----------
LIB_FNS = {
    'mom_10d_skip5': lambda s, c, v: c.shift(5) / c.shift(15) - 1.0,
    'mom_120d_skip5': lambda s, c, v: c.shift(5) / c.shift(125) - 1.0,
    'vol_of_vol20x60': lambda s, c, v: c.pct_change().rolling(20).std().rolling(60).std(),
}
lib_frames = {l: factor_values(closes, volumes, grid, fn) for l, fn in LIB_FNS.items()}


def vix_beta_frame(sym, close, volume):
    macro = panel['macro'].get('VIX')
    g = panel['grid']
    r_a = close.pct_change().reindex(g)
    r_m = macro.pct_change().reindex(g)
    beta = r_a.rolling(60, min_periods=30).cov(r_m) / r_m.rolling(60, min_periods=30).var()
    mm = macro.reindex(g) / macro.shift(20).reindex(g) - 1.0
    return (-1.0 * beta * mm).replace([np.inf, -np.inf], np.nan)


lib_frames['vix_beta_cond_60x20'] = factor_values(closes, volumes, grid, vix_beta_frame)
print(f'library frames built in {time.time()-t0:.0f}s')

# ---------- stage 1: build factor frames + h=10 gate ----------
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
print('DECAY (h=1..20):')
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

print('\n=== PASS/FAIL SUMMARY (incl. corr < 0.50) ===')
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
