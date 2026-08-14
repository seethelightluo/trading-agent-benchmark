"""miner_1 2034-01-19 precise re-validation of existing library factors.

Re-implements factor formulas explicitly (the persisted 'calculation.expression'
fields are prose for several factors) and re-validates at admission horizon 10
on data through 2034-01-18. Reports full-sample and trailing-365d metrics,
direction drift vs the persisted expected_direction, and gate status.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
H = 10
MIN_ASSETS = 8
GATE_IC = 0.0070
GATE_ICIR = 0.0840

def load():
    out = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=4000)
        if df is None or len(df) < 300:
            continue
        df = df.copy(); df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        out[s] = df
    idx = None
    for s, df in out.items():
        idx = df.index if idx is None else idx.union(df.index)
    idx = idx.sort_values()
    C, O, Hi, Lo = {}, {}, {}, {}
    for s, df in out.items():
        C[s] = df['close'].astype(float).reindex(idx)
        O[s] = df['open'].astype(float).reindex(idx) if 'open' in df else pd.Series(np.nan, index=idx)
        Hi[s] = df['high'].astype(float).reindex(idx) if 'high' in df else pd.Series(np.nan, index=idx)
        Lo[s] = df['low'].astype(float).reindex(idx) if 'low' in df else pd.Series(np.nan, index=idx)
    return (pd.DataFrame(C), pd.DataFrame(O), pd.DataFrame(Hi), pd.DataFrame(Lo))

def ic_series(fdf, fwd):
    ics, dates = [], []
    for dt in fdf.index:
        x = fdf.loc[dt]; y = fwd.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_ASSETS:
            v = x[m].rank().corr(y[m].rank())
            if np.isfinite(v):
                ics.append(v); dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def summ(ic_s):
    if len(ic_s) < 5:
        return None
    ic = float(ic_s.mean()); icir = float(ic_s.mean()/ic_s.std())
    return {'ic': round(ic,4), 'icir': round(icir,3), 'hit': round(float((ic_s>0).mean()),3), 'n': len(ic_s),
            'pass': abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR}

def run(fname, fn, C, R, fwd, expected_dir=1):
    fdf = fn(C, R)
    fdf = fdf.reindex(columns=C.columns)
    ic_s = ic_series(fdf, fwd)
    full = summ(ic_s)
    tr = ic_s[ic_s.index >= ic_s.index[-1] - pd.Timedelta(days=365)] if len(ic_s) else ic_s
    recent = summ(tr)
    if full:
        drift = 'REVERSED' if (expected_dir is not None and np.sign(full['ic']) != np.sign(expected_dir) and abs(full['ic']) > GATE_IC) else ('OK' if full['pass'] else 'FAIL')
        print(f"{fname:24s} full ic={full['ic']:+.4f} icir={full['icir']:+.3f} hit={full['hit']:.3f} n={full['n']:4d} PASS={str(full['pass']):5s} | 365d ic={recent['ic'] if recent else float('nan'):+.4f} icir={recent['icir'] if recent else float('nan'):+.3f} | {drift}")
    return fname, full, recent

def main():
    C, O, Hi, Lo = load()
    R = C.pct_change()
    fwd = C.shift(-H) / C - 1.0
    print('grid', C.shape, C.index.min().date(), '->', C.index.max().date())

    def f_mom180(C, R): return C.shift(5) / C.shift(185) - 1.0
    def f_mom120(C, R): return C.shift(5) / C.shift(125) - 1.0
    def f_mom60(C, R):  return C.shift(5) / C.shift(65) - 1.0
    def f_mom20(C, R):  return C.shift(5) / C.shift(25) - 1.0
    def f_range252(C, R):
        hi = C.rolling(252).max(); lo = C.rolling(252).min()
        return (C - lo) / (hi - lo).replace(0, np.nan)
    def f_spxcorr(C, R): return R.rolling(60, min_periods=15).corr(R['SPX'])
    def f_downbeta(C, R):
        spx = R['SPX']; down = spx < 0
        out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for col in C.columns:
            a = R[col].to_numpy(); s = spx.to_numpy(); d = down.to_numpy()
            out[col] = _rolling_down_beta(a, s, d, 60, 15)
        return out
    def f_maxcon_gain(C, R):
        pos = (R > 0).astype(int)
        out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for col in C.columns:
            out[col] = _max_run(pos[col].to_numpy(), 21)
        return out
    def f_maxcon_loss(C, R):
        neg = (R < 0).astype(int)
        out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for col in C.columns:
            out[col] = _max_run(neg[col].to_numpy(), 21)
        return out
    def f_days_since_high60(C, R):
        out = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
        for col in C.columns:
            out[col] = _days_since_high(C[col].to_numpy(), 60)
        return out
    def f_calmness20(C, R):
        sd = R.rolling(20).std()
        return (R.abs() < 0.5 * sd).rolling(20).mean()
    def f_close_pos20(C, R):
        rng = (Hi - Lo).replace(0, np.nan)
        return ((C - Lo) / rng).rolling(20).mean()
    def f_intraday20(C, R):
        return (C / O - 1.0).rolling(20).mean()
    def f_volofvol(C, R):
        v = R.rolling(20).std(); return v.rolling(60).std() / v.rolling(60).mean()
    def f_volcluster(C, R):
        ar = R.abs(); return ar.rolling(60, min_periods=40).corr(ar.shift(1))
    def f_drawup20(C, R): return C / C.rolling(60).max() - 1.0
    def f_sharpe20(C, R):
        mu = R.rolling(20).mean(); sd = R.rolling(20).std()
        return mu / sd
    def f_hl_rank20(C, R):
        lo = C.rolling(20).min(); hi = C.rolling(20).max()
        return (C - lo) / (hi - lo).replace(0, np.nan)

    print('\n--- REVALIDATION (horizon %d) ---' % H)
    run('mom_180d_skip5', f_mom180, C, R, fwd, 1)
    run('mom_120d_skip5', f_mom120, C, R, fwd, 1)
    run('mom_60d_skip5', f_mom60, C, R, fwd, 1)
    run('mom_20d_skip5', f_mom20, C, R, fwd, 1)
    run('range_pos_252', f_range252, C, R, fwd, 1)
    run('spx_corr60', f_spxcorr, C, R, fwd, 1)
    run('downbeta_spx_60', f_downbeta, C, R, fwd, 1)
    run('max_consec_gain_20', f_maxcon_gain, C, R, fwd, 1)
    run('max_consec_loss_20', f_maxcon_loss, C, R, fwd, -1)
    run('days_since_high_60', f_days_since_high60, C, R, fwd, -1)
    run('calmness_20', f_calmness20, C, R, fwd, 1)
    run('close_pos_20', f_close_pos20, C, R, fwd, 1)
    run('intraday_drift_20', f_intraday20, C, R, fwd, 1)
    run('vol_of_vol20x60', f_volofvol, C, R, fwd, 1)
    run('volcluster_60', f_volcluster, C, R, fwd, 1)
    run('drawup_60(drawdown)', f_drawup20, C, R, fwd, 1)

def _rolling_down_beta(a, s, d, win, min_down):
    n = len(a); out = np.full(n, np.nan)
    for t in range(win, n):
        sl = slice(t-win, t)
        dd = d[sl]
        if dd.sum() >= min_down:
            x = s[sl][dd]; y = a[sl][dd]
            if x.std() > 0 and y.std() > 0:
                out[t] = np.cov(x, y)[0, 1] / np.var(x)
    return out

def _max_run(v, win):
    n = len(v); out = np.full(n, np.nan)
    run = 0
    for t in range(n):
        run = run + 1 if v[t] == 1 else 0
        if t >= win - 1:
            w = v[t-win+1:t+1]
            # longest consecutive run ending anywhere in window
            best = 0; cur = 0
            for x in w:
                cur = cur + 1 if x == 1 else 0
                best = max(best, cur)
            out[t] = best
    return out

def _days_since_high(v, win):
    n = len(v); out = np.full(n, np.nan)
    last_hi_idx = -1; cur_max = -np.inf
    for t in range(n):
        if np.isnan(v[t]):
            continue
        if t - last_hi_idx > win or np.isnan(cur_max):
            seg = v[max(0, t-win+1):t+1]
            if np.isnan(seg).all():
                continue
            cur_max = np.nanmax(seg); last_hi_idx = t - int(np.nanargmax(seg[::-1]))  # approx
            # simpler: recompute argmax from scratch each step
        # brute force per row for correctness on window
        seg = v[max(0, t-win+1):t+1]
        if np.isnan(seg).all():
            continue
        arg = np.nanargmax(seg)
        out[t] = (len(seg) - 1) - arg
    return out

if __name__ == '__main__':
    main()
