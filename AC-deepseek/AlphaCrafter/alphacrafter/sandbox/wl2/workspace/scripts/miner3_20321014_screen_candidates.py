"""miner_3 screening: explore novel factor candidates on 15-instrument cross-asset universe.
Data through 2032-10-13 (current sim date 2032-10-14). h=10 admission horizon.
NOTE: 6 instruments (HSI,SX5E,BTC,US10Y,CN10Y,000300.SH) are flat-feed (zero returns) in
recent period; 9 live. Report both full-15 and live-9 IC for transparency."""
import sys; sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from miner3_util import load_all, close_panel, ret_panel, forward_ret, daily_spearman_ic, rank_turnover

data = load_all(3000)
px = close_panel(data)
rets = ret_panel(data)
LIVE = ['SPX','N225','000688.SH','SOX','NDX','XAU','COPPER','WTI','ETH']

def zscore_cs(x):
    m = x.mean(); s = x.std(ddof=0)
    return (x - m) / s if s > 0 else x * 0

def roll_mean(x, w):
    return x.rolling(w, min_periods=max(5, w//2)).mean()

def roll_std(x, w):
    return x.rolling(w, min_periods=max(5, w//2)).std(ddof=0)

# ---- candidate factor builders (date x symbol) ----
def f_xdisp_20(px, rets):
    r = rets.rolling(20, min_periods=10).sum()
    return r.apply(zscore_cs, axis=1)  # cross-sectional z of 20d return

def f_autocorr_60(rets):
    def ac(x):
        x = x.dropna()
        if len(x) < 30: return np.nan
        a = x.iloc[1:].values; b = x.iloc[:-1].values
        if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
        return np.corrcoef(a, b)[0, 1]
    return rets.rolling(60, min_periods=30).apply(ac, raw=False)

def f_volterm_20_60(rets):
    v20 = roll_std(rets, 20); v60 = roll_std(rets, 60)
    return v20 / v60

def f_semidev_ratio_60(rets):
    def sdr(x):
        x = x.dropna()
        if len(x) < 30: return np.nan
        d = x[x < 0]
        sd = np.sqrt((d**2).mean()) if len(d) > 5 else 0.0
        tot = x.std(ddof=0)
        return sd / tot if tot > 1e-12 else 0.0
    return rets.rolling(60, min_periods=30).apply(sdr, raw=False)

def f_gap_20(px, data):
    gaps = {}
    for s, d in data.items():
        g = (d.set_index('date')['open'] / d.set_index('date')['close'].shift(1) - 1).replace([np.inf,-np.inf], np.nan)
        gaps[s] = g
    g = pd.DataFrame(gaps).sort_index()
    return g.rolling(20, min_periods=10).mean()

def f_kurt_60(rets):
    def ku(x):
        x = x.dropna()
        if len(x) < 40: return np.nan
        if x.std(ddof=0) < 1e-12: return 0.0
        return pd.Series(x).kurt()
    return rets.rolling(60, min_periods=40).apply(ku, raw=False)

def f_drawdown_60(px):
    hi = px.rolling(60, min_periods=30).max()
    dd = px / hi - 1.0
    v60 = roll_std(ret_panel(px), 60)
    return dd / v60  # drawdown depth scaled by vol

def f_hi_lo_ratio_20(px, data):
    out = {}
    for s, d in data.items():
        d = d.set_index('date')
        hl = (d['high'] - d['low']) / d['close']
        out[s] = hl
    hlp = pd.DataFrame(out).sort_index()
    return hlp.rolling(20, min_periods=10).mean() / roll_std(ret_panel(px), 20)

CANDIDATES = {
    'xdisp_20': f_xdisp_20,
    'autocorr_60': f_autocorr_60,
    'volterm_20_60': f_volterm_20_60,
    'semidev_ratio_60': f_semidev_ratio_60,
    'gap_20': f_gap_20,
    'kurt_60': f_kurt_60,
    'drawdown_60': f_drawdown_60,
    'hi_lo_ratio_20': f_hi_lo_ratio_20,
}

fwd10 = forward_ret(px, 10)
fwd5 = forward_ret(px, 5)
fwd20 = forward_ret(px, 20)

print(f"Panel: {px.shape[0]} dates x {px.shape[1]} assets; forward-10 last valid date {fwd10.notna().sum(axis=1).gt(0).idxmax() if False else ''}")
for name, fn in CANDIDATES.items():
    try:
        f = fn(px, rets)
        f = f.reindex(index=px.index, columns=px.columns)
        # full-15 IC
        ic_full = daily_spearman_ic(f, fwd10, min_valid=8)
        # live-9 IC
        ic_live = daily_spearman_ic(f[LIVE], fwd10[LIVE], min_valid=5)
        def stats(ic):
            if len(ic) == 0: return None
            icv = ic['ic']
            return dict(n=len(ic), ic=icv.mean(), icir=icv.mean()/icv.std(ddof=1) if icv.std(ddof=1)>0 else np.nan,
                        hit=(icv>0).mean())
        s_full = stats(ic_full); s_live = stats(ic_live)
        cov = float(f.notna().mean().mean())
        to = rank_turnover(f, 10)
        # decay at h=5,20 (full-15)
        dec5 = daily_spearman_ic(f, fwd5, 8)['ic'].mean() if len(daily_spearman_ic(f, fwd5, 8)) else np.nan
        dec20 = daily_spearman_ic(f, fwd20, 8)['ic'].mean() if len(daily_spearman_ic(f, fwd20, 8)) else np.nan
        line = f"{name:16s} FULL n={s_full['n']:4d} ic={s_full['ic']:+.4f} icir={s_full['icir']:+.3f} hit={s_full['hit']:.2f} | LIVE n={s_live['n']:4d} ic={s_live['ic']:+.4f} icir={s_live['icir']:+.3f} | cov={cov:.2f} turn={to:.3f} decay5={dec5:+.4f} decay20={dec20:+.4f}"
        print(line)
    except Exception as e:
        print(name, 'ERR', repr(e))
