"""miner_1 2028-10-05: re-validation / drift check of current effective library factors.

Windows:
  W1 warm-up      2020-01-01..2026-07-15 (admission baseline)
  W2 online       2026-07-16..2028-10-04 (out-of-sample since online start)
  W3 recent 12M   2027-10-05..2028-10-04 (latest regime)
  W4 recent 6M    2028-04-06..2028-10-04 (granular drift check)

Gate: |IC| >= 0.007, |ICIR| >= 0.084 (h=10). Metrics on ALIGNED signal
(raw signal * expected_direction from library JSON).
"""
import sys, time, json, glob
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import load_prices, load_index, WATCHLIST

t0 = time.time()
prices = load_prices(days=4200)
dxy = load_index('DXY', days=4200, prices=prices)
vix = load_index('VIX', days=4200, prices=prices)
usdjpy = load_index('USDJPY', days=4200, prices=prices)
usdcny = load_index('USDCNY', days=4200, prices=prices)
eurusd = load_index('EURUSD', days=4200, prices=prices)
print(f"data loaded {time.time()-t0:.1f}s")

def rb(df, mkt, win=60, minp=30, diff=False):
    r = df['close'].pct_change()
    m = mkt.diff() if diff else mkt.pct_change()
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    b = z['r'].rolling(win, min_periods=minp).cov(z['m']) / z['m'].rolling(win, min_periods=minp).var()
    return b.reindex(z.index)

def f_down_beta(df, s):
    spx = prices['SPX']['close']; r = df['close'].pct_change(); rs = spx.pct_change()
    z = pd.concat([r.rename('r'), rs.rename('m')], axis=1).dropna()
    down = z[z['m'] < 0]
    b = down['r'].rolling(60, min_periods=30).cov(down['m']) / down['m'].rolling(60, min_periods=30).var()
    return b.reindex(z.index)
def f_cn10y_beta(df, s): return rb(df, prices['CN10Y']['close'], diff=True)
def f_spx_beta(df, s): return rb(df, prices['SPX']['close'])
def f_hs300_beta(df, s): return rb(df, prices['000300.SH']['close'])
def f_comm_beta(df, s):
    bk = pd.concat([prices['XAU']['close'].pct_change(), prices['COPPER']['close'].pct_change(),
                    prices['WTI']['close'].pct_change()], axis=1).mean(axis=1)
    return rb(df, bk)
def f_dxy_cond(df, s):
    if dxy is None: return None
    b = rb(df, dxy['close'])
    cond = dxy['close'] / dxy['close'].shift(20) - 1.0
    return (b * cond).reindex(pd.concat([df['close'].pct_change(), dxy['close'].pct_change()], axis=1).dropna().index)
def f_vix_cond(df, s):
    if vix is None: return None
    b = rb(df, vix['close'])
    cond = vix['close'] / vix['close'].shift(20) - 1.0
    return (b * cond).reindex(pd.concat([df['close'].pct_change(), vix['close'].pct_change()], axis=1).dropna().index)
def f_eur_cond(df, s):
    if eurusd is None: return None
    b = rb(df, eurusd['close'])
    cond = eurusd['close'] / eurusd['close'].shift(20) - 1.0
    return (b * cond).reindex(pd.concat([df['close'].pct_change(), eurusd['close'].pct_change()], axis=1).dropna().index)
def f_vol_adj_mom(df, s):
    mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
    vol = df['close'].pct_change().rolling(60).std()
    return mom / vol
def f_hilo_vol_ratio(df, s):
    hi = df['close'].rolling(20).max(); lo = df['close'].rolling(20).min()
    rng = (hi - lo) / df['close']; vol = df['close'].pct_change().rolling(20).std()
    return rng / vol
def f_intraday_skew(df, s):
    return (df['close'] / df['open'] - 1.0).rolling(20).skew()
def f_vov(df, s):
    return df['close'].pct_change().rolling(20).std().rolling(60).std()
def f_copper_gold(df, s):
    cg = prices['COPPER']['close'].pct_change() - prices['XAU']['close'].pct_change()
    return rb(df, cg, win=20, minp=12)
def f_dd_duration(df, s):
    hi = df['close'].rolling(120).max()
    dur = np.log1p((df.index - pd.Series(df.index, index=df.index).where(df['close'] == hi).ffill()).dt.days.fillna(0))
    mom = df['close'].shift(5) / df['close'].shift(125) - 1.0
    zmom = (mom - mom.rolling(250).mean()) / mom.rolling(250).std()
    b = rb(df, prices['SPX']['close'])
    return dur - b * zmom
def f_hilo_pos(df, s):
    hi = df['close'].rolling(60).max(); lo = df['close'].rolling(60).min()
    return (df['close'] - lo) / (hi - lo)
def f_mom_accel(df, s):
    return df['close'].shift(5) / df['close'].shift(65) - 1.0 - (df['close'].shift(5) / df['close'].shift(125) - 1.0)
def f_range_skew(df, s):
    return ((df['high'] - df['low']) / df['close']).rolling(20).skew()
def f_sign_persist(df, s):
    r = df['close'].pct_change()
    return ((np.sign(r) == np.sign(r.shift(1))).astype(float)).rolling(20).mean()
def f_streak(df, s):
    r = df['close'].pct_change()
    up = (r > 0).astype(int); dn = (r < 0).astype(int)
    def streak(x):
        out = np.zeros(len(x)); c = 0
        for i in range(len(x)):
            c = c + 1 if x.iloc[i] else 0
            out[i] = c
        return pd.Series(out, index=x.index)
    su = streak(up); sd = streak(dn)
    return ((su - sd).rolling(60).max() / 60.0)
def f_vol_regime_switch(df, s):
    rv = df['close'].pct_change().rolling(20).std()
    above = (rv > rv.rolling(60).median()).astype(float)
    return above.diff().abs().rolling(60).mean()

FACTORS = {
    'down_beta_60': (f_down_beta, 1), 'cn10y_beta_60': (f_cn10y_beta, -1),
    'spx_beta_60': (f_spx_beta, 1), 'vol_adj_mom_20_60': (f_vol_adj_mom, 1),
    'dxy_beta_cond_60x20': (f_dxy_cond, 1), 'hs300_beta_60': (f_hs300_beta, -1),
    'hilo_vol_ratio_20': (f_hilo_vol_ratio, 1), 'intraday_ret_skew_20': (f_intraday_skew, 1),
    'comm_basket_beta_60': (f_comm_beta, 1), 'vol_of_vol20x60': (f_vov, 1),
    'vix_beta_cond_60x20': (f_vix_cond, 1), 'copper_gold_beta_20': (f_copper_gold, 1),
    'dd_duration_120_resid': (f_dd_duration, 1), 'eurusd_beta_cond_60x20': (f_eur_cond, 1),
    'hilo_pos_60': (f_hilo_pos, 1), 'mom_accel_60_120': (f_mom_accel, -1),
    'range_skew_20': (f_range_skew, -1), 'sign_persist_20': (f_sign_persist, 1),
    'streak_60': (f_streak, 1), 'vol_regime_switch_20x60': (f_vol_regime_switch, 1),
}

def panel_from(fn):
    cols = {}
    for s, df in prices.items():
        try:
            ser = fn(df, s)
            if ser is not None and len(ser) > 0:
                cols[s] = ser.astype(float)
        except Exception:
            pass
    p = pd.DataFrame(cols)
    return p[~p.index.duplicated(keep='last')].sort_index()

def ic_stats(panel, wstart, wend, h=10, min_valid=8):
    fwd = {s: df['close'].shift(-h) / df['close'] - 1.0 for s, df in prices.items()}
    fr = pd.DataFrame(fwd).sort_index()
    ic = {}
    common = panel.index.intersection(fr.index)
    for d in common:
        if d < wstart or d > wend: continue
        x, y = panel.loc[d], fr.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            ic[d] = x[m].rank().corr(y[m].rank())
    s = pd.Series(ic).sort_index()
    if len(s) < 60:
        return None
    mu, sd = s.mean(), s.std(ddof=1)
    return {'ic': float(mu), 'icir': float(mu/sd) if sd > 0 else 0.0,
            'hit': float((s > 0).mean()), 'n': int(len(s))}

WINDOWS = [
    ('W1_warmup', pd.Timestamp('2020-01-01'), pd.Timestamp('2026-07-15')),
    ('W2_online', pd.Timestamp('2026-07-16'), pd.Timestamp('2028-10-04')),
    ('W3_recent12m', pd.Timestamp('2027-10-05'), pd.Timestamp('2028-10-04')),
    ('W4_recent6m', pd.Timestamp('2028-04-06'), pd.Timestamp('2028-10-04')),
]

print(f"{'factor':<26} dir | " + " | ".join(f"{w[0]:>26}" for w in WINDOWS))
print("-" * 160)
out = {}
for fid, (fn, dirc) in FACTORS.items():
    panel = panel_from(fn)
    aligned = panel * dirc
    row = [f"{fid:<26} {dirc:>3}"]
    r_ = {}
    for wname, ws, we in WINDOWS:
        st = ic_stats(aligned, ws, we)
        if st is None:
            row.append(f"{'n<60':>26}")
        else:
            gate = 'PASS' if (abs(st['ic']) >= 0.007 and abs(st['icir']) >= 0.084) else 'fail'
            row.append(f"ic={st['ic']:+.4f} ir={st['icir']:+.3f} {gate:<4} n={st['n']}")
        r_[wname] = st
    out[fid] = r_
    print(" | ".join(row))

print(f"\ntotal {time.time()-t0:.1f}s")
