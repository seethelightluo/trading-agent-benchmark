"""miner_1 2028-02-24: re-validation / drift check of current effective library factors.

Computes IC / ICIR at h=10 for each currently-effective factor on three windows:
  W1 warm-up    2020-01-01..2026-07-15 (library admission window, baseline)
  W2 online     2026-07-16..2028-02-23 (out-of-sample since online start)
  W3 recent 12M 2027-02-24..2028-02-23 (latest regime)

Gate: |IC| >= 0.007, |ICIR| >= 0.084 (h=10). Sign = raw IC sign; expected direction noted.
"""
import sys, time
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import load_prices, load_index, WATCHLIST

t0 = time.time()
prices = load_prices(days=4200)
dxy = load_index('DXY', days=4200, prices=prices)
vix = load_index('VIX', days=4200, prices=prices)
print(f"data loaded {time.time()-t0:.1f}s")

# ---------------- factor definitions (match persisted library) ----------------
def f_down_beta(df, s):
    spx = prices['SPX']['close']; r = df['close'].pct_change(); rs = spx.pct_change()
    z = pd.concat([r.rename('r'), rs.rename('m')], axis=1).dropna()
    down = z[z['m'] < 0]
    b = down['r'].rolling(60, min_periods=30).cov(down['m']) / down['m'].rolling(60, min_periods=30).var()
    return b.reindex(z.index)

def f_cn10y_beta(df, s):
    cy = prices['CN10Y']['close']; r = df['close'].pct_change(); dy = cy.diff()
    z = pd.concat([r.rename('r'), dy.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=40).cov(z['m']) / z['m'].rolling(60, min_periods=40).var()
    return b.reindex(z.index)

def f_us10y_beta(df, s):
    uy = prices['US10Y']['close']; r = df['close'].pct_change(); dy = uy.diff()
    z = pd.concat([r.rename('r'), dy.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=40).cov(z['m']) / z['m'].rolling(60, min_periods=40).var()
    return b.reindex(z.index)

def f_spx_beta(df, s):
    spx = prices['SPX']['close']; r = df['close'].pct_change(); rs = spx.pct_change()
    z = pd.concat([r.rename('r'), rs.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=40).cov(z['m']) / z['m'].rolling(60, min_periods=40).var()
    return b.reindex(z.index)

def f_hs300_beta(df, s):
    h = prices['000300.SH']['close']; r = df['close'].pct_change(); rh = h.pct_change()
    z = pd.concat([r.rename('r'), rh.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=40).cov(z['m']) / z['m'].rolling(60, min_periods=40).var()
    return b.reindex(z.index)

def f_vol_adj_mom(df, s):
    mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
    vol = df['close'].pct_change().rolling(60).std()
    return mom / vol

def f_dxy_beta_cond(df, s):
    if dxy is None: return None
    r = df['close'].pct_change(); rd = dxy['close'].pct_change()
    z = pd.concat([r.rename('r'), rd.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=40).cov(z['m']) / z['m'].rolling(60, min_periods=40).var()
    cond = dxy['close'] / dxy['close'].shift(20) - 1.0
    return (-b * cond).reindex(z.index)  # negate to align sign convention with persisted factor

def f_hilo_vol_ratio(df, s):
    hi = df['close'].rolling(20).max(); lo = df['close'].rolling(20).min()
    rng = (hi - lo) / df['close']
    vol = df['close'].pct_change().rolling(20).std()
    return rng / vol

def f_intraday_skew(df, s):
    g = df['close'] / df['open'] - 1.0
    return g.rolling(20).skew()

def f_comm_basket_beta(df, s):
    bk = pd.concat([prices['XAU']['close'].pct_change().rename('x'),
                    prices['COPPER']['close'].pct_change().rename('c'),
                    prices['WTI']['close'].pct_change().rename('w')], axis=1).mean(axis=1)
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), bk.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=40).cov(z['m']) / z['m'].rolling(60, min_periods=40).var()
    return b.reindex(z.index)

def f_vov(df, s):
    return df['close'].pct_change().rolling(20).std().rolling(60).std()

FACTORS = {
    'down_beta_60': (f_down_beta, 1),
    'cn10y_beta_60': (f_cn10y_beta, -1),
    'spx_beta_60': (f_spx_beta, 1),
    'vol_adj_mom_20_60': (f_vol_adj_mom, 1),
    'dxy_beta_cond_60x20': (f_dxy_beta_cond, 1),
    'hs300_beta_60': (f_hs300_beta, -1),
    'hilo_vol_ratio_20': (f_hilo_vol_ratio, 1),
    'intraday_ret_skew_20': (f_intraday_skew, 1),
    'comm_basket_beta_60': (f_comm_basket_beta, 1),
    'vol_of_vol20x60': (f_vov, 1),
    'us10y_beta_60': (f_us10y_beta, 1),  # extra library factor (exists as json)
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
    """mean cross-sectional Spearman IC, ICIR, hit ratio on window."""
    fwd = {}
    for s, df in prices.items():
        fwd[s] = df['close'].shift(-h) / df['close'] - 1.0
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
            'hit': float((s > 0).mean()), 'n': int(len(s)),
            'start': str(s.index.min().date()), 'end': str(s.index.max().date())}

WINDOWS = [
    ('W1_warmup', pd.Timestamp('2020-01-01'), pd.Timestamp('2026-07-15')),
    ('W2_online', pd.Timestamp('2026-07-16'), pd.Timestamp('2028-02-23')),
    ('W3_recent12m', pd.Timestamp('2027-02-24'), pd.Timestamp('2028-02-23')),
]

print(f"{'factor':<24} {'dir':>3} | " + " | ".join(f"{w[0]:>18}" for w in WINDOWS))
print("-" * 130)
results = {}
for fid, (fn, dirc) in FACTORS.items():
    panel = panel_from(fn)
    row = [f"{fid:<24} {dirc:>3}"]
    r_ = {}
    for wname, ws, we in WINDOWS:
        st = ic_stats(panel, ws, we)
        if st is None:
            row.append(f"{'n<60':>18}")
        else:
            gate = 'PASS' if (abs(st['ic']) >= 0.007 and abs(st['icir']) >= 0.084) else 'fail'
            row.append(f"ic={st['ic']:+.4f} ir={st['icir']:+.3f} {gate:<4} n={st['n']}")
        r_[wname] = st
    results[fid] = r_
    print(" | ".join(row))

print(f"\ntotal {time.time()-t0:.1f}s")
