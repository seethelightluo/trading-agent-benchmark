"""miner_1 2028-11-02 exploration batch: screen new cross-asset factor ideas.

Admission battery from factor_common (warm-up window 2020-01-01..2026-07-15,
h=10 daily paper IC). Robustness: also compute IC/ICIR on W3 (recent 12m) and
W4 (recent 6m) windows using live data through 2028-11-01.
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import factor_common as fc

t0 = time.time()
prices = fc.load_prices(days=2400)
indexes = {s: fc.load_index(s, days=2400, prices=prices) for s in fc.INDEX_SIGNALS}
print(f"data loaded {time.time()-t0:.1f}s; last {max(d.index.max() for d in prices.values()).date()}")

def ic_window(panel, fwd, wstart, wend, min_valid=8):
    common = panel.index.intersection(fwd.index)
    ic = {}
    for d in common:
        if d < wstart or d > wend:
            continue
        x = panel.loc[d]; y = fwd.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            ic[d] = x[m].rank().corr(y[m].rank())
    s = pd.Series(ic)
    if len(s) < 30:
        return np.nan, np.nan, len(s)
    return float(s.mean()), float(s.mean() / s.std(ddof=1)), len(s)

fwd10 = fc.forward_returns(prices, 10)

# ---------- candidate factor definitions ----------
def f_downside_vol_ratio(df, s, win=20):
    r = df['close'].pct_change()
    neg = r.where(r < 0, 0.0)
    ds = neg.rolling(win).std()
    tot = r.rolling(win).std()
    return (ds / tot).replace([np.inf, -np.inf], np.nan)

def f_max_drawdown_60(df, s, win=60):
    return df['close'] / df['close'].rolling(win, min_periods=30).max() - 1.0

def f_rsi_14_meanrev(df, s, win=14):
    d = df['close'].diff()
    up = d.clip(lower=0).rolling(win).mean()
    dn = (-d.clip(upper=0)).rolling(win).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return (rsi - 50) / 50.0

def f_overnight_gap_ratio_20(df, s, win=20):
    # overnight = open/prev_close-1 ; intraday = close/open-1
    ov = df['open'] / df['close'].shift(1) - 1.0
    idr = df['close'] / df['open'] - 1.0
    num = ov.rolling(win).mean()
    den = idr.abs().rolling(win).mean()
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def f_btc_beta_cond_60x20(df, s, prices):
    btc = prices['BTC']['close']
    r = df['close'].pct_change(); rb = btc.pct_change()
    z = pd.concat([r.rename('r'), rb.rename('b')], axis=1).dropna()
    beta = z['r'].rolling(60, min_periods=30).cov(z['b']) / z['b'].rolling(60, min_periods=30).var()
    trend = (btc / btc.shift(20) - 1.0).reindex(z.index)
    return (beta * np.sign(trend)).reindex(z.index)

def f_us10y_beta_cond_60x20(df, s, prices):
    y10 = prices['US10Y']['close']  # yield series
    r = df['close'].pct_change(); ry = y10.pct_change()
    z = pd.concat([r.rename('r'), ry.rename('y')], axis=1).dropna()
    beta = z['r'].rolling(60, min_periods=30).cov(z['y']) / z['y'].rolling(60, min_periods=30).var()
    trend = (y10 / y10.shift(20) - 1.0).reindex(z.index)
    return (beta * np.sign(trend)).reindex(z.index)

def f_rel_mom_vs_median_20(df, s, prices, win=20):
    panel = {k: v['close'].pct_change(win) for k, v in prices.items()}
    pm = pd.DataFrame(panel)
    med = pm.median(axis=1, skipna=True)
    out = df['close'].pct_change(win) - med
    return out

def f_tail_abs_move_20(df, s, win=20):
    r = df['close'].pct_change().abs()
    return r.rolling(win).max()

def f_range_vol_std_20(df, s, win=20):
    rng = (df['high'] - df['low']) / df['close']
    return rng.rolling(win).std()

def f_vol_ratio_5_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(5).std() / r.rolling(60).std()

cands = [
    ('dsvol_ratio_20', lambda df, s: f_downside_vol_ratio(df, s, 20), 'downside semideviation / total vol (vol asymmetry)'),
    ('max_dd_60', lambda df, s: f_max_drawdown_60(df, s, 60), 'depth of drawdown from 60d high'),
    ('rsi14_meanrev', lambda df, s: f_rsi_14_meanrev(df, s, 14), 'RSI(14) oscillator centered mean-reversion'),
    ('overnight_gap_ratio_20', lambda df, s: f_overnight_gap_ratio_20(df, s, 20), 'avg overnight ret / avg |intraday| ret'),
    ('btc_beta_cond_60x20', lambda df, s: f_btc_beta_cond_60x20(df, s, prices), '60d beta to BTC signed by BTC 20d trend'),
    ('us10y_beta_cond_60x20', lambda df, s: f_us10y_beta_cond_60x20(df, s, prices), '60d beta to US10Y yield signed by yield 20d trend'),
    ('rel_mom_vs_median_20', lambda df, s: f_rel_mom_vs_median_20(df, s, prices, 20), 'asset 20d return minus cross-sectional median'),
    ('tail_abs_move_20', lambda df, s: f_tail_abs_move_20(df, s, 20), 'max |daily return| over 20d (tail magnitude)'),
    ('range_vol_std_20', lambda df, s: f_range_vol_std_20(df, s, 20), 'std of daily (high-low)/close over 20d'),
    ('vol_ratio_5_60', lambda df, s: f_vol_ratio_5_60(df, s), 'short/long realized vol ratio (5/60)'),
]

results = {}
for fid, fn, desc in cands:
    try:
        panel = fc.factor_to_panel(fn, prices)
        m = fc.validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid:26s} INSUFFICIENT DATA")
            continue
        ic_w, ir_w, n_w = ic_window(panel, fwd10, pd.Timestamp('2027-11-01'), pd.Timestamp('2028-11-01'))
        ic_6, ir_6, n_6 = ic_window(panel, fwd10, pd.Timestamp('2028-05-01'), pd.Timestamp('2028-11-01'))
        gate = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"{fid:26s} warmup IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} n={m['n_dates']} | "
              f"W3_12m IC={ic_w:+.4f} IR={ir_w:+.4f} n={n_w} | W4_6m IC={ic_6:+.4f} IR={ir_6:+.4f} n={n_6} | gate={'PASS' if gate else 'fail'}")
        results[fid] = (m, panel, (ic_w, ir_w, n_w), (ic_6, ir_6, n_6))
    except Exception as e:
        print(f"{fid:26s} ERROR {e}")

print(f"\ntotal {time.time()-t0:.1f}s")
