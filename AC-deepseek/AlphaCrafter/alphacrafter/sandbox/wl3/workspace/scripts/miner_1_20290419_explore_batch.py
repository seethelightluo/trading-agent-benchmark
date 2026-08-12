"""miner_1 2029-04-19 exploration batch (lean): screen NEW cross-asset factor ideas.

Data visible through 2029-04-18 (current date 2029-04-19). Admission battery uses
the shared warm-up window 2020-01-01..2026-07-15 with h=10 daily paper IC
(|IC|>=0.007, |ICIR|>=0.084). Robustness: also report W3 (recent 12M) and W4
(recent 6M) IC/ICIR on live data visible through 2029-04-18. Correlation audit
vs the 21 currently-effective library factors is deferred to a second script and
only run for gate-passing candidates (keeps each run under the timeout).
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import factor_common as fc

t0 = time.time()
prices = fc.load_prices(days=2500)
indexes = {s: fc.load_index(s, days=2500, prices=prices) for s in fc.INDEX_SIGNALS}
last = max(d.index.max() for d in prices.values())
print(f"data loaded {time.time()-t0:.1f}s; last visible {last.date()}; assets {len(prices)}", flush=True)

fwd10 = fc.forward_returns(prices, 10)
VAL_START, VAL_END = fc.VAL_START, fc.VAL_END

def fast_ic_series(panel, fwd, wstart=None, wend=None, min_valid=8):
    """Vectorized-ish daily cross-sectional Spearman IC between factor and fwd ret."""
    common = panel.index.intersection(fwd.index)
    ic = {}
    X = panel.loc[common]
    Y = fwd.loc[common]
    Xr = X.rank(axis=1)
    Yr = Y.rank(axis=1)
    for i, d in enumerate(common):
        x = Xr.iloc[i]; y = Yr.iloc[i]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            xv = x[m].values.astype(float); yv = y[m].values.astype(float)
            xv = (xv - xv.mean()) / (xv.std(ddof=1) + 1e-12)
            yv = (yv - yv.mean()) / (yv.std(ddof=1) + 1e-12)
            ic[d] = float(np.dot(xv, yv) / (len(xv) - 1))
    s = pd.Series(ic).sort_index()
    if wstart is not None:
        s = s[(s.index >= wstart) & (s.index <= wend)]
    return s

def ic_stats(s, min_n=60):
    if len(s) < min_n:
        return float('nan'), float('nan'), len(s)
    mu, sd = s.mean(), s.std(ddof=1)
    return float(mu), float(mu / sd if sd > 0 else 0.0), len(s)

W3S, W3E = pd.Timestamp('2028-04-19'), pd.Timestamp('2029-04-18')
W4S, W4E = pd.Timestamp('2028-10-19'), pd.Timestamp('2029-04-18')

# ---------------- candidate factor definitions ----------------
def f_kurt_20(df, s, win=20):
    return df['close'].pct_change().rolling(win, min_periods=15).kurt()

def f_autocorr_1_60(df, s, win=60):
    r = df['close'].pct_change()
    return r.rolling(win, min_periods=40).corr(r.shift(1))

def f_r2_trend_60(df, s, win=60):
    t = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    c = df['close'].rolling(win, min_periods=30).corr(t)
    return (c ** 2)

def f_gold_beta_cond_60x20(df, s, prices, win=60, cond=20):
    g = prices['XAU']['close']
    r = df['close'].pct_change(); rg = g.pct_change()
    z = pd.concat([r.rename('r'), rg.rename('g')], axis=1).dropna()
    b = z['r'].rolling(win, min_periods=30).cov(z['g']) / z['g'].rolling(win, min_periods=30).var()
    trend = (g / g.shift(cond) - 1.0).reindex(z.index)
    return (b * np.sign(trend)).reindex(z.index)

def f_volume_conf_mom_20(df, s, win=20):
    mom = df['close'] / df['close'].shift(win) - 1.0
    if 'volume' not in df or df['volume'].notna().mean() < 0.5:
        return mom * 0.0 + np.nan
    v5 = df['volume'].rolling(5).mean(); v20 = df['volume'].rolling(win).mean()
    vexp = (v5 / v20.replace(0, np.nan) - 1.0)
    return (mom * vexp).replace([np.inf, -np.inf], np.nan)

def f_ulcer_60(df, s, win=60):
    dd = df['close'] / df['close'].rolling(win, min_periods=30).max() - 1.0
    return np.sqrt((dd ** 2).rolling(win, min_periods=30).mean())

def f_range_exp_20(df, s, win=20, base=60):
    hi = df['close'].rolling(win).max(); lo = df['close'].rolling(win).min()
    rng20 = (hi - lo) / df['close']
    rng = (df['high'] - df['low']) / df['close']
    return (rng20 / rng.rolling(base, min_periods=30).mean()).replace([np.inf, -np.inf], np.nan)

def f_overnight_mom_20(df, s, win=20):
    ov = df['open'] / df['close'].shift(1) - 1.0
    return ov.rolling(win).sum()

def f_crypto_corr_60(df, s, prices, win=60):
    b = prices['BTC']['close']
    r = df['close'].pct_change(); rb = b.pct_change()
    z = pd.concat([r.rename('r'), rb.rename('b')], axis=1)
    return z['r'].rolling(win, min_periods=30).corr(z['b'])

def f_yieldspread_beta_60(df, s, prices, win=60, cond=20):
    sp = prices['US10Y']['close'] - prices['CN10Y']['close']
    sc = sp.diff()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), sc.rename('s')], axis=1).dropna()
    b = z['r'].rolling(win, min_periods=30).cov(z['s']) / z['s'].rolling(win, min_periods=30).var()
    trend = (sp - sp.shift(cond)).reindex(z.index)
    return (b * np.sign(trend)).reindex(z.index)

def f_vol_asym_20(df, s, win=20):
    r = df['close'].pct_change()
    neg = r.where(r < 0, 0.0); pos = r.where(r > 0, 0.0)
    ds = neg.rolling(win).std(); us = pos.rolling(win).std()
    tot = r.rolling(win).std()
    return ((ds - us) / tot.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def f_stress_disp_beta_60(df, s, prices, win=60):
    rets = pd.DataFrame({k: v['close'].pct_change() for k, v in prices.items()})
    disp = rets.std(axis=1, skipna=True)
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), disp.rename('d')], axis=1).dropna()
    b = z['r'].rolling(win, min_periods=30).cov(z['d']) / z['d'].rolling(win, min_periods=30).var()
    return b.reindex(z.index)

def f_wti_beta_cond_60x20(df, s, prices, win=60, cond=20):
    w = prices['WTI']['close']
    r = df['close'].pct_change(); rw = w.pct_change()
    z = pd.concat([r.rename('r'), rw.rename('w')], axis=1).dropna()
    b = z['r'].rolling(win, min_periods=30).cov(z['w']) / z['w'].rolling(win, min_periods=30).var()
    trend = (w / w.shift(cond) - 1.0).reindex(z.index)
    return (b * np.sign(trend)).reindex(z.index)

cands = [
    ('kurt_20', lambda df, s: f_kurt_20(df, s), 'excess kurtosis of 20d daily returns (tail weight)'),
    ('autocorr_1_60', lambda df, s: f_autocorr_1_60(df, s), 'first-order autocorrelation of daily returns (60d)'),
    ('r2_trend_60', lambda df, s: f_r2_trend_60(df, s), 'R^2 of 60d linear trend fit (trend quality)'),
    ('gold_beta_cond_60x20', lambda df, s: f_gold_beta_cond_60x20(df, s, prices), '60d beta to XAU signed by XAU 20d trend (safe-haven sensitivity)'),
    ('volume_conf_mom_20', lambda df, s: f_volume_conf_mom_20(df, s), '20d momentum x volume expansion (volume-confirmed trend)'),
    ('ulcer_60', lambda df, s: f_ulcer_60(df, s), '60d ulcer index (mean squared drawdown depth)'),
    ('range_exp_20', lambda df, s: f_range_exp_20(df, s), '20d range vs trailing 60d avg daily range (range expansion)'),
    ('overnight_mom_20', lambda df, s: f_overnight_mom_20(df, s), 'cumulative overnight return over 20d (gap persistence)'),
    ('crypto_corr_60', lambda df, s: f_crypto_corr_60(df, s, prices), '60d correlation with BTC returns (crypto regime beta)'),
    ('yieldspread_beta_60', lambda df, s: f_yieldspread_beta_60(df, s, prices), '60d beta to US10Y-CN10Y spread change x spread 20d trend'),
    ('vol_asym_20', lambda df, s: f_vol_asym_20(df, s), 'signed (downside vol - upside vol)/total vol over 20d'),
    ('stress_disp_beta_60', lambda df, s: f_stress_disp_beta_60(df, s, prices), '60d beta of asset to cross-sectional return dispersion'),
    ('wti_beta_cond_60x20', lambda df, s: f_wti_beta_cond_60x20(df, s, prices), '60d beta to WTI signed by WTI 20d trend (energy sensitivity)'),
]

results = {}
passing = []
for fid, fn, desc in cands:
    t1 = time.time()
    try:
        panel = fc.factor_to_panel(fn, prices)
        ic_all = fast_ic_series(panel, fwd10, VAL_START, VAL_END)
        icw, irw, nw = ic_stats(ic_all)
        ic3, ir3, n3 = ic_stats(fast_ic_series(panel, fwd10, W3S, W3E))
        ic4, ir4, n4 = ic_stats(fast_ic_series(panel, fwd10, W4S, W4E))
        # coverage & turnover on warm-up window
        fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
        total_cells = fac.shape[0] * fac.shape[1]
        cov = float(fac.notna().sum().sum() / total_cells) if total_cells else 0.0
        ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
        ranked = fac.rank(axis=1)
        turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
        gate = (abs(icw) >= 0.007) and (abs(irw) >= 0.084)
        print(f"{fid:26s} warmup IC={icw:+.4f} ICIR={irw:+.4f} n={nw} cov={cov:.3f} ge8={ge8:.3f} turn={turn:.2f} | "
              f"W3_12m IC={ic3:+.4f} IR={ir3:+.4f} n={n3} | W4_6m IC={ic4:+.4f} IR={ir4:+.4f} n={n4} | "
              f"gate={'PASS' if gate else 'fail'} [{time.time()-t1:.1f}s]", flush=True)
        results[fid] = {'desc': desc,
                        'warmup': {'ic': icw, 'icir': irw, 'n_ic_dates': nw, 'coverage': cov,
                                   'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn},
                        'w3_12m': {'ic': ic3, 'icir': ir3, 'n': n3},
                        'w4_6m': {'ic': ic4, 'icir': ir4, 'n': n4},
                        'gate_pass': bool(gate)}
        if gate:
            passing.append(fid)
    except Exception as e:
        print(f"{fid:26s} ERROR {e}", flush=True)
        results[fid] = {'error': str(e)}

print(f"\nGATE-PASSING candidates: {passing}", flush=True)

out = {'generated': '2029-04-19', 'last_visible': str(last.date()), 'n_assets': len(prices),
       'candidates': results, 'passing': passing}
with open('scripts/miner_1_20290419_explore_batch_out.json', 'w') as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"total {time.time()-t0:.1f}s", flush=True)
