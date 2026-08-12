"""miner_1 2029-04-05 exploration batch: screen NEW cross-asset factor ideas.

Admission battery from factor_common (warm-up 2020-01-01..2026-07-15, h=10 daily
paper IC). Robustness: also compute W3 (recent 12M) and W4 (recent 6M) IC/ICIR
using live data visible through 2029-04-04. For gate-passing candidates, report
max_abs_library_correlation vs the 20 currently-effective library factors.
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import factor_common as fc

t0 = time.time()
prices = fc.load_prices(days=4200)
indexes = {s: fc.load_index(s, days=4200, prices=prices) for s in fc.INDEX_SIGNALS}
last = max(d.index.max() for d in prices.values())
print(f"data loaded {time.time()-t0:.1f}s; last visible {last.date()}; assets {len(prices)}", flush=True)

fwd10 = fc.forward_returns(prices, 10)

def ic_window(panel, wstart, wend, min_valid=8):
    common = panel.index.intersection(fwd10.index)
    ic = {}
    for d in common:
        if d < wstart or d > wend:
            continue
        x = panel.loc[d]; y = fwd10.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            ic[d] = x[m].rank().corr(y[m].rank())
    s = pd.Series(ic)
    if len(s) < 60:
        return np.nan, np.nan, len(s)
    mu, sd = s.mean(), s.std(ddof=1)
    return float(mu), float(mu / sd if sd > 0 else 0.0), len(s)

W3S, W3E = pd.Timestamp('2028-04-05'), pd.Timestamp('2029-04-04')
W4S, W4E = pd.Timestamp('2028-10-05'), pd.Timestamp('2029-04-04')

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
for fid, fn, desc in cands:
    try:
        panel = fc.factor_to_panel(fn, prices)
        m = fc.validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid:26s} INSUFFICIENT DATA (warm-up)")
            results[fid] = {'error': 'insufficient_data'}
            continue
        ic3, ir3, n3 = ic_window(panel, W3S, W3E)
        ic4, ir4, n4 = ic_window(panel, W4S, W4E)
        gate = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"{fid:26s} warmup IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} n={m['n_dates']} cov={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f} | "
              f"W3_12m IC={ic3:+.4f} IR={ir3:+.4f} n={n3} | W4_6m IC={ic4:+.4f} IR={ir4:+.4f} n={n4} | gate={'PASS' if gate else 'fail'}", flush=True)
        results[fid] = {'desc': desc, 'warmup': {k: m[k] for k in ('ic', 'icir', 'ic_hit_ratio', 'n_ic_dates', 'coverage_dates_ge8', 'turnover_10d_rank')},
                        'w3': {'ic': ic3, 'icir': ir3, 'n': n3}, 'w4': {'ic': ic4, 'icir': ir4, 'n': n4},
                        'gate_pass': gate, 'panel': panel}
    except Exception as e:
        print(f"{fid:26s} ERROR {e}", flush=True)
        results[fid] = {'error': str(e)}

# correlation audit vs full 20-factor library for gate-passing candidates
passing = [fid for fid, r in results.items() if r.get('gate_pass')]
if passing:
    print("\nBuilding 20-factor library panels for correlation audit ...", flush=True)
    dxy = indexes.get('DXY'); vix = indexes.get('VIX'); eurusd = indexes.get('EURUSD')
    def rb(df, mkt, win=60, minp=30, diff=False):
        r = df['close'].pct_change()
        m = mkt.diff() if diff else mkt.pct_change()
        z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
        return (z['r'].rolling(win, min_periods=minp).cov(z['m']) / z['m'].rolling(win, min_periods=minp).var()).reindex(z.index)
    def f_down_beta(df, s):
        spx = prices['SPX']['close']; r = df['close'].pct_change(); rs = spx.pct_change()
        z = pd.concat([r.rename('r'), rs.rename('m')], axis=1).dropna()
        down = z[z['m'] < 0]
        return (down['r'].rolling(60, min_periods=30).cov(down['m']) / down['m'].rolling(60, min_periods=30).var()).reindex(z.index)
    def f_cn10y_beta(df, s): return rb(df, prices['CN10Y']['close'], diff=True)
    def f_spx_beta(df, s): return rb(df, prices['SPX']['close'])
    def f_hs300_beta(df, s): return rb(df, prices['000300.SH']['close'])
    def f_comm_beta(df, s):
        bk = pd.concat([prices['XAU']['close'].pct_change(), prices['COPPER']['close'].pct_change(),
                        prices['WTI']['close'].pct_change()], axis=1).mean(axis=1)
        return rb(df, bk)
    def cond_fn(df, s, mkt):
        b = rb(df, mkt)
        tr = mkt / mkt.shift(20) - 1.0
        return (b * tr).reindex(pd.concat([df['close'].pct_change(), mkt.pct_change()], axis=1).dropna().index)
    def f_dxy(df, s): return cond_fn(df, s, dxy['close']) if dxy is not None else None
    def f_vix(df, s): return cond_fn(df, s, vix['close']) if vix is not None else None
    def f_eur(df, s): return cond_fn(df, s, eurusd['close']) if eurusd is not None else None
    def f_vam(df, s):
        mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
        vol = df['close'].pct_change().rolling(60).std()
        return mom / vol
    def f_hvr(df, s):
        hi = df['close'].rolling(20).max(); lo = df['close'].rolling(20).min()
        return ((hi - lo) / df['close']) / df['close'].pct_change().rolling(20).std()
    def f_isk(df, s): return (df['close'] / df['open'] - 1.0).rolling(20).skew()
    def f_vov(df, s): return df['close'].pct_change().rolling(20).std().rolling(60).std()
    def f_cg(df, s):
        cg = prices['COPPER']['close'].pct_change() - prices['XAU']['close'].pct_change()
        return rb(df, cg, win=20, minp=12)
    def f_dd(df, s):
        hi = df['close'].rolling(120).max()
        ds = (df.index - pd.Series(df.index, index=df.index).where(df['close'] == hi).ffill()).dt.days
        dur = np.log1p(ds.fillna(0))
        mom = df['close'].shift(5) / df['close'].shift(125) - 1.0
        zm = (mom - mom.rolling(250).mean()) / mom.rolling(250).std()
        return dur - rb(df, prices['SPX']['close']) * zm
    def f_hp(df, s):
        hi = df['close'].rolling(60).max(); lo = df['close'].rolling(60).min()
        return (df['close'] - lo) / (hi - lo)
    def f_ma(df, s):
        return df['close'].shift(5) / df['close'].shift(65) - 1.0 - (df['close'].shift(5) / df['close'].shift(125) - 1.0)
    def f_rs(df, s): return ((df['high'] - df['low']) / df['close']).rolling(20).skew()
    def f_sp(df, s):
        r = df['close'].pct_change()
        return ((np.sign(r) == np.sign(r.shift(1))).astype(float)).rolling(20).mean()
    def f_st(df, s):
        r = df['close'].pct_change()
        up = (r > 0).astype(int); dn = (r < 0).astype(int)
        su = up.rolling(60, min_periods=1).apply(lambda x: (x == 1).sum() if (x == 1).all() else 0, raw=True)
        # streak via simple cumulative reset
        def streak(x):
            out = np.zeros(len(x)); c = 0
            for i in range(len(x)):
                c = c + 1 if x.iloc[i] else 0
                out[i] = c
            return pd.Series(out, index=x.index)
        return ((streak(up) - streak(dn)).rolling(60).max() / 60.0)
    def f_vrs(df, s):
        rv = df['close'].pct_change().rolling(20).std()
        above = (rv > rv.rolling(60).median()).astype(float)
        return above.diff().abs().rolling(60).mean()
    lib_fns = {
        'down_beta_60': f_down_beta, 'cn10y_beta_60': f_cn10y_beta, 'spx_beta_60': f_spx_beta,
        'hs300_beta_60': f_hs300_beta, 'comm_basket_beta_60': f_comm_beta, 'dxy_beta_cond_60x20': f_dxy,
        'vix_beta_cond_60x20': f_vix, 'eurusd_beta_cond_60x20': f_eur, 'vol_adj_mom_20_60': f_vam,
        'hilo_vol_ratio_20': f_hvr, 'intraday_ret_skew_20': f_isk, 'vol_of_vol20x60': f_vov,
        'copper_gold_beta_20': f_cg, 'dd_duration_120_resid': f_dd, 'hilo_pos_60': f_hp,
        'mom_accel_60_120': f_ma, 'range_skew_20': f_rs, 'sign_persist_20': f_sp,
        'streak_60': f_st, 'vol_regime_switch_20x60': f_vrs,
    }
    lib_panels = {}
    for fid, fn in lib_fns.items():
        try:
            p = fc.factor_to_panel(fn, prices)
            if len(p) > 0:
                lib_panels[fid] = p
        except Exception as e:
            print(f"lib {fid} error {e}")
    print(f"library panels built: {len(lib_panels)}")
    for fid in passing:
        panel = results[fid]['panel']
        rho, bid = fc.max_library_correlation(panel, lib_panels)
        results[fid]['max_abs_library_correlation'] = rho
        results[fid]['max_corr_library_id'] = bid
        print(f"{fid:26s} max_abs_library_correlation={rho:.4f} vs {bid}", flush=True)

out = {fid: {k: v for k, v in r.items() if k != 'panel'} for fid, r in results.items()}
with open('scripts/miner_1_20290405_explore_batch_out.json', 'w') as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\ntotal {time.time()-t0:.1f}s")
