"""miner2 2028-10-13: screen trend-quality factor candidates on the 15-name cross-asset panel.
Visible data through 2028-10-12 only. Daily rank IC vs 1-day forward return, ICIR, hit ratio,
coverage, year splits, and correlation vs existing library factors.
"""
import pandas as pd, numpy as np, json

VISIBLE = '2028-10-12'
START = '2021-01-01'

p = pd.read_pickle('scripts/panel_cache.pkl')
close = p['close'].loc[:VISIBLE]
ret = close.pct_change()
n = close.shape[0]
print(f"panel dates: {close.index.min().date()} -> {close.index.max().date()} rows={n} assets={close.shape[1]}")

def er(px, w):
    """Kaufman efficiency ratio: |P_t - P_{t-w}| / sum(|dP|) over w days."""
    d = px.diff().abs()
    path = d.rolling(w).sum()
    net = (px - px.shift(w)).abs()
    return net / path

def ic_series(fv, fwd_ret, min_cov=8):
    """Daily cross-sectional Spearman IC between factor values and forward returns."""
    ic, dates = [], []
    for dt, row in fv.iterrows():
        fr = fwd_ret.loc[dt]
        m = row.notna() & fr.notna()
        if int(m.sum()) < min_cov:
            continue
        ic.append(row[m].rank().corr(fr[m].rank()))
        dates.append(dt)
    return pd.Series(ic, index=dates)

cands = {}

# C1: Kaufman efficiency ratio, direction-signed, 60d
er60 = er(close, 60)
cands['trend_eff_60d'] = er60 * np.sign(close - close.shift(60))
# C2: Kaufman efficiency ratio, direction-signed, 20d
er20 = er(close, 20)
cands['trend_eff_20d'] = er20 * np.sign(close - close.shift(20))
# C3: drawdown distance from 60d high (negative => below high)
cands['dd_dist_60d'] = close / close.rolling(60).max() - 1.0
# C4: drawdown distance from 120d high
cands['dd_dist_120d'] = close / close.rolling(120).max() - 1.0
# C5: MA alignment (continuous): sum of (close/MA_k - 1) for k in 20,60,120
cands['ma_align'] = (close / close.rolling(20).mean() - 1) + (close / close.rolling(60).mean() - 1) + (close / close.rolling(120).mean() - 1)
# C6: vol-scaled 120d momentum
mom120 = close.shift(5) / close.shift(125) - 1.0
vol60 = ret.rolling(60).std()
cands['mom_120d_voladj'] = mom120 / vol60
# C7: extension above 120d MA (overbought/extension)
cands['ext_120d'] = close / close.rolling(120).mean() - 1.0
# C8: down-capture/up-capture ratio proxy: 20d downside vol / upside vol (asymmetry)
up = ret.clip(lower=0).rolling(20).mean()
dn = (-ret.clip(upper=0)).rolling(20).mean()
cands['asym_20d'] = dn / (up + dn) - 0.5  # >0 means downside dominance

# existing library signals for correlation
lib = {}
lib['nclv_1d'] = -(close - close.rolling(1).min()) / (close.rolling(1).max() - close.rolling(1).min())
lib['rev_2d'] = -(np.log(close) - np.log(close.shift(2)))
lib['mom_120d'] = mom120
lib['vol_of_vol'] = ret.rolling(20).std().rolling(60).std()
vix = p['macro']['VIX'].loc[:VISIBLE]
lib['vix_beta_cond'] = -ret.rolling(60).cov(vix.pct_change()) / vix.pct_change().rolling(60).var() * (vix / vix.shift(20) - 1.0)

fwd1 = ret.shift(-1)

results = []
for name, fv in cands.items():
    ic = ic_series(fv, fwd1)
    ic = ic.dropna()
    ic_mean = ic.mean()
    icir = ic_mean / ic.std(ddof=1) if ic.std(ddof=1) > 0 else 0.0
    hit = (np.sign(ic) == np.sign(ic_mean)).mean()
    cov = fv.notna().mean().mean()
    # year splits
    yrs = {}
    for y, g in ic.groupby(ic.index.year):
        yrs[str(y)] = dict(ic=round(float(g.mean()), 4), icir=round(float(g.mean() / g.std(ddof=1)), 3) if g.std(ddof=1) > 0 else 0.0, n=int(g.shape[0]))
    # max abs correlation with library
    corrs = {}
    for lname, lv in lib.items():
        both = pd.concat([fv.stack(), lv.stack()], axis=1, keys=['f', 'l']).dropna()
        if len(both) > 100:
            corrs[lname] = float(both['f'].corr(both['l']))
    maxcorr = max([abs(v) for v in corrs.values()], default=np.nan)
    results.append(dict(name=name, ic=ic_mean, icir=icir, hit=hit, n_dates=int(ic.shape[0]),
                        coverage=round(float(cov), 3), max_lib_corr=maxcorr, corrs=corrs, years=yrs))
    print(f"{name:22s} IC={ic_mean:+.4f} ICIR={icir:+.3f} hit={hit:.3f} n={ic.shape[0]:5d} cov={cov:.3f} maxLibCorr={maxcorr:.3f}")

print("\n--- ranking by |IC| (gate: |IC|>=0.007, |ICIR|>=0.084) ---")
for r in sorted(results, key=lambda x: -abs(x['ic'])):
    ok = abs(r['ic']) >= 0.007 and abs(r['icir']) >= 0.084
    print(f"{r['name']:22s} IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} {'PASS' if ok else 'fail'}")
    print("   years:", {k: f"{v['ic']:+.3f}/{v['icir']:+.2f}({v['n']})" for k, v in r['years'].items()})
    print("   corr:", {k: round(v, 2) for k, v in r['corrs'].items()})
