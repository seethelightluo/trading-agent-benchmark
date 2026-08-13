"""miner3 2032-06-18: re-validate 5 ensemble factors on fresh panel (through 2032-06-17).
Vectorized cross-sectional rank IC."""
import pandas as pd, numpy as np, pickle, time

panel = pd.read_pickle('scripts/panel_cache_20320618.pkl')
close = panel['close']; high = panel['high']; low = panel['low']
ret = panel['ret']; macro = panel['macro']
vix = macro['VIX']

# ---------- factor computations ----------
rng = high.rolling(1).max() - low.rolling(1).min()
nclv1 = -((close - low.rolling(1).min()) / rng.replace(0, np.nan))
rev2 = -(np.log(close) - np.log(close.shift(2)))
rv20 = ret.rolling(20).std()
vov = rv20.rolling(60).std()
vix_ret = vix.pct_change()
beta60 = ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
vix20 = vix / vix.shift(20) - 1.0
vixbeta = -beta60 * vix20
mom120 = close.shift(5) / close.shift(125) - 1.0

factors = {
    'miner2_20260715_nclv_1d': nclv1,
    'miner2_20260715_rev_2d': rev2,
    'vol_of_vol20x60': vov,
    'vix_beta_cond_60x20': vixbeta,
    'mom_120d_skip5': mom120,
}

fwd = {h: (close.shift(-h) / close - 1.0) for h in [1, 2, 3, 5, 10, 20]}

def rank_ic_fast(fdf, rdf, min_assets=8):
    """Rank IC per date, vectorized with numpy."""
    F = fdf.values.astype(float)
    R = rdf.values.astype(float)
    dates = fdf.index
    ics = np.full(len(dates), np.nan)
    for i in range(len(dates)):
        f = F[i]; r = R[i]
        m = ~(np.isnan(f) | np.isnan(r))
        n = m.sum()
        if n < min_assets:
            continue
        # rank of valid entries
        fv = f[m]; rv = r[m]
        fr = np.empty(n); rr = np.empty(n)
        fr[np.argsort(fv)] = np.arange(1, n + 1)
        rr[np.argsort(rv)] = np.arange(1, n + 1)
        fm = fr - fr.mean(); rm = rr - rr.mean()
        denom = np.sqrt((fm * fm).sum() * (rm * rm).sum())
        ics[i] = (fm * rm).sum() / denom if denom > 0 else np.nan
    return pd.Series(ics, index=dates).dropna()

def summarize(ic_series):
    ic = ic_series.mean()
    icir = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else np.nan
    hit = (np.sign(ic_series) == np.sign(ic)).mean() if not np.isnan(ic) else np.nan
    return ic, icir, hit

full_start = close.index.min()
recent_start = close.index[-1] - pd.Timedelta(days=730)
recent6m_start = close.index[-1] - pd.Timedelta(days=183)

print("=" * 104)
print("RE-VALIDATION of ensemble factors | panel through", close.index.max().date())
print("Gates: |IC1| >= 0.0070 and |ICIR1| >= 0.0840")
print("=" * 104)

rows = []
t0 = time.time()
for name, f in factors.items():
    print("\n###", name)
    for label, start in [('FULL', full_start), ('2Y', recent_start), ('6M', recent6m_start)]:
        sub = f[f.index >= start]
        ic1 = rank_ic_fast(sub, fwd[1])
        ic, icir, hit = summarize(ic1)
        cov_ge8 = round((sub.notna().sum(axis=1) >= 8).mean(), 3)
        mean_assets = round(sub.notna().sum(axis=1).mean(), 2)
        rows.append({'window': label, 'factor': name, 'ic': ic, 'icir': icir, 'hit': hit,
                     'n_dates': len(ic1), 'cov_ge8': cov_ge8, 'mean_assets': mean_assets})
        print(f"  {label:5s} IC1={ic:.4f} ICIR1={icir:.3f} hit={hit:.3f} n={len(ic1)} cov_ge8={cov_ge8} mean_assets={mean_assets}")

print("\nDecay profile (IC by horizon, full sample):")
for name, f in factors.items():
    dec = {}
    for h in [1, 2, 3, 5, 10, 20]:
        ics = rank_ic_fast(f, fwd[h])
        dec[h] = round(ics.mean(), 4)
    print(f"  {name}: {dec}")

print("\nSUMMARY:")
for r in rows:
    print(f"  {r['window']:5s} {r['factor']:28s} IC={r['ic']:.4f} ICIR={r['icir']:.3f} hit={r['hit']:.3f} n={r['n_dates']}")
print(f"\nelapsed {time.time()-t0:.1f}s")
