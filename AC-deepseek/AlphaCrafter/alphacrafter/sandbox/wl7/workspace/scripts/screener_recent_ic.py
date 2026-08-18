"""Compute recent cross-sectional IC for active factors (live names only, thru 2028-08-28)."""
import csv, math, json
import numpy as np
import pandas as pd

VIS = '2028-08-28'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
FROZEN = {'000688.SH','SOX','NDX','CN10Y'}
LIVE = [a for a in ASSETS if a not in FROZEN]

def load(fp):
    rows = list(csv.reader(open(fp)))
    hdr = rows[0]; idx = {c: i for i, c in enumerate(hdr)}
    out = {}
    for r in rows[1:]:
        d = r[idx['date']]
        if d > VIS: continue
        try: c = float(r[idx['close']])
        except: continue
        out[d] = c
    return pd.Series(out).sort_index()

prices = {a: load(f'../persistent/stock_data/{a}.csv') for a in ASSETS}
dxy = load('../persistent/index_data/DXY.csv')
eur = load('../persistent/index_data/EURUSD.csv')

# Build aligned DataFrame of closes
df = pd.DataFrame(prices)
df['DXY'] = dxy
df['EURUSD'] = eur
df = df.dropna(how='all').ffill().dropna()
# restrict to dates with at least 8 live prices
live_count = df[LIVE].notna().sum(axis=1)
df = df[live_count >= 8]
rets = df.pct_change()

def cs_demean(s):
    m = s.mean()
    return s - m

def factor_signals():
    out = {}
    closes = df
    r = rets
    # rel_mom_20d_skip5: 20d momentum skip 5 -> pct_change(20) then demean; skip5 approx via shift(5)
    mom20 = closes[ASSETS].pct_change(20)
    out['rel_mom_20d_skip5'] = mom20.sub(mom20.median(axis=1), axis=0)
    # beta_ew_60d: rolling 60d beta vs EW market (live market)
    mkt = r[LIVE].mean(axis=1)
    beta = {}
    for a in ASSETS:
        cov = r[a].rolling(60).cov(mkt)
        var = mkt.rolling(60).var()
        beta[a] = cov / var
    out['beta_ew_60d'] = pd.DataFrame(beta)
    # downside_vol_ratio_20 flipped: -(downside semi-vol/total vol)
    neg = r[ASSETS].clip(upper=0)
    semi = (neg**2).rolling(20).mean().apply(np.sqrt)
    tot = (r[ASSETS]**2).rolling(20).mean().apply(np.sqrt)
    out['downside_vol_ratio_20'] = -(semi / tot)
    # dxy_beta_cond_60x20: -beta(asset, DXY,60)*(DXY 20d ret)
    dxy_r = r['DXY']
    b = {}
    for a in ASSETS:
        cov = r[a].rolling(60).cov(dxy_r)
        var = dxy_r.rolling(60).var()
        b[a] = cov / var
    bdf = pd.DataFrame(b)
    dxy20 = dxy.pct_change(20)
    out['dxy_beta_cond_60x20'] = -(bdf.mul(dxy20, axis=0))
    # eurusd_beta_cond_60x20: beta(asset, EURUSD,60)*(EURUSD 20d ret)
    eur_r = r['EURUSD']
    b2 = {}
    for a in ASSETS:
        cov = r[a].rolling(60).cov(eur_r)
        var = eur_r.rolling(60).var()
        b2[a] = cov / var
    b2df = pd.DataFrame(b2)
    eur20 = eur.pct_change(20)
    out['eurusd_beta_cond_60x20'] = b2df.mul(eur20, axis=0)
    # corr_ew_60: mean pairwise 60d corr with other assets (live cross-section)
    corr = {}
    for a in ASSETS:
        others = [x for x in LIVE if x != a]
        cc = r[a].rolling(60).corr(r[others]).mean(axis=1)
        corr[a] = cc
    out['corr_ew_60'] = pd.DataFrame(corr)
    # max_ret_20d
    out['max_ret_20d'] = r[ASSETS].rolling(20).max()
    # kurt_20d_skip5
    kurt = r[ASSETS].rolling(20).kurt()
    out['kurt_20d_skip5'] = kurt
    return out

sig = factor_signals()
fwd = closes[ASSETS].pct_change(10).shift(-10)

def spearman_ic(f, fwd_, mask):
    fv = f[mask]; rv = fwd_[mask]
    valid = fv.notna() & rv.notna()
    if valid.sum() < 5: return np.nan
    return fv[valid].corr(rv[valid], method='spearman')

print(f"{'factor':26s} {'IC60':>8s} {'IC120':>8s} {'IC250':>8s} {'IC60_hit':>8s} {'dir':>4s}")
for name, f in sig.items():
    row = {}
    for n in (60, 120, 250):
        mask = f.index[-n:]
        ics = []
        for dt in mask:
            ic = spearman_ic(f.loc[dt], fwd.loc[dt], LIVE)
            if not math.isnan(ic): ics.append(ic)
        row[n] = np.mean(ics) if ics else np.nan
        if n == 60: hit60 = np.mean([1 if x > 0 else 0 for x in ics]) if ics else np.nan
    print(f"{name:26s} {row[60]*100:8.3f} {row[120]*100:8.3f} {row[250]*100:8.3f} {hit60*100:8.1f}")

# also: factor correlations (last 120d, on live names, cross-sectional time series)
print("\n=== factor signal corr (last 120d, daily IC series) ===")
ics_all = {}
for name, f in sig.items():
    mask = f.index[-120:]
    ics = []
    for dt in mask:
        ic = spearman_ic(f.loc[dt], fwd.loc[dt], LIVE)
        ics.append(ic)
    ics_all[name] = pd.Series(ics, index=mask)
icdf = pd.DataFrame(ics_all).dropna()
print(icdf.corr().round(2).to_string())

# top/bottom live names by rel_mom (latest)
print("\n=== rel_mom latest cross-section (live) ===")
last_row = sig['rel_mom_20d_skip5'].iloc[-1].sort_values(ascending=False)
print(last_row.round(4).to_string())
