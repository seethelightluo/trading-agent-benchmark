"""Screener cycle 16: market regime & factor ensemble check (data through 2026-07-15)."""
import json, os, glob
import numpy as np
import pandas as pd

DATA = '../persistent/stock_data'
IDX = '../persistent/index_data'
WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

def load(sym, folder=DATA):
    df = pd.read_csv(os.path.join(folder, sym + '.csv'), parse_dates=['date'])
    df = df.sort_values('date').set_index('date')
    return df

# Align all on common anchor dates
anchor = load('000300.SH').index
rets = {}
closes = {}
for s in WATCH:
    df = load(s)
    df = df[~df.index.duplicated(keep='last')]
    closes[s] = df['close'].reindex(anchor)
    rets[s] = closes[s].pct_change()

px = pd.DataFrame(closes)
ret = pd.DataFrame(rets)
px = px.dropna(how='all').dropna(axis=1, how='all')
ret = ret.reindex(px.index)

cutoff = '2026-07-15'
px_c = px.loc[:cutoff]
ret_c = ret.loc[:cutoff]

print(f"Data range: {px_c.index[0].date()} .. {px_c.index[-1].date()}  rows={len(px_c)}")

# 1) Trend: price vs 200d SMA, 60d/20d returns
sma200 = px_c.rolling(200).mean()
above = (px_c.iloc[-1] > sma200.iloc[-1])
r60 = px_c.iloc[-1] / px_c.iloc[-61] - 1
r20 = px_c.iloc[-1] / px_c.iloc[-21] - 1
r120 = px_c.iloc[-1] / px_c.iloc[-121] - 1
print("\n=== Trend (through %s) ===" % cutoff)
print(f"{'asset':<10}{'above200':>9}{'r20':>9}{'r60':>9}{'r120':>9}")
for s in WATCH:
    print(f"{s:<10}{str(above[s]):>9}{r20[s]*100:>8.1f}%{r60[s]*100:>8.1f}%{r120[s]*100:>8.1f}%")
print(f"\n#above 200d SMA: {int(above.sum())}/15")
print(f"EW portfolio r20: {r20.mean()*100:.2f}%  r60: {r60.mean()*100:.2f}%  r120: {r120.mean()*100:.2f}%")

# 2) Volatility regime
vol20 = ret_c.rolling(20).std() * np.sqrt(252)
vol250 = ret_c.rolling(250).std() * np.sqrt(252)
ratio = vol20.iloc[-1] / vol250.median()
print("\n=== Volatility (20d ann. vs 250d median) ===")
for s in WATCH:
    print(f"{s:<10} vol20={vol20.iloc[-1][s]*100:6.1f}%  ratio={ratio[s]:.2f}x")
print(f"mean ratio: {ratio.mean():.2f}x")

# 3) Cross-sectional correlation & dispersion (60d)
r60d = ret_c.tail(60)
corr = r60d.corr()
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
avg_corr = corr.values[mask].mean()
disp = r60d.std(axis=1).mean() * 100
print(f"\nAvg pairwise 60d corr: {avg_corr:.3f} | mean daily cross-sectional dispersion: {disp:.2f}%")

# 4) Macro obs signals
vix = load('VIX', IDX)['close'].reindex(anchor).dropna()
vix_c = vix.loc[:cutoff]
vix_med250 = vix_c.rolling(250).median().iloc[-1]
print(f"\nVIX: {vix_c.iloc[-1]:.2f} (250d median {vix_med250:.2f}, 20d chg {(vix_c.iloc[-1]/vix_c.iloc[-21]-1)*100:.1f}%)")
dxy = load('DXY', IDX)['close'].reindex(anchor).dropna()
dxy_c = dxy.loc[:cutoff]
print(f"DXY: {dxy_c.iloc[-1]:.2f} (20d chg {(dxy_c.iloc[-1]/dxy_c.iloc[-21]-1)*100:.1f}%)")

# 5) Recent factor IC recomputation on 10d forward rank IC (full + recent 250d)
def rank_ic(factor_series, fwd_ret, dates):
    f = factor_series.reindex(dates)
    fr = fwd_ret.reindex(dates)
    ics = []
    for i in range(len(dates)):
        x = f.iloc[i].dropna()
        y = fr.iloc[i].dropna()
        common = x.index.intersection(y.index)
        if len(common) < 8:
            continue
        ics.append(np.corrcoef(x[common].rank(), y[common].rank())[0, 1])
    return np.array(ics)

fwd10 = ret_c.shift(-10)
dates = ret_c.index[:-11]
ics = {}
for name, expr in [
    ('mom_120d_skip5', lambda c: c.shift(5) / c.shift(125) - 1),
    ('mom_10d_skip5', lambda c: c.shift(5) / c.shift(15) - 1),
]:
    f = pd.DataFrame({s: expr(px_c[s]) for s in WATCH})
    a = rank_ic(f, fwd10, dates)
    a_recent = rank_ic(f, fwd10, dates[-250:])
    ics[name] = (a.mean(), a.mean()/a.std()*np.sqrt(12), a_recent.mean(), a_recent.mean()/a_recent.std()*np.sqrt(12))
    print(f"{name}: full IC={a.mean():+.4f} ICIR={a.mean()/a.std()*np.sqrt(12):+.3f} | recent250 IC={a_recent.mean():+.4f} ICIR={a_recent.mean()/a_recent.std()*np.sqrt(12):+.3f}")

# vol-of-vol
f = pd.DataFrame({s: px_c[s].pct_change().rolling(20).std().rolling(60).std() for s in WATCH})
a = rank_ic(f, fwd10, dates); a_recent = rank_ic(f, fwd10, dates[-250:])
print(f"vol_of_vol20x60: full IC={a.mean():+.4f} ICIR={a.mean()/a.std()*np.sqrt(12):+.3f} | recent250 IC={a_recent.mean():+.4f} ICIR={a_recent.mean()/a_recent.std()*np.sqrt(12):+.3f}")

# vix beta conditional (ad-hoc anchor-calendar; known caveat)
vix_ret = vix.pct_change()
f = pd.DataFrame(index=px_c.index, columns=WATCH, dtype=float)
for s in WATCH:
    ar = ret_c[s]
    dfv = pd.concat([ar, vix_ret], axis=1, keys=['a', 'v']).dropna()
    beta = dfv['a'].rolling(60).cov(dfv['v']) / dfv['v'].rolling(60).var()
    vix_chg = vix_ret.reindex(dfv.index) * 20
    f[s] = (-beta * vix_chg).reindex(px_c.index)
a = rank_ic(f, fwd10, dates); a_recent = rank_ic(f, fwd10, dates[-250:])
print(f"vix_beta_cond_60x20 (ad-hoc): full IC={a.mean():+.4f} ICIR={a.mean()/a.std()*np.sqrt(12):+.3f} | recent250 IC={a_recent.mean():+.4f} ICIR={a_recent.mean()/a_recent.std()*np.sqrt(12):+.3f}")

# factor cross-correlations (latest cross-section)
print("\n=== Factor cross-sectional correlation (latest) ===")
fl = {}
fl['mom_120d'] = px_c['SOX'].shift(5)/px_c['SOX'].shift(125)-1
# use mean over assets for a quick pairwise on latest cross-section instead:
fsec = pd.DataFrame({
    'mom_120d': {s: (px_c[s].iloc[-6]/px_c[s].iloc[-126]-1) for s in WATCH},
    'mom_10d': {s: (px_c[s].iloc[-6]/px_c[s].iloc[-16]-1) for s in WATCH},
    'vol_of_vol': {s: px_c[s].pct_change().rolling(20).std().rolling(60).std().iloc[-1] for s in WATCH},
})
print(fsec.corr().round(3).to_string())
