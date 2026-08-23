"""Trader: replicate Screener ensemble factor values at VISIBLE date 2029-07-25.
CSVs extend to 2035; truncate to the current sim horizon."""
import pandas as pd, numpy as np, glob, os, json

VISIBLE = '2029-07-25'

files = sorted(glob.glob('../persistent/stock_data/*.csv'))
P = {}
for f in files:
    s = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    P[s] = df['close']
order = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
panel = pd.DataFrame(P).dropna(how='all')[order]
panel = panel.loc[:VISIBLE]
R = panel.pct_change().replace([np.inf, -np.inf], np.nan)

def kaufman(px, n=20):
    sig = px.diff(n).abs()
    noise = px.diff().abs().rolling(n).sum()
    return sig / noise

def rolling_beta(y, x, n):
    out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    for c in y.columns:
        yy = pd.concat([y[c], x], axis=1).dropna()
        b = yy.iloc[:, 0].rolling(n).cov(yy.iloc[:, 1]) / yy.iloc[:, 1].rolling(n).var()
        out[c] = b
    return out

F = {}
F['mom_120d_skip5'] = panel / panel.shift(126) - 1
F['mom_10d_skip5'] = panel / panel.shift(15) - 1
F['vol_z_20d'] = R.rolling(20).std() * np.sqrt(252)
F['bb_width_20d'] = F['vol_z_20d']
F['ac1_120d'] = R.rolling(120).apply(lambda x: x.autocorr() if len(x) == 120 and x.std() > 0 else np.nan, raw=False)
F['skew_20d'] = R.rolling(20).skew()
F['kaufman_eff_20d'] = kaufman(panel, 20)

vix = pd.read_csv('../persistent/index_data/VIX.csv')
vix.columns = [c.strip().lower() for c in vix.columns]
vix['date'] = pd.to_datetime(vix['date'])
vix = vix.sort_values('date').set_index('date')['close'].loc[:VISIBLE]
rvix = vix.pct_change()
F['beta_VIX_60'] = rolling_beta(R, rvix, 60)
cny = pd.read_csv('../persistent/index_data/USDCNY.csv')
cny.columns = [c.strip().lower() for c in cny.columns]
cny['date'] = pd.to_datetime(cny['date'])
cny = cny.sort_values('date').set_index('date').iloc[:, 0].loc[:VISIBLE]
rcny = cny.pct_change()
F['cny_beta_60'] = rolling_beta(R, rcny, 60)
dxy = pd.read_csv('../persistent/index_data/DXY.csv')
dxy.columns = [c.strip().lower() for c in dxy.columns]
dxy['date'] = pd.to_datetime(dxy['date'])
dxy = dxy.sort_values('date').set_index('date').iloc[:, 0].loc[:VISIBLE]
rdxy = dxy.pct_change()
c20 = R.rolling(20).corr(rdxy); c60 = R.rolling(60).corr(rdxy)
F['dxy_corr_change_20_60'] = c20 - c60

ens = json.load(open('factor_ensemble.json'))
sel = {x['factor_id']: (x['weight'], x['direction']) for x in ens['selected_factors']}
print('=== visible date', VISIBLE, '===')
print('=== 20d trailing return ===')
print((panel.iloc[-1] / panel.iloc[-21] - 1).sort_values(ascending=False).round(4).to_string())
print('\n=== 60d trailing return ===')
print((panel.iloc[-1] / panel.iloc[-61] - 1).sort_values(ascending=False).round(3).to_string())
print('\n=== VIX closes last 5 ===', vix.tail(5).round(2).tolist())
print('\n=== composite score (signed) ===')
score = pd.Series({a: 0.0 for a in order})
for name, (w, d) in sel.items():
    fv = F[name].iloc[-1].astype(float)
    z = (fv - fv.mean()) / fv.std(ddof=0)
    score = score.add(w * z * d, fill_value=np.nan)
print(score.sort_values(ascending=False).round(3).to_string())
print('\n=== per-factor z at visible date ===')
zz = pd.DataFrame({k: (F[k].iloc[-1].astype(float) - F[k].iloc[-1].astype(float).mean()) / F[k].iloc[-1].astype(float).std(ddof=0) for k in sel})
print(zz.round(2).to_string())
print('\n=== composite 10d fwd IC (trailing 60 visible dates) ===')
fwd10 = panel.shift(-10) / panel - 1
composite = pd.DataFrame({a: sum(w * (F[k].loc[:, a] - F[k].loc[:, a].mean()) / F[k].loc[:, a].std(ddof=0) * d
                                  for k, (w, d) in sel.items() if k in F)
                          for a in order}, index=panel.index).replace([np.inf, -np.inf], np.nan)
ics = []
for d in composite.index[-60:]:
    if d not in fwd10.index:
        continue
    fv = composite.loc[d]; fr = fwd10.loc[d]
    m = fv.notna() & fr.notna()
    if m.sum() >= 8:
        ics.append(fv[m].rank(pct=True).corr(fr[m].rank(pct=True)))
ics = pd.Series(ics)
print('composite IC mean %.4f hits %.3f n=%d last %.4f' % (ics.mean(), (ics > 0).mean(), len(ics), ics.iloc[-1]))