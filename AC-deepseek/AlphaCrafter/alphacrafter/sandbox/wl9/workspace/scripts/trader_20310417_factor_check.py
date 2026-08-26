"""Trader cycle validation @ 2031-04-17 (visible 2031-04-16).
Recompute selected ensemble factors on completed bars only, print regime
snapshot, signed composite z-score and short-term IC to confirm ensemble fit.
"""
import json
import numpy as np
import pandas as pd

VISIBLE = '2031-04-16'
order = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
P = {}
for s in order:
    try:
        df = pd.read_csv('../persistent/stock_data/%s.csv' % s)
    except Exception:
        df = pd.read_csv('../persistent/index_data/%s.csv' % s)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    P[s] = df['close']
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


def normz(s):
    s = s.astype(float)
    return (s - s.mean()) / s.std(ddof=0)


F = {}
F['mom_120d_skip5'] = panel / panel.shift(126) - 1
F['mom_10d_skip5'] = panel / panel.shift(15) - 1
F['bb_width_20d'] = 4 * panel.rolling(20).std() / panel.rolling(20).mean()
F['ac1_120d'] = R.rolling(120).apply(lambda x: x.autocorr() if x.std() > 0 else np.nan, raw=False)
F['skew_20d'] = R.rolling(20).skew()
F['kaufman_eff_20d'] = kaufman(panel, 20)
F['vol_z_20d'] = R.rolling(20).std() * np.sqrt(252)

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
c20 = R.rolling(20).corr(rdxy)
c60 = R.rolling(60).corr(rdxy)
F['dxy_corr_change_20_60'] = c20 - c60

ens = json.load(open('factor_ensemble.json'))
sel = {x['factor_id']: (x['weight'], x['direction']) for x in ens['selected_factors']}
print('=== visible date', VISIBLE, '=== VIX tail:', vix.tail(6).round(2).tolist())
print('=== 20d trailing return ===')
print((panel.iloc[-1] / panel.iloc[-21] - 1).sort_values(ascending=False).round(4).to_string())
print('\n=== 90d trailing return ===')
print((panel.iloc[-1] / panel.iloc[-91] - 1).sort_values(ascending=False).round(3).to_string())
print('\n=== composite signed z-score ===')
score = pd.Series({a: 0.0 for a in order})
zz = pd.DataFrame(index=order, dtype=float)
for name, (w, d) in sel.items():
    fv = F[name].iloc[-1]
    fz = normz(fv)
    zz[name] = fz
    score = score.add(w * fz * d, fill_value=np.nan)
print(score.sort_values(ascending=False).round(3).to_string())
print('\n=== per-factor cross-section rank (pct) at visible date ===')
print(zz.rank(axis=0, pct=True).round(2).to_string())
print('\n=== composite 10d fwd IC (trailing 40 visible dates) ===')
fwd10 = panel.shift(-10) / panel - 1
composite = pd.DataFrame({a: sum(w * normz(F[k].loc[:, a]) * d
                                  for k, (w, d) in sel.items() if k in F)
                          for a in order}, index=panel.index).replace([np.inf, -np.inf], np.nan)
ics = []
for d in composite.index[-40:]:
    if d not in fwd10.index:
        continue
    fv = composite.loc[d]
    fr = fwd10.loc[d]
    m = fv.notna() & fr.notna()
    if m.sum() >= 8:
        ics.append(fv[m].rank(pct=True).corr(fr[m].rank(pct=True)))
ics = pd.Series(ics)
if len(ics):
    print('composite IC mean %.4f hits %.3f n=%d last %.4f' % (ics.mean(), (ics > 0).mean(), len(ics), ics.iloc[-1]))
else:
    print('no IC computed')

print('\n=== per-factor trailing 10d IC (mean over last 40 visible dates) ===')
for name in sel:
    fv = F[name]
    fz = fv.apply(normz, axis=1) if len(fv) else fv
    fwd = panel.shift(-10) / panel - 1
    ic10 = []
    for d in fz.index[-40:]:
        if d not in fwd.index:
            continue
        a = fz.loc[d]
        b = fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= 8:
            ic10.append(a[m].rank(pct=True).corr(b[m].rank(pct=True)))
    ic10 = pd.Series(ic10)
    if len(ic10):
        print('%-22s ic_mean %+.4f  hits %.2f  n=%d  last %+.4f' % (
            name, ic10.mean(), (ic10 > 0).mean(), len(ic10), ic10.iloc[-1]))
    else:
        print('%-22s no ic' % name)