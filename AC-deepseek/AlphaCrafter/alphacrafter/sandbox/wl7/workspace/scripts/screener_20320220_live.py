import json, pandas as pd, numpy as np

cutoff = '2032-02-19'
assets = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

closes = {}
for a in assets:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df.columns = [c.strip() for c in df.columns]
    df = df[df['date'] <= cutoff].set_index('date')['close']
    closes[a] = df
C = pd.DataFrame(closes).dropna()
R = C.pct_change()
ew = C.mean(axis=1)

def macro(sym):
    df = pd.read_csv(f'../persistent/index_data/{sym}.csv')
    df.columns = [c.strip() for c in df.columns]
    df = df[df['date'] <= cutoff].set_index('date')['close']
    return df

dxy = macro('DXY'); eurusd = macro('EURUSD')
rdxy = dxy.pct_change(); reurusd = eurusd.pct_change()

S = {}
m20 = (C / C.shift(20) - 1).shift(5)
S['rel_mom_20d_skip5'] = m20.sub(m20.median(axis=1), axis=0)
cov = R.rolling(60).cov(ew); var = ew.rolling(60).var()
S['beta_ew_60d'] = cov.div(var, axis=0)
neg = R.clip(upper=0)
S['downside_vol_ratio_20'] = -(neg.rolling(20).std() / R.rolling(20).std())
S['max_ret_20d'] = R.rolling(20).max()
S['kurt_20d_skip5'] = R.rolling(20).kurt().shift(5)
cov2 = R.rolling(60).cov(reurusd); var2 = reurusd.rolling(60).var()
S['eurusd_beta_cond_60x20'] = (cov2.div(var2, axis=0)) * (eurusd/eurusd.shift(20)-1)
cov3 = R.rolling(60).cov(rdxy); var3 = rdxy.rolling(60).var()
S['dxy_beta_cond_60x20'] = -(cov3.div(var3, axis=0)) * (dxy/dxy.shift(20)-1)
corr_out = {}
rc = R.rolling(60).corr()
for a in assets:
    col = rc.xs(a, level=1)
    others = [x for x in assets if x != a]
    corr_out[a] = col[others].abs().mean(axis=1)
S['corr_ew_60'] = pd.DataFrame(corr_out)

common = R.index
S = {k: v.reindex(common) for k, v in S.items()}

names = list(S.keys())
flat = pd.DataFrame({k: S[k].stack() for k in names}).dropna()
cc = flat.corr()
print('=== pairwise |corr|>0.35 (last 120d) ===')
for i in range(len(names)):
    for j in range(i+1, len(names)):
        c = cc.iloc[i,j]
        if abs(c) > 0.35:
            print(f'{names[i]:22s}~{names[j]:22s}: {c:+.3f}')

fwd = C.shift(-10)/C - 1
print('\n=== live h10 IC (rank corr) ===')
rows=[]
for k in names:
    s = S[k].loc[common[:-10]]
    ic=[]
    for t in common[:-10]:
        row = s.loc[t].dropna()
        fr = fwd.loc[t].reindex(row.index).dropna()
        idx = row.index.intersection(fr.index)
        if len(idx)>=8:
            x = pd.Series(row.loc[idx]).rank()
            y = pd.Series(fr.loc[idx]).rank()
            ic.append(x.corr(y))
    ic=pd.Series(ic)
    rec=dict(ic60=ic.tail(60).mean(), ic120=ic.tail(120).mean(),
             ic240=ic.tail(240).mean(), cell=ic.mean(), n=len(ic))
    rows.append((k,rec))
    print(f'{k:22s} IC60={rec["ic60"]:+.4f} IC120={rec["ic120"]:+.4f} IC240={rec["ic240"]:+.4f} ICall={rec["cell"]:+.4f} n={rec["n"]}')

print('\n=== regime ===')
print('VIX last:', macro('VIX').tail(1).values[0])
print('SPX 20d:', (C['SPX'].iloc[-1]/C['SPX'].iloc[-21]-1))
print('DXY 20d:', (dxy.iloc[-1]/dxy.iloc[-21]-1))
print('mdx (mean asset 20d):', (ew.iloc[-1]/ew.iloc[-21]-1))
print('MA20 slope (equity mean):', (ew.iloc[-1]-ew.iloc[-20]))