import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2034-09-17')
def load(p): return pd.read_csv(p,parse_dates=['date']).set_index('date').close
px=pd.concat([load('../persistent/stock_data/'+s+'.csv').rename(s) for s in U],axis=1).sort_index(); px=px.loc[:END]
vix=load('../persistent/index_data/VIX.csv').reindex(px.index).ffill(); rv=(px.pct_change(5)-px.pct_change(20)).to_numpy(); stress=(vix.pct_change(10)>0).to_numpy(); f=rv*(1+0.5*stress[:,None])
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h); vals=[];dates=[]; ns=[]
 for i,dt in enumerate(px.index):
  a=f[i];b=y.loc[dt].to_numpy();ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: vals.append(spearmanr(a[ok],b[ok]).statistic);dates.append(dt);ns.append(ok.sum())
 z=pd.Series(vals,index=dates).dropna();print('h',h,'IC %.5f ICIR %.5f dates %d avgN %.2f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),len(z),np.mean(ns[:len(z)]),(z>0).mean()))
h=5;y=px.pct_change(h).shift(-h);vals=[];dates=[]
for i,dt in enumerate(px.index):
 a=f[i];b=y.loc[dt].to_numpy();ok=np.isfinite(a)&np.isfinite(b)
 if ok.sum()>=8: vals.append(spearmanr(a[ok],b[ok]).statistic);dates.append(dt)
z=pd.Series(vals,index=dates).dropna()
for a,b in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034')]:
 q=z.loc[a:b];print(a,b,'IC %.5f ICIR %.5f n %d'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),len(q)))
print('last',px.index.max(),'coverage',np.isfinite(f).mean())
