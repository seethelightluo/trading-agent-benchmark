import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=4000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x['close'].astype(float)
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change(); med=r.median(axis=1); res=r.sub(med,axis=0)
fac=(p.pct_change(20).sub(p.pct_change(20).median(axis=1),axis=0))/res.rolling(20).std(); fac=fac.shift(1)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in fac.index:
  a=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ns.append(len(a))
 z=pd.Series(vals).dropna(); print(f'h={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
fr=p.shift(-10)/p-1; vals=[]
for dt in fac.index:
 a=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8: vals.append((dt,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
z=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
for n in [250,500]:
 q=z.tail(n).ic; print(f'recent{n} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f}')
print('coverage',fac.notna().mean().mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
