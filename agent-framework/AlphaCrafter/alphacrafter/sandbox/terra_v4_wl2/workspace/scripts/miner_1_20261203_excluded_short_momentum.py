import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-03')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change()
# medium-term momentum measured over days t-25 through t-5, excluding the most recent week
f=p.shift(5)/p.shift(25)-1
for h in [1,5,10]:
 vals=[]; ns=[]; ds=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.y).statistic); ns.append(len(z)); ds.append(p.index[i])
 a=np.asarray(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 print('regimes',[(y,round(np.mean(a[[d.year==y for d in ds]]),5),len(a[[d.year==y for d in ds]])) for y in range(2020,2027)])
print('coverage',round(f.notna().sum().sum()/f.size,4),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'range',p.index.min(),p.index.max())
# provenance signal artifact for possible audit
pd.DataFrame([(dt,s,f.loc[dt,s]) for dt in f.index for s in U if pd.notna(f.loc[dt,s])],columns=['date','symbol','signal']).to_csv('../persistent/factor_signals_miner_1_20261203_exclshortmom.csv',index=False)
