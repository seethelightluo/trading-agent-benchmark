import numpy as np, pandas as pd, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data/'; frames={}
for s in U:
 p=base+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); frames[s]=d
idx=sorted(set().union(*[set(x.index) for x in frames.values()]))
close=pd.DataFrame({s:frames[s]['close'].reindex(idx) for s in frames}); vol=pd.DataFrame({s:frames[s]['volume'].reindex(idx) for s in frames})
r3=close.pct_change(3); vr=(vol/vol.rolling(20,min_periods=10).median()).clip(0.5,3.0)
sig=(-r3*np.log(vr)).replace([np.inf,-np.inf],np.nan); sig[vr.isna()]=np.nan
rows=[]; artifact=sig.reindex(columns=U)
for i,dt in enumerate(close.index[:-1]):
 z=pd.concat([sig.loc[dt],close.iloc[i+1]/close.iloc[i]-1],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna(); mean=res.ic.mean(); sd=res.ic.std(ddof=1)
print('dates',len(res),'avg_n',res.n.mean(),'coverage',sig.notna().sum(axis=1).mean()/len(U),'IC',mean,'ICIR',mean/sd*np.sqrt(252),'hit',(res.ic>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-25')]:
 q=res.loc[a:b,'ic']; print('regime',a,b,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
for h in [5,10]:
 rr=[]
 for i,dt in enumerate(close.index[:-h]):
  z=pd.concat([sig.loc[dt],close.iloc[i+h]/close.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(rr).dropna(); print('horizon',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252))
out='scripts/miner_2_20270325_volume_reversal_signal.csv'; artifact.to_csv(out); print('artifact',out)
