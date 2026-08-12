import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={};vol={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date');px[a]=d.close.astype(float);vol[a]=d.volume.astype(float)
p=pd.DataFrame(px).sort_index();v=pd.DataFrame(vol).reindex(p.index);r=p.pct_change()
# volume-confirmed medium momentum: return weighted by own relative activity, all lagged
mom=p.pct_change(20); vr=v.rolling(20,min_periods=15).mean()/(v.rolling(60,min_periods=40).mean()+1e-12)
f=(mom*vr).shift(1)
for h in [1,3,5,10]:
 y=p.pct_change(h).shift(-h); rows=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(rows,columns=['d','n','ic']).set_index('d');x=q.ic
 print('h',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for nm,s in [('2020-22',q.loc['2020':'2022']),('2023-25',q.loc['2023':'2025']),('2026-27',q.loc['2026':'2027']),('2028',q.loc['2028':])]:
  z=s.ic;print(nm,len(z),round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('coverage',round(f.notna().sum().sum()/(len(p)*len(A)),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
