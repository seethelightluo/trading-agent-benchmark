import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
ds={}
for s in U:
 try:
  d=get_stock_daily_data(s,5000)
  if d is not None: ds[s]=d.set_index('date')['close'].astype(float)
 except Exception as e: print('SKIP',s,type(e).__name__)
px=pd.DataFrame(ds).sort_index().loc[:'2027-03-23']; r=px.pct_change(); neg=r.where(r<0); dd=neg.rolling(120,min_periods=15).std(); f=((px/px.shift(20)-1)/dd).shift(1)
print('range',px.index.min(),px.index.max(),'rows',len(px),'assets',len(ds))
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 a=pd.Series(dict(vals)); print(h,'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),4),'dates',len(a),'avgN',round(sum(pd.concat([f.loc[x],fr.loc[x]],axis=1).dropna().shape[0] for x in a.index)/len(a),2),'hit',round((a>0).mean(),4))
valid=f.notna(); ranks=f.rank(axis=1,pct=True); print('coverage',round(valid.sum().sum()/(len(valid)*len(ds)),4),'turn',round(ranks.diff().abs().mean(axis=1).dropna().mean(),4),'avg',round(valid.sum(axis=1).mean(),2))
fr=px.shift(-1)/px-1; q=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:q.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
q=pd.DataFrame(q,columns=['date','ic']).set_index('date')
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 x=q.loc[a:b].ic; print('regime',a,b,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1)*np.sqrt(252),4) if len(x)>1 else None)
f.stack().rename('signal').to_csv('scripts/miner_2_20270324_downside_sortino_signal.csv')
