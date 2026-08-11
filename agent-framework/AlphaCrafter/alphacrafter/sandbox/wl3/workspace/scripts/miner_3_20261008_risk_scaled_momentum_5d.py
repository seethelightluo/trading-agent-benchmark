import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-10-07'; allrows=[]; total=0
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date'); total+=len(x)
 r=x.close.pct_change(); f=r.rolling(5,min_periods=4).sum()/(r.rolling(15,min_periods=10).std()*np.sqrt(5)+1e-12); y=x.close.shift(-1)/x.close-1
 allrows.append(pd.DataFrame({'date':x.date,'symbol':s,'f':f,'y':y}))
a=pd.concat(allrows,ignore_index=True).dropna(); vals=[]; ns=[]
for d,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vals.append((d,spearmanr(g.f,g.y).statistic)); ns.append(len(g))
z=pd.DataFrame(vals,columns=['date','ic']).set_index('date'); q=z.ic
rank=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='symbol',values='rank')
print('candidate risk_scaled_momentum_5d cutoff',cut,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(len(a)/total,4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),6))
for name,lo,hi in [('early','2020','2022-12-31'),('mid','2023','2024-12-31'),('late','2025','2026-10-07')]:
 v=z.loc[lo:hi].ic; print('regime',name,len(v),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6) if len(v)>1 else np.nan)
for h in [3,5,10]:
 vv=[]
 for s in U:
  g=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date'); r=g.close.pct_change(); f=r.rolling(5,min_periods=4).sum()/(r.rolling(15,min_periods=10).std()*np.sqrt(5)+1e-12); g=g.assign(f=f,y=g.close.shift(-h)/g.close-1); allg=g[['date','f','y']].assign(symbol=s); 
  if s==U[0]: tmp=allg
  else: tmp=pd.concat([tmp,allg])
 aa=tmp.dropna()
 for d,g in aa.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vv.append(spearmanr(g.f,g.y).statistic)
 vv=np.array(vv); print('decay',h,len(vv),round(vv.mean(),6),round(vv.mean()/vv.std(ddof=1),6))
a[['date','symbol','f']].rename(columns={'f':'signal'}).to_csv('scripts/miner_3_20261008_risk_scaled_momentum_5d_signal.csv',index=False)
