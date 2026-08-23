import numpy as np,pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-04-02')
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date).dt.normalize(); D[s]=x.drop_duplicates('date').set_index('date').sort_index().close.astype(float).loc[:CUT]
P=pd.concat(D,axis=1).sort_index(); R=P.pct_change(); bench=R.mean(axis=1); rows=[]
for s in U:
 r=R[s]; beta=r.rolling(60,min_periods=40).cov(bench)/(bench.rolling(60,min_periods=40).var()+1e-12); resid=r-beta*bench
 f=-(resid.rolling(5,min_periods=5).sum()/(resid.rolling(20,min_periods=15).std()+1e-12)).shift(1)
 rows.append(pd.DataFrame({'date':P.index,'asset':s,'f':f.values,'fr':(P[s].shift(-1)/P[s]-1).values}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z); return len(z),round(np.mean(ns),2),round(z.mean(),5),round(z.mean()/z.std(ddof=1)*np.sqrt(252),4),round((z>0).mean(),4)
print('cutoff',CUT.date(),'assets',len(D),'dates',q.date.nunique(),'rows',len(q),'coverage',round(len(q)/(q.date.nunique()*len(U)),4))
for h in [1,5,10,20]:
 xx=[]
 for s in U: xx.append(pd.DataFrame({'date':P.index,'asset':s,'f':q[q.asset==s].set_index('date').f.reindex(P.index).values,'fr':P[s].shift(-h).div(P[s]).sub(1).values}))
 xx=pd.concat(xx,ignore_index=True).dropna(); print('horizon',h,stats(xx))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',round(float(r.diff().abs().mean().mean()),5))
q.to_csv('scripts/miner_2_20270402_residual_reversal_signal.csv',index=False)
