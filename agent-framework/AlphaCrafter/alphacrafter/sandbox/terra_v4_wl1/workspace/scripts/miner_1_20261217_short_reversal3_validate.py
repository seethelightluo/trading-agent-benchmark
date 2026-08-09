import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); D={}
for s in U:
 p=Path('../persistent/stock_data')/(s+'.csv'); x=pd.read_csv(p); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1,sort=True).sort_index(); F=-R.rolling(3,min_periods=3).sum()
# artifact is the signal available at each completed date
out=F.reset_index().rename(columns={'index':'date'}); out.to_csv('scripts/miner_1_20261217_short_reversal3_signal.csv',index=False)
Y=sum(R.shift(-k) for k in range(1,6)); qs=[]; dates=[]; ns=[]
for dt in R.index:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: qs.append(spearmanr(z.f,z.y).statistic); dates.append(dt); ns.append(len(z))
q=pd.Series(qs,index=dates); print('factor short_reversal_3d dates',len(q),'avg_n',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',F.notna().sum().sum()/(len(F)*len(U)),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean(),'period',R.index.min().date(),cut.date())
for name,mask in [('2020-22',(q.index<'2023-01-01')),('2023-24',((q.index>='2023-01-01')&(q.index<'2025-01-01'))),('2025-26',(q.index>='2025-01-01'))]:
 z=q[mask]; print('regime',name,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
for h in [1,5,10]:
 yy=sum(R.shift(-k) for k in range(1,h+1)); a=[]
 for dt in R.index:
  z=pd.concat([F.loc[dt],yy.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'dates',len(a),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1))
