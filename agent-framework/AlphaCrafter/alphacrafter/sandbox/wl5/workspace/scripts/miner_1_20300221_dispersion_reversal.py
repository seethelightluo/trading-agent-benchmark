import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-02-20')
px={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}
p=pd.DataFrame(px).sort_index().dropna(); r=p.pct_change(); r5=p.pct_change(5); vol=r.rolling(20,min_periods=15).std(); disp=r5.std(axis=1).rolling(20,min_periods=15).mean(); baseline=disp.rolling(120,min_periods=60).median(); gate=(disp/baseline).clip(.5,2)
s=(-r5).sub((-r5).median(axis=1),axis=0).div(vol.replace(0,np.nan)).mul(gate,axis=0)
print('rows',len(p),'assets',len(U),'cut',cut.date())
for h in [3,5,10,20]:
 xs=[]; ns=[]
 for i in range(len(p)-h):
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: xs.append(q.f.corr(q.y,method='spearman'));ns.append(len(q))
 z=pd.Series(xs);print('TEST',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'coverage',round(np.mean(np.array(ns)/15),4))
print('turnover',s.rank(axis=1,pct=True).diff().abs().mean().mean(),'coverage',s.notna().mean().mean())
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20300221_dispersion_reversal_signal.csv',index=False);print('artifact rows',len(out))
for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2029-01-01'),('2029-01-02','2030-02-20')]:
 xs=[]
 for i in range(len(p)-10):
  if pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b):
   q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1:xs.append(q.f.corr(q.y,method='spearman'))
 z=pd.Series(xs);print('REGIME',a,b,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
