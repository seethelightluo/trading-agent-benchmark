import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2028-01-13')
p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); d['date']=pd.to_datetime(d.date); p[s]=d.set_index('date').close
p=pd.DataFrame(p).sort_index().loc[:cutoff]; r=p.pct_change()
short=r.rolling(5).sum().shift(1); vol=r.rolling(20).std().shift(1)
f=-(short-short.median(axis=1).values[:,None])/(vol+1e-8)
f=pd.DataFrame(f,index=p.index,columns=U)
for h in [1,5,10,20]:
 vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],p.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); m=q.ic.mean(); ir=m/q.ic.std(ddof=1)*np.sqrt(252)
 print(h,'dates',len(q),'mean_n',q.n.mean(),'coverage',q.n.mean()/15,'IC',round(m,6),'ICIR',round(ir,6),'hit',round((q.ic>0).mean(),4))
 for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2027-01','2028-01-13')]:
  z=q.loc[a:b]; print(' ',a,b,len(z),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252),4) if len(z)>2 else None)
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_2_20280113_residual_reversal_signal.csv',index=False)
