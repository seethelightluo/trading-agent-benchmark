import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); D[s]=x.close
P=pd.DataFrame(D).sort_index().loc[:'2028-02-24']; F=(P/P.shift(20)-1)-(P/P.shift(60)-1); H=10
obs=[]; ds=[]; ns=[]
for i in range(60,len(P)-H):
 z=pd.concat([F.iloc[i],P.iloc[i+H]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: obs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(P.index[i]); ns.append(len(z))
a=np.asarray(obs); dt=pd.DatetimeIndex(ds); print('dates',len(a),'avg_n',np.mean(ns),'coverage',F.notna().mean().mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'turnover',F.rank(pct=True).diff().abs().mean().mean())
for label,st,en in [('2020-21','2020-01-01','2022-01-01'),('2022-23','2022-01-01','2024-01-01'),('2024-25','2024-01-01','2026-01-01'),('2026-28','2026-01-01','2028-02-25')]:
 q=a[(dt>=pd.Timestamp(st))&(dt<pd.Timestamp(en))]; print(label,len(q),q.mean(),q.mean()/q.std(ddof=1))
