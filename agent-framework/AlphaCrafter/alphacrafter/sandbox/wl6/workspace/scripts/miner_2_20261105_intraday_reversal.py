import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; S={}
for a in A:
 p=f'{base}/{a}.csv'
 if not os.path.exists(p): continue
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 # fade completed day's open-to-close move; forecast next close-to-close return
 f=-(d.close/d.open-1); r=d.close.shift(-1)/d.close-1
 S[a]=pd.DataFrame({'f':f,'r':r})
for h in [1,3,5,10]:
 rows=[]
 for a in A:
  if a not in S: continue
  d=pd.read_csv(f'{base}/{a}.csv'); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); q=pd.DataFrame({'f':-(d.close/d.open-1),'r':d.close.shift(-h)/d.close-1})
  for dt,v in q.dropna().iterrows(): rows.append((dt,a,v.f,v.r))
 x=pd.DataFrame(rows,columns=['date','a','f','r']); ics=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8: ics.append(spearmanr(g.f,g.r).statistic); ns.append(len(g))
 z=np.array(ics); print(h,'dates',len(z),'avg_n',np.mean(ns),'coverage',sum(ns)/(len(z)*15),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
