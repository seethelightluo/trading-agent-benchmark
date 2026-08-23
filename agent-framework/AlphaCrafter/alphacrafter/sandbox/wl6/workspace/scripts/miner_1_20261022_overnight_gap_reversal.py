import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2026-10-21')
S={}
for a in assets:
 d=pd.read_csv(f'{base}/{a}.csv'); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); d=d.loc[:cutoff]
 # gap from prior close to today's open; negative gap is expected to rebound
 gap=d.open/d.close.shift(1)-1
 S[a]=pd.DataFrame({'f':-gap,'c':d.close,'o':d.open})
for h in [1,5,10]:
 rows=[]
 dates=sorted(set().union(*[set(x.index) for x in S.values()]))
 for dt in dates:
  v=[]
  for a,x in S.items():
   if dt not in x.index: continue
   i=x.index.get_loc(dt); end=i+h
   if end>=len(x): continue
   f=x.iloc[i].f; r=x.iloc[end].c/x.iloc[i].c-1
   if pd.notna(f) and pd.notna(r): v.append((f,r))
  if len(v)>=8:
   rows.append((dt,spearmanr(np.array(v)[:,0],np.array(v)[:,1]).statistic,len(v)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=z.ic
 print(h,'dates',len(z),'avg_n',z.n.mean(),'coverage',z.n.sum()/(len(z)*15),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0))
 for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-10-21')]:
  q=ic.loc[lo:hi]; print(label,round(q.mean(),5),round(q.mean()/q.std(ddof=1),5),len(q))
 # rank turnover
