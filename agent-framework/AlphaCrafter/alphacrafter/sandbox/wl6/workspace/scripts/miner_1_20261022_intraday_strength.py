import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-10-21'); S={}
for a in A:
 d=pd.read_csv(f'{base}/{a}.csv'); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date').loc[:cut]
 # intraday close-open strength, volatility scaled, lagged only by using today's signal for next close return
 intr=d.close/d.open-1; vol=d.close.pct_change().rolling(20,min_periods=10).std(); S[a]=pd.DataFrame({'f':intr/vol,'c':d.close})
for h in [1,5,10]:
 R=[]
 for dt in sorted(set().union(*[set(x.index) for x in S.values()])):
  v=[]
  for x in S.values():
   if dt not in x.index: continue
   i=x.index.get_loc(dt)
   if i+h>=len(x): continue
   f=x.iloc[i].f; r=x.iloc[i+h].c/x.iloc[i].c-1
   if pd.notna(f) and pd.notna(r):v.append((f,r))
  if len(v)>=8:
   q=spearmanr(np.array(v)[:,0],np.array(v)[:,1]).statistic
   if pd.notna(q):R.append((dt,q,len(v)))
 z=pd.DataFrame(R,columns=['date','ic','n']).set_index('date'); q=z.ic
 print(h,'dates',len(q),'avg_n',z.n.mean(),'coverage',z.n.sum()/(len(q)*15),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
 for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-10-21')]:
  x=q.loc[lo:hi];print(lab,x.mean(),x.mean()/x.std(ddof=1),len(x))
