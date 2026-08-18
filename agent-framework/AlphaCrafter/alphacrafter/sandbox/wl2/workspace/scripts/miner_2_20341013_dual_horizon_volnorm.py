import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2034-10-13'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); D[s]=x.loc[:end]
rows=[]
for s,x in D.items():
 r=x.close.pct_change(); v=r.rolling(20).std()*np.sqrt(252)
 sig=(x.close.pct_change(60)/v-.30*x.close.pct_change(20)/v).shift(1); fw=x.close.shift(-10)/x.close-1
 for dt in sig.index:
  if pd.notna(sig.loc[dt]) and pd.notna(fw.loc[dt]): rows.append((dt,s,sig.loc[dt],fw.loc[dt]))
a=pd.DataFrame(rows,columns=['date','symbol','sig','fwd']); ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8: ics.append(spearmanr(g.sig,g.fwd).statistic)
r=np.array(ics); print('dates',len(r),'avg_n',a.groupby('date').size().mean(),'coverage',len(a)/(len(D)*len(set(a.date)))); print('IC10',r.mean(),'ICIR',r.mean()/r.std(ddof=1),'hit',np.mean(r>0));
for h in [5,20]:
 rr=[]
 for dt,g in a.groupby('date'):
  z=[]
  for s in D:
   x=D[s]; ix=x.index.get_loc(dt) if dt in x.index else -1
   if ix>=0 and ix+h<len(x) and pd.notna(g.loc[g.symbol==s,'sig']).any(): z.append((g.loc[g.symbol==s,'sig'].iloc[0],x.close.iloc[ix+h]/x.close.iloc[ix]-1))
  if len(z)>=8: rr.append(spearmanr(*zip(*z)).statistic)
 rr=np.array(rr); print('h',h,'IC',rr.mean(),'ICIR',rr.mean()/rr.std(ddof=1),'dates',len(rr))
wide=a.pivot(index='date',columns='symbol',values='sig'); print('turnover',wide.rank(pct=True,axis=1).diff().abs().mean().mean()); print('period',a.date.min(),a.date.max()); a[['date','symbol','sig']].to_csv('../persistent/miner_2_20341013_dual_horizon_volnorm_signal.csv',index=False)
