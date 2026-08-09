import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-07-15'); D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date')
 except Exception as e: print('missing',s,e)
# Volatility shock mean reversion: unusually high recent realized volatility versus its trailing baseline
# is assigned a negative score, hypothesizing transient shocks reverse cross-sectionally.
for w,base in [(5,60),(10,60),(5,120)]:
 rows=[]
 for s,x in D.items():
  r=x.close.pct_change(); v=r.rolling(w,min_periods=w).std(); b=v.rolling(base,min_periods=base//2).median(); f=-(v/(b+1e-12)-1); y=x.close.shift(-1)/x.close-1
  for dt in x.index:
   if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]): rows.append((dt,s,float(f.loc[dt]),float(y.loc[dt])))
 a=pd.DataFrame(rows,columns=['date','s','f','y']); z=[]; ns=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic);ns.append(len(g))
 z=np.asarray(z); print(w,base,'dates',len(z),'avg_names',np.mean(ns),'coverage',a.s.nunique()/15,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
 ranks=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank').sort_index(); print('turnover',ranks.diff().abs().mean().mean())
 for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15')]:
  q=[]
  for dt,g in a.groupby('date'):
   if lo<=str(dt.date())<=hi and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
  q=np.asarray(q); print(lo[:4],len(q),q.mean(),q.mean()/q.std(ddof=1))
