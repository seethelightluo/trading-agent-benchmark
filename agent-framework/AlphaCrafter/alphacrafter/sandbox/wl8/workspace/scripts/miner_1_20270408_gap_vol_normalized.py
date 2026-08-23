import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-04-08'); rows=[]
for a in A:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=end].sort_values('date')
 prev=x.close.shift(1); gap=x.open/prev-1; vol=x.close.pct_change().rolling(20).std(); f=gap/vol
 # fade gap, with a modest intraday close confirmation filter embedded as signed gap
 for i in range(len(x)-10):
  if pd.notna(f.iloc[i]) and pd.notna(x.close.iloc[i+1]) and pd.notna(x.close.iloc[i+5]) and pd.notna(x.close.iloc[i+10]): rows.append([x.date.iloc[i],a,f.iloc[i],x.close.iloc[i+1]/x.close.iloc[i]-1,x.close.iloc[i+5]/x.close.iloc[i]-1,x.close.iloc[i+10]/x.close.iloc[i]-1])
z=pd.DataFrame(rows,columns=['date','a','f','r1','r5','r10'])
for h in ['r1','r5','r10']:
 ic=[]
 for d,g in z.groupby('date'):
  if len(g)>=8:
   q=spearmanr(g.f,g[h]).statistic
   if np.isfinite(q):ic.append(q)
 ic=np.array(ic);print(h,'dates',len(ic),'n',len(z),'names/day',len(z)/z.date.nunique(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0),'coverage',len(z)/(15*z.date.nunique()))
print('annual')
for y,g in z.groupby(z.date.dt.year):
 q=[]
 for d,h in g.groupby('date'):
  if len(h)>=8:
   v=spearmanr(h.f,h.r1).statistic
   if np.isfinite(v):q.append(v)
 print(y,len(q),np.mean(q) if q else np.nan)
