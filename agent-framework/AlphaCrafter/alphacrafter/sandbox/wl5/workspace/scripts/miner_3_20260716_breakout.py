import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 except:pass
# 60d breakout distance, clipped and volatility-normalized: current close / prior 60d max - 1
# use as continuation signal, deliberately not same as 20d momentum
rec=[]
for s,x in D.items():
 c=x.close; r=c.pct_change(); f=c/c.shift(1).rolling(60).max()-1; vol=r.rolling(20).std(); f=f/(vol*np.sqrt(20)+1e-12)
 for i,dt in enumerate(x.index):
  if pd.notna(f.iloc[i]) and i+1<len(x):rec.append((dt,s,float(f.iloc[i]),float(c.iloc[i+1]/c.iloc[i]-1)))
a=pd.DataFrame(rec,columns=['date','s','f','y']); out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1:out.append(spearmanr(g.f,g.y).statistic)
z=np.array(out);print('idea=60d breakout/vol; dates',len(z),'avg names',a.groupby('date').size().mean(),'coverage',a.s.nunique()/15);print('1d IC %.6f ICIR %.6f hit %.4f std %.6f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0),z.std(ddof=1)))
for k in [5,10,20]:
 q=[]
 for s,x in D.items():
  c=x.close;r=c.pct_change();f=c/c.shift(1).rolling(60).max()-1;f=f/(r.rolling(20).std()*np.sqrt(20)+1e-12)
  for i,dt in enumerate(x.index):
   if pd.notna(f.iloc[i]) and i+k<len(x):q.append((dt,float(f.iloc[i]),float(c.iloc[i+k]/c.iloc[i]-1)))
 b=pd.DataFrame(q,columns=['date','f','y']);ic=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:ic.append(spearmanr(g.f,g.y).statistic)
 ic=np.array(ic);print(k,'d IC %.6f ICIR %.6f dates %d'%(ic.mean(),ic.mean()/ic.std(ddof=1),len(ic)))
