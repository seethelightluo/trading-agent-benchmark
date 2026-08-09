import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:'2026-08-26']
 except:pass
# Volatility-managed quality: trailing 20d return divided by realized 20d volatility, but sign-independent defensive risk score
for mode in ['invvol','sharpe']:
 for k in [1]:
  rows=[]
  for s,x in D.items():
   r=x.close.pct_change(); vol=r.rolling(20).std()
   f=1/(vol+1e-12) if mode=='invvol' else r.rolling(20).sum()/(vol+1e-12)
   for i,dt in enumerate(x.index):
    if pd.notna(f.iloc[i]) and i+k<len(x):rows.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+k]/x.close.iloc[i]-1)))
  a=pd.DataFrame(rows,columns=['date','s','f','y']); q=[]
  for dt,g in a.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
  q=np.array(q);print(mode,'dates',len(q),'names',a.groupby('date').size().mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
