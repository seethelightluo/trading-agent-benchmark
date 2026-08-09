import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-07-15'); D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date')
 except: pass
# Relative range position: close location within trailing 60-session high-low envelope, with short-term volatility normalization.
# Lower values (near lows) may mean rebound potential; test signed raw position and inverted position.
for mode in ['raw','inverse']:
 for h in [1,5,10]:
  rows=[]
  for s,x in D.items():
   c=x.close; hi=x.high.rolling(60,min_periods=45).max(); lo=x.low.rolling(60,min_periods=45).min()
   pos=(c-lo)/(hi-lo+1e-12)
   f=pos if mode=='raw' else -pos
   for i,dt in enumerate(x.index):
    if pd.notna(f.iloc[i]) and i+h<len(x): rows.append((dt,s,float(f.iloc[i]),float(c.iloc[i+h]/c.iloc[i]-1)))
  a=pd.DataFrame(rows,columns=['date','s','f','y']); ic=[]; ns=[]
  for dt,g in a.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ic.append(spearmanr(g.f,g.y).statistic); ns.append(len(g))
  z=np.array(ic); print(mode,h,'dates',len(z),'avg_names',np.mean(ns),'coverage',a.s.nunique()/15,'IC %.8f ICIR %.8f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0)))
  if h==1:
   ranks=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank').sort_index(); print('turnover',ranks.diff().abs().mean().mean())
