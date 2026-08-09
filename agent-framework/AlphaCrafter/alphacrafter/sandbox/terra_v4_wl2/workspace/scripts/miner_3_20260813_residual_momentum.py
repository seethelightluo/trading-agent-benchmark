import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cutoff=pd.Timestamp('2026-07-15');D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cutoff').set_index('date').close
 except:pass
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); m=r.median(axis=1)
# idiosyncratic medium momentum: 20d asset return minus rolling beta*20d market return, beta from 60 daily returns
for h in [1,5,10]:
 rows=[]
 for s in U:
  if s not in r:continue
  beta=r[s].rolling(60,min_periods=45).cov(m)/m.rolling(60,min_periods=45).var()
  f=(p[s]/p[s].shift(20)-1)-beta*(p.median(axis=1)/p.median(axis=1).shift(20)-1)
  for i,dt in enumerate(p.index):
   if pd.notna(f.iloc[i]) and i+h<len(p): rows.append((dt,s,float(f.iloc[i]),float(p[s].iloc[i+h]/p[s].iloc[i]-1)))
 a=pd.DataFrame(rows,columns=['date','s','f','y']);ics=[];ns=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:ics.append(spearmanr(g.f,g.y).statistic);ns.append(len(g))
 z=np.array(ics);print('h',h,'dates',len(z),'avg',np.mean(ns),'coverage',a.s.nunique()/15,'IC %.8f ICIR %.8f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0)))
