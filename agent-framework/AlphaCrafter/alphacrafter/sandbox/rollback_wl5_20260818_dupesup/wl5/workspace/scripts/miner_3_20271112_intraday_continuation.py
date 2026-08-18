import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 except:pass
for h in [1,5,10,20]:
 rows=[]
 for s,x in D.items():
  f=x.close/x.open-1; y=x.close.shift(-h)/x.close-1
  for d in x.index:
   if d>=pd.Timestamp('2020-01-02') and pd.notna(f.loc[d]) and pd.notna(y.loc[d]):rows.append((d,s,float(f.loc[d]),float(y.loc[d])))
 a=pd.DataFrame(rows,columns=['d','s','f','y']); z=[];ns=[]
 for d,g in a[(a.d>='2020-01-01')&(a.d<='2027-11-12')].groupby('d'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic);ns.append(len(g))
 z=np.array(z);print('h',h,'dates',len(z),'avgN',np.mean(ns),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1),'hit',np.mean(z>0),'online',np.mean([v for d,v in zip(a.d,z) if d>=pd.Timestamp('2026-07-16')]) if False else 'see pooled')
