import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:'2026-08-26']
 except: pass
# Range shock reversal: unusually large daily true range, favor subsequent reversal; standardized range against trailing 20d.
for k in [1,5,10]:
 rows=[]
 for s,x in D.items():
  r=x.close.pct_change(); rng=(x.high-x.low)/x.close
  z=(rng-rng.rolling(20).mean())/(rng.rolling(20).std()+1e-12)
  f=-z
  for i,dt in enumerate(x.index):
   if pd.notna(f.iloc[i]) and i+k<len(x):rows.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+k]/x.close.iloc[i]-1)))
 a=pd.DataFrame(rows,columns=['date','s','f','y']);o=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:o.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
 q=np.array([v for _,v,_ in o]);print('horizon',k,'dates',len(q),'avg_names',np.mean([n for _,_,n in o]),'coverage',a.s.nunique()/15,'IC %.8f ICIR %.8f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0)))
 for yr in range(2020,2027):
  v=np.array([z for d,z,_ in o if d.year==yr]);
  if len(v):print(yr,len(v),round(v.mean(),5),round(v.mean()/v.std(ddof=1),4))
