import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:'2026-07-15']
for lb in [3,5,10,20]:
 r=px.pct_change(lb); rows=[]
 for i,dt in enumerate(px.index):
  if i+10>=len(px.index): continue
  x=r.iloc[i]; peer=x.median(); f=peer-x
  for h in [1,5,10]:
   fut=px.iloc[i+h]/px.iloc[i]-1; z=pd.concat([f,fut],axis=1).dropna()
   if len(z)>=8: rows.append([dt,h,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)])
 d=pd.DataFrame(rows,columns=['date','h','ic','n'])
 print('LB',lb)
 for h in [1,5,10]:
  q=d[d.h==h].ic; print(h,len(q),round(q.mean(),5),round(q.mean()/q.std(ddof=1),5),(q>0).mean(),round(d[d.h==h].n.mean(),2))
