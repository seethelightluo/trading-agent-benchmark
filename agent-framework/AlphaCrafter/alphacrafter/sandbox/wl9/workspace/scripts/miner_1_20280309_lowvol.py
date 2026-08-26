import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
for w in [10,20,60]:
 for h in [5,10]:
  a=[]; ds=[]
  for i in range(w,len(p)-h):
   f=-r.iloc[i-w:i].std(); y=p.iloc[i+h]/p.iloc[i]-1; z=pd.concat([f,y],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(p.index[i])
  a=np.array(a); print(w,h,len(a),round(a.mean(),6),round(a.mean()/a.std(),4),round((a>0).mean(),4), 'online',round(a[np.array(ds)>=pd.Timestamp('2026-07-16')].mean(),6))
