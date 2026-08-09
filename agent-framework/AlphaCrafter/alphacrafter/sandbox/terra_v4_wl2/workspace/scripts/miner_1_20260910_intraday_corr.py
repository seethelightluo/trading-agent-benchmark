import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=2500); D[s]=d.sort_values('date').set_index('date')
p=pd.concat({s:d['close'] for s,d in D.items()},axis=1).sort_index(); o=pd.concat({s:d['open'] for s,d in D.items()},axis=1).reindex(p.index); r=p.pct_change(); intr=p/o-1
f=-intr; old=-r.rolling(5).mean(); vals=pd.concat([f.stack(),old.stack()],axis=1).dropna(); print('pooled corr intraday vs rev5',vals.iloc[:,0].corr(vals.iloc[:,1],method='spearman'),len(vals))
for lab,mask in [('2020-22',p.index<'2023-01-01'),('2023-24',(p.index>='2023-01-01')&(p.index<'2025-01-01')),('2025-26',p.index>='2025-01-01')]:
 a=[]
 for i in range(1,len(p)-1):
  if not mask[i]:continue
  z=pd.concat([f.iloc[i],r.iloc[i+1]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a);print(lab,len(a),a.mean(),a.mean()/a.std(ddof=1))
