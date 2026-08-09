import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').set_index('date')
for variant in ['large','signed']:
 rows=[]
 for s,x in D.items():
  gap=np.log(x.open/x.close.shift(1)); med=gap.abs().rolling(60).median().shift(1); vol=gap.rolling(20).std().shift(1)
  f=-gap/(vol+1e-8)
  if variant=='large': f=f.where(gap.abs()>med,0.0)
  y=np.log(x.close.shift(-1)/x.close)
  rows.append(pd.DataFrame({'date':x.index,'f':f,'y':y}).dropna())
 z=pd.concat(rows,ignore_index=True);out=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:out.append(spearmanr(g.f,g.y).statistic)
 a=np.array(out);print(variant,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
