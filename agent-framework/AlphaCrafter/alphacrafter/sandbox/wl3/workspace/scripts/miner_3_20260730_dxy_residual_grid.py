import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(p):
 d=pd.read_csv(p);d.date=pd.to_datetime(d.date);return d[d.date<='2026-07-15'].set_index('date')
m=L('../persistent/index_data/DXY.csv').close.pct_change()
for bw in [20,60,100]:
 for lb in [10,20,30,40]:
  rows=[]
  for s in U:
   d=L('../persistent/stock_data/'+s+'.csv');r=d.close.pct_change();z=pd.concat([r,m],axis=1,join='inner');z.columns=['r','m'];b=z.r.rolling(bw,min_periods=max(10,bw//2)).cov(z.m)/z.m.rolling(bw,min_periods=max(10,bw//2)).var();f=(r-b*m).rolling(lb,min_periods=max(5,lb//2)).sum();y=r.shift(-1);q=pd.concat([f,y],axis=1);q.columns=['f','y'];q['date']=q.index;rows.append(q.reset_index(drop=True))
  a=pd.concat(rows,ignore_index=True).dropna(); vals=[]
  for dt,g in a.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1:vals.append(spearmanr(g.f,g.y).statistic)
  o=pd.Series(vals).dropna();print(bw,lb,len(o),round(o.mean(),5),round(o.mean()/o.std(),5),round((o>0).mean(),4))
