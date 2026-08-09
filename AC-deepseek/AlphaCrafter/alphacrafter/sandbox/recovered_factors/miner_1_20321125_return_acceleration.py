import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in syms}).sort_index(); r=p.pct_change()
# acceleration: recent 10d return minus preceding 10d return, lagged
f=r.rolling(10).sum()-r.shift(10).rolling(10).sum()
for h in [1,5,10,20]:
 a=[];ns=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);print(h,len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0))
print('cov',f.notna().sum().sum()/f.size,'turn',np.mean((f.rank(axis=1,pct=True)-f.shift(10).rank(axis=1,pct=True)).abs().mean(axis=1)))
for lo,hi in [('2024','2027'),('2028','2030'),('2031','2032')]:
 a=[]
 for i in range(len(p)-10):
  y=p.index[i].year
  if int(lo)<=y<=int(hi):
   z=pd.concat([f.iloc[i],p.iloc[i+10]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print(lo,len(a),a.mean(),a.mean()/a.std(ddof=1))
