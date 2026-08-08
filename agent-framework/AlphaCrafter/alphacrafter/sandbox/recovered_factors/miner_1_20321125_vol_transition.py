import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 f='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(f); d['date']=pd.to_datetime(d.date); px[s]=d.set_index('date').close
p=pd.DataFrame(px).sort_index()
# volatility transition: inverse of recent/long vol, lagged one day
r=p.pct_change()
short=r.rolling(5).std(); long=r.rolling(40).std()
f=-(short/long) # prefer lower recent vol relative to baseline
# evaluate signal at t and forward cumulative return t+1..t+h
for h in [1,5,10,20]:
  vals=[]; ns=[]
  for i in range(len(p)-h):
   x=f.iloc[i]; y=p.iloc[i+1:i+h+1].iloc[-1]/p.iloc[i]-1
   z=pd.concat([x,y],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  a=np.array(vals); print('H',h,'dates',len(a),'meanN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
print('coverage',f.notna().sum().sum()/f.size,'turnover10',np.mean((f.rank(axis=1,pct=True)-f.shift(10).rank(axis=1,pct=True)).abs().mean(axis=1)))
# regimes
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2032')]:
 vals=[]
 for i in range(len(p)-10):
  dt=p.index[i].strftime('%Y')
  if lo<=dt<=hi:
   z=pd.concat([f.iloc[i],p.iloc[i+10]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals);print(lo,hi,len(a),np.mean(a) if len(a) else np.nan,np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan)
