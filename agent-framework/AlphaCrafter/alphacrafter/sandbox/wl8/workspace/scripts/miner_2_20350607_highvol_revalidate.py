import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
syms=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);px[s]=d.sort_values('date').set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); f=r.rolling(60).std().shift(1); ics=[];ns=[]
for i in range(1,len(p)-10):
 z=pd.concat([f.iloc[i].rename('s'),(p.iloc[i+10]/p.iloc[i]-1).rename('r')],axis=1).dropna()
 if len(z)>=8:
  x=spearmanr(z.s,z.r).statistic
  if np.isfinite(x):ics.append(x);ns.append(len(z))
print('dates',len(ics),'avgN',np.mean(ns),'IC',np.mean(ics),'ICIR',np.mean(ics)/(np.std(ics,ddof=1)+1e-12),'hit',np.mean(np.array(ics)>0),'coverage',len(ics)/(len(p)-11))
for h in [1,5,10,20]:
 a=[]
 for i in range(1,len(p)-h):
  z=pd.concat([f.iloc[i].rename('s'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.s,z.r).statistic)
 print('decay',h,np.mean(a),len(a))
for n in [365,750,1260]:print('recent',n,np.mean(ics[-n:]),np.mean(ics[-n:])/(np.std(ics[-n:],ddof=1)+1e-12))
