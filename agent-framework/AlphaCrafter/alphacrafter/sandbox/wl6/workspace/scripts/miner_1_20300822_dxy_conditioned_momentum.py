import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
dx=pd.read_csv('../persistent/index_data/DXY.csv');dx.date=pd.to_datetime(dx.date);dx=dx.set_index('date').close.astype(float).reindex(p.index).ffill()
# Trend momentum is favored only when DXY is below its 60d average (risk/liquidity supportive); otherwise damp it.
base=(p/p.shift(20)-1)/(r.rolling(30,min_periods=15).std()*np.sqrt(252))
reg=(dx/dx.rolling(60,min_periods=30).mean()).clip(0.8,1.2)
# continuous, causal macro multiplier, centered around 1; reduces exposure to momentum in tight-dollar regime
sig=base.mul((1.5-reg),axis=0)
print('range',p.index.min(),p.index.max(),'assets',len(p.columns),'rows',len(p),'macro_coverage',dx.notna().mean())
for h in [5,10,20]:
  a=[];ds=[];ns=[];turn=[];prev=None
  for i in range(len(p)-h):
   f=sig.iloc[i];y=p.iloc[i+h]/p.iloc[i]-1;z=pd.concat([f,y],axis=1).dropna()
   if len(z)>=8:
    ic=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
    if np.isfinite(ic):a.append(ic);ds.append(p.index[i]);ns.append(len(z))
   q=f.dropna().rank()
   if prev is not None:
    w=pd.concat([q,prev],axis=1).dropna()
    if len(w):turn.append((w.iloc[:,0]!=w.iloc[:,1]).mean())
   prev=q
  a=np.array(a);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),3),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),5),'hit',round((a>0).mean(),5),'turn',round(np.mean(turn),5))
  if h==10: z10=pd.Series(a,index=ds)
print('regimes10')
for yr,g in z10.groupby(z10.index.year):print(yr,len(g),round(g.mean(),6))
print('coverage',round(sig.notna().sum(axis=1).mean()/len(U),5))
