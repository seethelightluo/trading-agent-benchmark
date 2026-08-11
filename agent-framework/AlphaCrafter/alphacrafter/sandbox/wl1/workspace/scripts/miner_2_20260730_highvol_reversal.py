import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
def read(s,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv'
 return pd.read_csv(p,parse_dates=['date']).query('date<=@cut').set_index('date').close.sort_index()
p=pd.concat({s:read(s) for s in U},axis=1).sort_index(); r=p.pct_change()
vix=read('VIX',True).reindex(p.index).ffill(); vr=vix.pct_change()
# Novel idea: reversal only when volatility shock is elevated, with continuous cross-sectional VIX shock multiplier.
# Positive factor = prior 3d loss, amplified by trailing VIX level relative to 60d median.
base=-(p/p.shift(3)-1); mult=(vix/vix.rolling(60,min_periods=30).median()).clip(0.5,2.5)
f=base.mul(mult,axis=0)
ics=[]; ns=[]
for i in range(len(p)-1):
 q=pd.concat([f.iloc[i],(p.iloc[i+1]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: ics.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q))
a=np.array(ics); print('idea high-vol amplified reversal; horizon 1; dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'turnover',round(np.mean([np.mean(np.sign(f.iloc[i].fillna(0).values)!=np.sign(f.iloc[i-1].fillna(0).values)) for i in range(1,len(f))]),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 z=[]
 for i in range(len(p)-1):
  if lo<=p.index[i].year<=hi:
   q=pd.concat([f.iloc[i],(p.iloc[i+1]/p.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.y).statistic)
 z=np.array(z);print('regime',lo,hi,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
for h in [5,10]:
 z=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.y).statistic)
 z=np.array(z);print('decay',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
