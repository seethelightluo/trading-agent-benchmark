import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 d=pd.read_csv(f); d['date']=pd.to_datetime(d.date); px[a]=d.set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# lagged 3-day short-term reversal, risk scaled; signal at t predicts t+h return from t+1 onward
sig=-(p.pct_change(3).shift(1))/(r.rolling(20).std().shift(1).clip(lower=.003))
print('range',p.index.min().date(),p.index.max().date(),'assets',len(assets))
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1 # signal date t, h-day endpoint; all observable through t
 vals=[]; dates=[]; ns=[]
 for dt in p.index:
  x=sig.loc[dt]; y=fwd.loc[dt]
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 v=np.array(vals)
 print('H',h,'dates',len(v),'meanN',np.mean(ns),'IC',np.mean(v),'ICIR',np.mean(v)/np.std(v,ddof=1),'hit',np.mean(v>0),'se',np.std(v,ddof=1)/np.sqrt(len(v)))
# recent regimes
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
 z=[]
 for dt in dates if False else p.index:
  if not(str(dt.year)>=lo and str(dt.year)<=hi): continue
  x=sig.loc[dt]; y=fwd.loc[dt]; q=pd.concat([x,y],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 print(lo,hi,len(z),np.mean(z) if z else np.nan, (np.mean(z)/np.std(z,ddof=1)) if len(z)>1 else np.nan)
print('coverage',sig.notna().sum().sum()/sig.size,'turnover',sig.rank(axis=1,pct=True).diff(10).abs().mean().mean())
