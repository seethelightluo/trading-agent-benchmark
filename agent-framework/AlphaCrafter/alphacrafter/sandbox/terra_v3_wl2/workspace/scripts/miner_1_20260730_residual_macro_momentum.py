import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv'; d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').close
px=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=px.pct_change(); dxy=load('DXY',1).reindex(px.index).ffill().pct_change()
# Macro-residual medium momentum: 20d return less rolling 60d DXY beta contribution; all inputs lagged at decision date.
beta=pd.DataFrame(index=px.index,columns=U,dtype=float)
for s in U: beta[s]=r[s].rolling(60,min_periods=40).cov(dxy)/dxy.rolling(60,min_periods=40).var()
raw=r.rolling(20,min_periods=15).sum(); fac=raw-beta.mul(dxy.rolling(20,min_periods=15).sum(),axis=0)
for h in [1,5,10]:
 vals=[]; ns=[]
 for i in range(len(px)-h):
  q=pd.concat([fac.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 a=np.array(vals); print('residual_macro_momentum',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
# regime slices daily
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 vals=[]
 for i in range(len(px)-1):
  if not (lo<=str(px.index[i].year)<=hi):continue
  q=pd.concat([fac.iloc[i],px.iloc[i+1]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 a=np.array(vals);print('regime',lo,hi,len(a),round(a.mean(),5),round(a.mean()/a.std(),5))
print('coverage',fac.notna().sum(axis=1).ge(8).mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
