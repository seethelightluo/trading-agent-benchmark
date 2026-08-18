import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym, macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+sym+'.csv'; d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').close
px=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=px.pct_change()
m=pd.concat({'vix':load('VIX',1).pct_change(),'dxy':load('DXY',1).pct_change(),'us10':load('US10Y').pct_change()},axis=1).reindex(px.index).ffill()
z=(m-m.rolling(60,min_periods=40).mean())/m.rolling(60,min_periods=40).std(); shock=(z.vix+z.dxy-z.us10)/3
beta=pd.DataFrame(index=px.index,columns=U,dtype=float)
for s in U: beta[s]=r[s].rolling(60,min_periods=40).cov(shock)/shock.rolling(60,min_periods=40).var()
fac=-beta
for h in [1,5,10]:
 a=[]; ns=[]
 for i in range(len(px)-h):
  q=pd.concat([fac.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 a=np.array(a);print('macro_resilience',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean())
rev=-r.rolling(5).sum(); bs=(beta.sub(beta.mean(axis=1),axis=0)).div(beta.std(axis=1),axis=0)
fac2=rev*(1+0.35*shock.abs().values[:,None]*(-bs))
for h in [1,5,10]:
 a=[]
 for i in range(len(px)-h):
  q=pd.concat([fac2.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 a=np.array(a);print('conditional_rev',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean())
