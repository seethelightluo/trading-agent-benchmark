import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); return d.close
px=pd.concat({s:load(s) for s in U},axis=1).sort_index().loc[:'2026-07-15']; r=px.pct_change()
m=pd.concat({'vix':load('VIX',1).pct_change(),'dxy':load('DXY',1).pct_change(),'us10':load('US10Y').pct_change()},axis=1).reindex(px.index).ffill()
z=(m-m.rolling(60,min_periods=40).mean())/m.rolling(60,min_periods=40).std(); shock=(z.vix+z.dxy-z.us10)/3
beta=pd.DataFrame({s:r[s].rolling(60,min_periods=40).cov(shock)/shock.rolling(60,min_periods=40).var() for s in U})
for name,f in [('macro_resilience',-beta),('conditional_reversal',-r.rolling(5).sum()*(1+0.35*shock.abs().values[:,None]*(-(beta.sub(beta.mean(axis=1),axis=0)).div(beta.std(axis=1),axis=0))) )]:
 for h in [1,5,10]:
  vals=[]; ns=[]
  for i in range(len(px)-h):
   q=pd.concat([f.iloc[i],(px.iloc[i+h]/px.iloc[i]-1)],axis=1).dropna()
   if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
  a=np.array(vals); print(name,h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
