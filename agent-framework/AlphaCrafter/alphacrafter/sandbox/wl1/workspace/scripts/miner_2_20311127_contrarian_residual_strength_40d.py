import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT='2031-11-26'
px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); px[s]=d['close'].replace(0,np.nan)
P=pd.DataFrame(px).sort_index().loc[:CUT]; r=np.log(P).diff(); resid=r.sub(r.mean(axis=1),axis=0)
# Contrarian residual strength: fade 40d cumulative residual return, volatility normalized, lagged one day.
F=-(resid.rolling(40,min_periods=30).sum()/resid.rolling(60,min_periods=45).std()).shift(1); Y={h:np.log(P.shift(-h)/P) for h in [1,5,10,20]}; rows=[]; sig=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],Y[20].loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
  for a,v in F.loc[dt].dropna().items(): sig.append((dt,a,float(v)))
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('cutoff',CUT)
for h in [1,5,10,20]:
 q=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],Y[h].loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q); print('decay',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for name,m in [('2026-28',(res.index>='2026-01-01')&(res.index<'2029-01-01')),('2029-30',(res.index>='2029-01-01')&(res.index<'2031-01-01')),('2031',(res.index>='2031-01-01'))]:
 q=res.loc[m,'ic']; print(name,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
print('obs',len(res),'avgN',res.n.mean(),'coverage',res.n.sum()/(len(res)*15),'turnover',np.mean([np.mean(np.sign(F.iloc[i]).values!=np.sign(F.iloc[i-1]).values) for i in range(1,len(F))]))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20311127_contrarian_residual_strength_40d_signal.csv',index=False)
