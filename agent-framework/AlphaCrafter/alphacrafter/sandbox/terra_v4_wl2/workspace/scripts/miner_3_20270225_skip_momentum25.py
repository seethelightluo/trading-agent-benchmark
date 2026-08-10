import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def rd(s): return pd.read_csv('../persistent/index_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float)
# stock_data is benchmark tradable history
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U},axis=1).sort_index()
R=P.pct_change()
# Medium momentum with a 5-day skip to reduce short-term reversal contamination.
f=(P.shift(5)/P.shift(25)-1).shift(1)
out=[]
for h in [1,3,5,10]:
 y=P.shift(-h)/P-1; a=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-06-30'),('2026-07-01','2027-02-25')]:
   q=[]
   for d in f.loc[lo:hi].index:
    z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
    if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
   q=np.array(q); print('REG',lo,len(q),np.mean(q) if len(q) else np.nan,(np.mean(q)/np.std(q,ddof=1)) if len(q)>1 else np.nan)
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean(axis=1).mean(),'assets',P.shape[1],'dates',P.shape[0])
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('../persistent/factor_signals_miner_3_20270225_skip_momentum25.csv',index=False)
